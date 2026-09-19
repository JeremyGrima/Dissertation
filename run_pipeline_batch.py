"""Run the complete retail-behaviour pipeline over MERL splits.

Validation videos are the default.  Test videos require --confirm-frozen so
they are not accidentally used while thresholds are still being adjusted.
Every stage runs sequentially for one video before the next video starts.
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
POSE_OUTPUT_DIR = PROJECT_DIR / "pose_tests"
DEFAULT_DATASET_MANIFEST = (
    PROJECT_DIR / "evaluation" / "merl_video_manifest.csv"
)
DEFAULT_ZONES_JSON = PROJECT_DIR / "frames" / "zones.json"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "evaluation" / "batch_runs"

PIPELINE_SCRIPTS = {
    "pose": PROJECT_DIR / "zone_pose.py",
    "features": PROJECT_DIR / "behaviour_features.py",
    "classifier": PROJECT_DIR / "behaviour_classifier.py",
    "trained_classifier": PROJECT_DIR / "trained_behaviour_classifier.py",
    "emergence": PROJECT_DIR / "product_emergence_tracker.py",
    "product_interaction": PROJECT_DIR / "product_interaction.py",
}
DEFAULT_V2_MODEL_MANIFEST = (
    PROJECT_DIR / "models" / "hybrid_v2" / "hybrid_v2_model_manifest.json"
)
DEFAULT_V2_ORIGIN_CALIBRATION = (
    PROJECT_DIR / "models" / "hybrid_v2" / "product_origin_calibration.json"
)

LATEST_MANIFESTS = {
    "pose": POSE_OUTPUT_DIR / "latest_zone_pose_run.json",
    "features": POSE_OUTPUT_DIR / "latest_behaviour_features_run.json",
    "classifier": POSE_OUTPUT_DIR / "latest_behaviour_classifier_run.json",
    "emergence": POSE_OUTPUT_DIR / "latest_product_emergence_run.json",
    "product_interaction": (
        POSE_OUTPUT_DIR / "latest_product_interaction_run.json"
    ),
}

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def resolved(path):
    return str(Path(path).resolve())

def same_path(left, right):
    if not left or not right:
        return False
    return Path(left).resolve() == Path(right).resolve()

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def safe_identifier(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError("The batch run ID must contain a letter or number.")
    return cleaned

def source_hashes(
    zones_json,
    classifier_mode="rules_v1",
    model_manifest=None,
    origin_calibration=None,
):
    source_files = dict(PIPELINE_SCRIPTS)
    source_files["zones"] = Path(zones_json).resolve()
    source_files["pose_model"] = PROJECT_DIR / "yolov8s-pose.pt"
    if classifier_mode == "hybrid_v2" and model_manifest:
        model_manifest = Path(model_manifest).resolve()
        source_files["v2_model_manifest"] = model_manifest
        if model_manifest.is_file():
            model_value = load_json(model_manifest).get("model_path")
            if model_value and Path(model_value).is_file():
                source_files["v2_model"] = Path(model_value).resolve()
    if origin_calibration and Path(origin_calibration).is_file():
        source_files["v2_origin_calibration"] = Path(origin_calibration).resolve()
    return {name: sha256_file(path) for name, path in source_files.items()}

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Process MERL videos through pose, features, behaviour "
            "classification and product interaction in order."
        )
    )
    parser.add_argument(
        "--split",
        choices=("development", "train", "validation", "test", "all"),
        default="validation",
        help="Dataset split to process (default: validation).",
    )
    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help=(
            "Required before processing test data; confirms that zones, "
            "rules and thresholds will not be tuned using test results."
        ),
    )
    parser.add_argument(
        "--classifier-mode",
        choices=("rules_v1", "hybrid_v2"),
        default="rules_v1",
        help="Keep rules_v1 as the baseline or use the trained hybrid_v2 path.",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_V2_MODEL_MANIFEST,
        help="Validation-tuned hybrid_v2 model manifest.",
    )
    parser.add_argument(
        "--origin-calibration",
        type=Path,
        default=DEFAULT_V2_ORIGIN_CALIBRATION,
        help="Validation-only adaptive product-origin calibration.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("features", "full"),
        default="full",
        help="Use 'features' to prepare the 60 development videos for training.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--zones-json", type=Path, default=DEFAULT_ZONES_JSON
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--batch-run-id",
        default=None,
        help="Optional identifier for a new batch run.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume an existing batch_manifest.json and skip successes. "
            "Use the same split/settings as the original command."
        ),
    )
    parser.add_argument(
        "--recover-downstream",
        action="store_true",
        help=(
            "With --resume, reuse completed pose and feature outputs for "
            "failed videos and rerun only classifier/emergence/product stages."
        ),
    )
    parser.add_argument(
        "--reuse-upstream-batch",
        type=Path,
        default=None,
        help=(
            "Create a new batch while reusing pose and feature outputs from "
            "a completed batch. Only downstream classifier/emergence/product "
            "stages are run. Upstream hashes and video IDs must match."
        ),
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Only process this video ID; may be supplied more than once.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit the ordered video list, mainly for smoke testing.",
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Processed-frame limit per video; 0 means the full video.",
    )
    parser.add_argument(
        "--render-videos",
        action="store_true",
        help="Render all four annotated videos per input video.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed video instead of continuing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the selected videos without running any models.",
    )
    return parser.parse_args()

def read_dataset_rows(path, split, selected_ids, max_videos):
    if split == "all":
        wanted_splits = {"validation", "test"}
    elif split in {"development", "train"}:
        wanted_splits = {"development"}
    else:
        wanted_splits = {split}
    with open(path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    rows = [row for row in rows if row.get("split") in wanted_splits]
    if selected_ids:
        selected = set(selected_ids)
        rows = [row for row in rows if row.get("video_id") in selected]
        missing = sorted(selected - {row.get("video_id") for row in rows})
        if missing:
            raise ValueError(
                "Video IDs are absent from the selected split: "
                + ", ".join(missing)
            )

    rows.sort(
        key=lambda row: (
            {"development": 0, "validation": 1, "test": 2}.get(
                row["split"], 3
            ),
            int(row["subject_id"]),
            int(row["session_id"]),
        )
    )
    if max_videos is not None:
        if max_videos < 1:
            raise ValueError("--max-videos must be at least 1")
        rows = rows[:max_videos]
    return rows

def validate_files(args, rows):
    required = [
        args.dataset_manifest,
        args.zones_json,
        PROJECT_DIR / "yolov8s-pose.pt",
    ]
    required.extend(PIPELINE_SCRIPTS.values())
    required.extend(Path(row["video_path"]) for row in rows)
    if args.stop_after == "full" and args.classifier_mode == "hybrid_v2":
        required.append(args.model_manifest)
        if args.split in {"test", "all"}:
            required.append(args.origin_calibration)
    if args.reuse_upstream_batch is not None:
        required.append(args.reuse_upstream_batch)
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required pipeline files are missing:\n" + "\n".join(missing)
        )
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.stop_after == "features" and args.render_videos:
        print(
            "Note: --stop-after features renders only the pose video; no "
            "behaviour/product videos will be created.",
            flush=True,
        )
    if args.stop_after == "full" and args.classifier_mode == "hybrid_v2":
        model_manifest = load_json(args.model_manifest)
        if model_manifest.get("pipeline_version") != "hybrid_v2":
            raise ValueError("--model-manifest is not a hybrid_v2 model")
        if model_manifest.get("test_data_used_for_training_or_tuning") is not False:
            raise ValueError("The v2 model provenance does not exclude test data")
        model_path = Path(model_manifest.get("model_path") or "")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if args.split in {"test", "all"} and not model_manifest.get(
            "frozen_for_test"
        ):
            raise ValueError(
                "The hybrid_v2 model is not frozen. Train/tune on development "
                "and validation, then create it with --freeze before test use."
            )

def validate_upstream_reuse(args, rows):
    if args.reuse_upstream_batch is None:
        return None
    source_path = args.reuse_upstream_batch.resolve()
    source = load_json(source_path)
    if source.get("status") != "complete":
        raise ValueError("--reuse-upstream-batch must reference a complete batch")
    source_records = {
        str(record.get("video_id")): record
        for record in source.get("videos", [])
        if record.get("status") == "success"
    }
    missing = [row["video_id"] for row in rows if row["video_id"] not in source_records]
    if missing:
        raise ValueError(
            "The upstream batch lacks successful rows for: " + ", ".join(missing)
        )
    for row in rows:
        source_record = source_records[row["video_id"]]
        if str(source_record.get("split")) != str(row["split"]):
            raise ValueError(f"Upstream split mismatch for {row['video_id']}")
        stages = source_record.get("stages", {})
        pose = stages.get("pose", {})
        features = stages.get("features", {})
        if not Path(pose.get("frame_csv") or "").is_file():
            raise FileNotFoundError(
                f"Reusable pose CSV is missing for {row['video_id']}"
            )
        if not Path(features.get("output_csv") or "").is_file():
            raise FileNotFoundError(
                f"Reusable feature CSV is missing for {row['video_id']}"
            )
        if not same_path(pose.get("video_path"), row["video_path"]):
            raise ValueError(f"Reusable source video mismatch for {row['video_id']}")
        if args.render_videos and not Path(pose.get("annotated_video") or "").is_file():
            raise ValueError(
                "--render-videos cannot reuse this upstream batch because its "
                f"pose video is absent ({row['video_id']})"
            )

    current = source_hashes(
        args.zones_json,
        args.classifier_mode,
        args.model_manifest,
        args.origin_calibration,
    )
    original = source.get("source_sha256", {})
    immutable = ("pose", "features", "zones", "pose_model")
    changed = [
        name for name in immutable
        if original.get(name) != current.get(name)
    ]
    if changed:
        raise ValueError(
            "Cannot reuse upstream outputs because these inputs changed: "
            + ", ".join(changed)
        )
    source_settings = source.get("settings", {})
    if int(source_settings.get("stride", args.stride)) != int(args.stride):
        raise ValueError("The reused upstream batch used a different stride")
    if int(source_settings.get("max_processed_frames", args.max_frames)) != int(
        args.max_frames
    ):
        raise ValueError("The reused upstream batch used a different frame limit")
    return source

def run_command(command, log_file):
    printable = subprocess.list2cmdline([str(part) for part in command])
    print(f"\n> {printable}", flush=True)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n> {printable}\n")
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def require_latest(stage, checks=None):
    path = LATEST_MANIFESTS[stage]
    if not path.is_file():
        raise FileNotFoundError(
            f"{stage} did not create its expected latest manifest: {path}"
        )
    manifest = load_json(path)
    for key, expected in (checks or {}).items():
        actual = manifest.get(key)
        if key.endswith(("_csv", "_video", "_path")) or key in {
            "input_csv",
            "video_path",
            "input_features",
            "input_behaviour_events",
        }:
            matches = same_path(actual, expected)
        else:
            matches = str(actual) == str(expected)
        if not matches:
            raise RuntimeError(
                f"{stage} manifest mismatch for {key}: "
                f"expected {expected!r}, found {actual!r}"
            )
    return manifest

def process_downstream(row, args, batch_id, log_file, pose, features):
    python = Path(sys.executable).resolve()
    video_id = row["video_id"]
    video_path = Path(row["video_path"]).resolve()     #PoseID is kept to trace the exact reused feature CSV.
    pose_run_id = str(
        pose.get("run_id")
        or features.get("pose_run_id")
        or f"{video_id}_{batch_id}"
    )
    skip_video = [] if args.render_videos else ["--skip-video"]
    classifier_initial = None
    if args.classifier_mode == "rules_v1":
        classifier_command = [
            python,
            PIPELINE_SCRIPTS["classifier"],
            "--input-csv",
            features["output_csv"],
        ]
        if args.render_videos:
            classifier_command.extend(
                ["--input-video", pose["annotated_video"]]
            )
        else:
            classifier_command.append("--skip-video")
        run_command(classifier_command, log_file)
        classifier = require_latest(
            "classifier",
            {
                "pose_run_id": pose_run_id,
                "input_feature_csv": features["output_csv"],
            },
        )
    else:
        classifier_command = [
            python,
            PIPELINE_SCRIPTS["trained_classifier"],
            "--input-csv",
            features["output_csv"],
            "--model-manifest",
            args.model_manifest,
            "--zones-json",
            args.zones_json,
            "--pose-run-id",
            pose_run_id,
            "--stage",
            "initial",
            "--skip-video",
        ]
        if args.origin_calibration.is_file():
            classifier_command.extend(
                ["--origin-calibration", args.origin_calibration]
            )
        if args.split in {"test", "all"}:
            classifier_command.append("--require-frozen")
        run_command(classifier_command, log_file)
        classifier_initial = require_latest(
            "classifier",
            {
                "pose_run_id": pose_run_id,
                "input_feature_csv": features["output_csv"],
                "classification_stage": "initial",
            },
        )
        classifier = classifier_initial

    emergence_command = [
        python,
        PIPELINE_SCRIPTS["emergence"],
        "--features",
        features["output_csv"],
        "--behaviour-events",
        classifier["event_output_csv"],
        "--raw-video",
        video_path,
    ]
    if args.render_videos:
        emergence_input = (
            pose["annotated_video"]
            if args.classifier_mode == "hybrid_v2"
            else classifier["behaviour_video"]
        )
        emergence_command.extend(["--input-video", emergence_input])
    else:
        emergence_command.append("--skip-video")
    run_command(emergence_command, log_file)
    emergence = require_latest(
        "emergence",
        {
            "pose_run_id": pose_run_id,
            "input_features": features["output_csv"],
            "input_behaviour_events": classifier["event_output_csv"],
        },
    )

    if args.classifier_mode == "hybrid_v2":
        final_command = [
            python,
            PIPELINE_SCRIPTS["trained_classifier"],
            "--input-csv",
            features["output_csv"],
            "--emergence-features",
            emergence["augmented_feature_csv"],
            "--model-manifest",
            args.model_manifest,
            "--zones-json",
            args.zones_json,
            "--pose-run-id",
            pose_run_id,
            "--stage",
            "final",
        ]
        if args.origin_calibration.is_file():
            final_command.extend(
                ["--origin-calibration", args.origin_calibration]
            )
        if args.render_videos:
            final_command.extend(
                ["--input-video", emergence["annotated_video"]]
            )
        else:
            final_command.append("--skip-video")
        if args.split in {"test", "all"}:
            final_command.append("--require-frozen")
        run_command(final_command, log_file)
        classifier = require_latest(
            "classifier",
            {
                "pose_run_id": pose_run_id,
                "input_feature_csv": features["output_csv"],
                "classification_stage": "final",
            },
        )

    product_command = [
        python,
        PIPELINE_SCRIPTS["product_interaction"],
        "--behaviour-events",
        classifier["event_output_csv"],
        "--features",
        emergence["augmented_feature_csv"],
    ]
    if args.render_videos:
        product_input = (
            classifier["behaviour_video"]
            if args.classifier_mode == "hybrid_v2"
            else emergence["annotated_video"]
        )
        product_command.extend(
            ["--input-video", product_input]
        )
    else:
        product_command.append("--skip-video")
    run_command(product_command, log_file)
    product = require_latest(
        "product_interaction",
        {
            "pose_run_id": pose_run_id,
            "input_behaviour_events": classifier["event_output_csv"],
            "input_features": emergence["augmented_feature_csv"],
        },
    )

    result = {
        "pose": pose,
        "features": features,
        "classifier": classifier,
        "emergence": emergence,
        "product_interaction": product,
    }
    if classifier_initial is not None:
        result["classifier_initial"] = classifier_initial
    return result

def process_video(row, args, batch_id, log_file):
    python = Path(sys.executable).resolve()
    video_id = row["video_id"]
    video_path = Path(row["video_path"]).resolve()
    pose_run_id = f"{video_id}_{batch_id}"
    skip_video = [] if args.render_videos else ["--skip-video"]

    pose_command = [
        python,
        PIPELINE_SCRIPTS["pose"],
        "--video",
        video_path,
        "--zones-json",
        args.zones_json.resolve(),
        "--run-id",
        pose_run_id,
        "--stride",
        str(args.stride),
        "--max-frames",
        str(args.max_frames),
        *skip_video,
    ]
    run_command(pose_command, log_file)
    pose = require_latest(
        "pose",
        {"run_id": pose_run_id, "video_path": video_path},
    )

    feature_command = [
        python,
        PIPELINE_SCRIPTS["features"],
        "--input-csv",
        pose["frame_csv"],
        "--zones-json",
        args.zones_json.resolve(),
    ]
    run_command(feature_command, log_file)
    features = require_latest(
        "features", {"input_csv": pose["frame_csv"]}
    )
    if args.stop_after == "features":
        return {"pose": pose, "features": features}
    return process_downstream(
        row, args, batch_id, log_file, pose, features
    )

def reusable_pose_and_features(row, batch_id):
    """Find successful upstream outputs left by an interrupted/failed run."""
    pose_run_id = f"{row['video_id']}_{batch_id}"
    pose_manifest_path = POSE_OUTPUT_DIR / f"zone_pose_run_{pose_run_id}.json"
    if not pose_manifest_path.is_file():
        raise FileNotFoundError(
            f"Reusable pose manifest not found: {pose_manifest_path}"
        )
    pose = load_json(pose_manifest_path)
    if str(pose.get("run_id")) != pose_run_id:
        raise RuntimeError(f"Pose manifest run mismatch for {row['video_id']}")
    if not same_path(pose.get("video_path"), row["video_path"]):
        raise RuntimeError(f"Pose video mismatch for {row['video_id']}")
    pose_csv = Path(pose.get("frame_csv") or "")
    if not pose_csv.is_file():
        raise FileNotFoundError(f"Reusable pose CSV not found: {pose_csv}")

    candidates = sorted(
        POSE_OUTPUT_DIR.glob(f"behaviour_features_{pose_run_id}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for feature_manifest_path in candidates:
        features = load_json(feature_manifest_path)
        feature_csv = Path(features.get("output_csv") or "")
        if (
            str(features.get("pose_run_id")) == pose_run_id
            and same_path(features.get("input_csv"), pose_csv)
            and feature_csv.is_file()
        ):
            return pose, features
    raise FileNotFoundError(
        f"No reusable feature manifest/CSV found for {pose_run_id}"
    )

def recover_video_downstream(row, args, batch_id, log_file):
    pose, features = reusable_pose_and_features(row, batch_id)
    print(
        "Reusing pose/features for downstream recovery:",
        row["video_id"],
        flush=True,
    )
    print("  Pose CSV:", pose["frame_csv"], flush=True)
    print("  Feature CSV:", features["output_csv"], flush=True)
    return process_downstream(
        row, args, batch_id, log_file, pose, features
    )

def reuse_video_downstream(row, args, batch_id, log_file, source_batch):
    source_record = next(
        record
        for record in source_batch["videos"]
        if str(record.get("video_id")) == str(row["video_id"])
        and record.get("status") == "success"
    )
    stages = source_record["stages"]
    pose = stages["pose"]
    features = stages["features"]
    print(
        "Reusing completed pose/features from batch "
        f"{source_batch.get('batch_run_id')}: {row['video_id']}",
        flush=True,
    )
    print("  Pose CSV:", pose["frame_csv"], flush=True)
    print("  Feature CSV:", features["output_csv"], flush=True)
    return process_downstream(
        row, args, batch_id, log_file, pose, features
    )

def write_video_summary(path, videos):
    fields = [
        "video_id",
        "subject_id",
        "session_id",
        "split",
        "status",
        "video_path",
        "pose_frame_csv",
        "behaviour_frame_csv",
        "behaviour_event_csv",
        "product_track_csv",
        "product_event_csv",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in videos:
            stages = record.get("stages", {})
            writer.writerow(
                {
                    "video_id": record.get("video_id"),
                    "subject_id": record.get("subject_id"),
                    "session_id": record.get("session_id"),
                    "split": record.get("split"),
                    "status": record.get("status"),
                    "video_path": record.get("video_path"),
                    "pose_frame_csv": stages.get("pose", {}).get(
                        "frame_csv"
                    ),
                    "behaviour_frame_csv": stages.get(
                        "classifier", {}
                    ).get("frame_output_csv"),
                    "behaviour_event_csv": stages.get(
                        "classifier", {}
                    ).get("event_output_csv"),
                    "product_track_csv": stages.get(
                        "product_interaction", {}
                    ).get("track_output_csv"),
                    "product_event_csv": stages.get(
                        "product_interaction", {}
                    ).get("event_output_csv"),
                    "error": record.get("error"),
                }
            )

def create_manifest(args, batch_id, batch_dir, rows):
    return {
        "schema_version": 1,
        "batch_run_id": batch_id,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "requested_split": args.split,
        "selected_video_count": len(rows),
        "selected_video_ids": [row["video_id"] for row in rows],
        "dataset_manifest": resolved(args.dataset_manifest),
        "zones_json": resolved(args.zones_json),
        "output_directory": resolved(batch_dir),
        "frozen_configuration_confirmed": bool(args.confirm_frozen),
        "settings": {
            "stride": args.stride,
            "max_processed_frames": args.max_frames,
            "full_video": args.max_frames <= 0,
            "render_videos": bool(args.render_videos),
            "stop_after": args.stop_after,
            "classifier_mode": args.classifier_mode,
            "model_manifest": (
                resolved(args.model_manifest)
                if args.classifier_mode == "hybrid_v2" else None
            ),
            "origin_calibration": (
                resolved(args.origin_calibration)
                if args.classifier_mode == "hybrid_v2"
                and args.origin_calibration.is_file() else None
            ),
            "reuse_upstream_batch": (
                resolved(args.reuse_upstream_batch)
                if args.reuse_upstream_batch is not None else None
            ),
            "python_executable": resolved(sys.executable),
        },
        "source_sha256": source_hashes(
            args.zones_json,
            args.classifier_mode,
            args.model_manifest,
            args.origin_calibration,
        ),
        "videos": [],
    }

def main():
    args = parse_args()
    args.dataset_manifest = args.dataset_manifest.resolve()
    args.zones_json = args.zones_json.resolve()
    args.output_root = args.output_root.resolve()
    args.model_manifest = args.model_manifest.resolve()
    args.origin_calibration = args.origin_calibration.resolve()
    if args.reuse_upstream_batch is not None:
        args.reuse_upstream_batch = args.reuse_upstream_batch.resolve()
    if args.recover_downstream and not args.resume:
        raise SystemExit("--recover-downstream requires --resume.")
    if args.recover_downstream and args.stop_after != "full":
        raise SystemExit("--recover-downstream requires --stop-after full.")
    if args.reuse_upstream_batch is not None and args.recover_downstream:
        raise SystemExit(
            "--reuse-upstream-batch and --recover-downstream are mutually exclusive."
        )
    if args.reuse_upstream_batch is not None and args.stop_after != "full":
        raise SystemExit("--reuse-upstream-batch requires --stop-after full.")

    if (
        args.split in {"test", "all"}
        and not args.confirm_frozen
        and not args.dry_run
    ):
        raise SystemExit(
            "Refusing to process test data without --confirm-frozen. "
            "Tune only on validation, freeze zones/rules/thresholds, then "
            "run the test split once."
        )

    rows = read_dataset_rows(
        args.dataset_manifest,
        args.split,
        args.video_id,
        args.max_videos,
    )
    validate_files(args, rows)
    reused_upstream = validate_upstream_reuse(args, rows)
    if not rows:
        raise SystemExit("No videos matched the requested selection.")

    print(
        f"Selected {len(rows)} video(s) in this order: "
        + ", ".join(row["video_id"] for row in rows)
    )
    if args.dry_run:
        print("Dry run complete; no pipeline stages were executed.")
        return

    if args.resume:
        manifest_path = args.resume.resolve()
        manifest = load_json(manifest_path)
        batch_id = str(manifest["batch_run_id"])
        batch_dir = manifest_path.parent
        if manifest.get("requested_split") != args.split:
            raise ValueError(
                "The resumed manifest was created for split "
                f"{manifest.get('requested_split')!r}, not {args.split!r}."
            )
        selected_ids = [row["video_id"] for row in rows]
        if manifest.get("selected_video_ids") != selected_ids:
            raise ValueError(
                "The resumed command selects a different video list. Use "
                "the same --video-id/--max-videos options as the first run."
            )
        expected_settings = dict(manifest.get("settings", {}))
        expected_settings.setdefault("reuse_upstream_batch", None)  # Older manifests imply the same null setting.
        current_settings = {
            "stride": args.stride,
            "max_processed_frames": args.max_frames,
            "full_video": args.max_frames <= 0,
            "render_videos": bool(args.render_videos),
            "stop_after": args.stop_after,
            "classifier_mode": args.classifier_mode,
            "model_manifest": (
                resolved(args.model_manifest)
                if args.classifier_mode == "hybrid_v2" else None
            ),
            "origin_calibration": (
                resolved(args.origin_calibration)
                if args.classifier_mode == "hybrid_v2"
                and args.origin_calibration.is_file() else None
            ),
            "reuse_upstream_batch": (
                resolved(args.reuse_upstream_batch)
                if args.reuse_upstream_batch is not None else None
            ),
            "python_executable": resolved(sys.executable),
        }
        if expected_settings != current_settings:
            raise ValueError(
                "The resumed command changes pipeline settings. Re-run with "
                "the original stride/max-frames/render-videos options."
            )
        if reused_upstream is not None:
            original_upstream_hash = manifest.get(
                "upstream_reuse", {}
            ).get("source_manifest_sha256")
            if original_upstream_hash != sha256_file(args.reuse_upstream_batch):
                raise ValueError(
                    "The reused upstream batch manifest changed after this "
                    "downstream batch began."
                )
        current_hashes = source_hashes(
            args.zones_json,
            args.classifier_mode,
            args.model_manifest,
            args.origin_calibration,
        )
        original_hashes = manifest.get("source_sha256", {})
        if args.recover_downstream:
            immutable_upstream = ("pose", "features", "zones", "pose_model")
            changed_upstream = [
                name for name in immutable_upstream
                if original_hashes.get(name) != current_hashes.get(name)
            ]
            if changed_upstream:
                raise ValueError(
                    "Cannot reuse pose/features because upstream inputs changed: "
                    + ", ".join(changed_upstream)
                )
            manifest.setdefault("recovery", {}).update({
                "mode": "reuse_pose_and_features",
                "started_at": now_iso(),
                "source_sha256": current_hashes,
            })
        elif original_hashes != current_hashes:
            raise ValueError(
                "A pipeline script, zone file or pose model changed since "
                "this batch began. Start a new batch instead of mixing "
                "configurations in one manifest."
            )
        manifest["status"] = "running"
        manifest["updated_at"] = now_iso()
    else:
        batch_id = safe_identifier(
            args.batch_run_id
            or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        batch_dir = args.output_root / batch_id
        manifest_path = batch_dir / "batch_manifest.json"
        if manifest_path.exists():
            raise FileExistsError(
                f"Batch already exists; use --resume {manifest_path}"
            )
        batch_dir.mkdir(parents=True, exist_ok=False)
        manifest = create_manifest(args, batch_id, batch_dir, rows)
        if reused_upstream is not None:
            manifest["upstream_reuse"] = {
                "source_batch_manifest": resolved(args.reuse_upstream_batch),
                "source_batch_run_id": reused_upstream.get("batch_run_id"),
                "source_manifest_sha256": sha256_file(args.reuse_upstream_batch),
                "reused_stages": ["pose", "features"],
                "downstream_only": True,
            }

    completed_ids = {
        record["video_id"]
        for record in manifest.get("videos", [])
        if record.get("status") == "success"
    }
    write_json(manifest_path, manifest)

    for position, row in enumerate(rows, start=1):
        video_id = row["video_id"]
        if video_id in completed_ids:
            print(f"[{position}/{len(rows)}] Skipping completed {video_id}")
            continue

        print(f"\n[{position}/{len(rows)}] Processing {video_id}", flush=True)
        manifest["videos"] = [
            old
            for old in manifest.get("videos", [])
            if old.get("video_id") != video_id
        ]
        record = {
            "video_id": video_id,
            "subject_id": int(row["subject_id"]),
            "session_id": int(row["session_id"]),
            "split": row["split"],
            "video_path": resolved(row["video_path"]),
            "status": "running",
            "started_at": now_iso(),
            "completed_at": None,
            "log_path": resolved(batch_dir / "logs" / f"{video_id}.log"),
            "stages": {},
            "error": None,
        }
        manifest["videos"].append(record)
        manifest["updated_at"] = now_iso()
        write_json(manifest_path, manifest)

        log_file = Path(record["log_path"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            if args.recover_downstream:
                record["stages"] = recover_video_downstream(
                    row, args, batch_id, log_file
                )
            elif reused_upstream is not None:
                record["stages"] = reuse_video_downstream(
                    row, args, batch_id, log_file, reused_upstream
                )
            else:
                record["stages"] = process_video(
                    row, args, batch_id, log_file
                )
            record["status"] = "success"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"FAILED {video_id}: {record['error']}", flush=True)
        finally:
            record["completed_at"] = now_iso()
            manifest["updated_at"] = now_iso()
            write_json(manifest_path, manifest)
            write_video_summary(batch_dir / "video_runs.csv", manifest["videos"])

        if record["status"] == "failed" and args.fail_fast:
            break

    selected_ids = {row["video_id"] for row in rows}
    selected_records = [
        record
        for record in manifest["videos"]
        if record.get("video_id") in selected_ids
    ]
    succeeded = sum(r.get("status") == "success" for r in selected_records)
    failed = sum(r.get("status") == "failed" for r in selected_records)
    manifest["status"] = (
        "complete" if succeeded == len(rows) else "partial_failure"
    )
    manifest["completed_at"] = now_iso()
    manifest["updated_at"] = now_iso()
    manifest["summary"] = {
        "selected": len(rows),
        "succeeded": succeeded,
        "failed": failed,
    }
    if args.recover_downstream:
        manifest.setdefault("recovery", {})["completed_at"] = now_iso()
    write_json(manifest_path, manifest)
    write_video_summary(batch_dir / "video_runs.csv", manifest["videos"])

    print("\nBatch pipeline finished.")
    print("Status:", manifest["status"])
    print("Succeeded:", succeeded)
    print("Failed:", failed)
    print("Batch manifest:", manifest_path)
    print("Video summary:", batch_dir / "video_runs.csv")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
