"""Prepare causal MERL model rows from completed pose/feature batches.

Only the MERL development and validation splits are accepted.  Test data is
explicitly rejected so it cannot leak into model fitting or threshold tuning.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from merl_ml import (
    MODEL_SCHEMA_VERSION,
    MODEL_VERSION,
    attach_ground_truth,
    build_temporal_features,
    label_counts,
    model_column_groups,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GT_FRAMES = PROJECT_DIR / "evaluation" / "merl_ground_truth_frames.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "model_data" / MODEL_VERSION
ALLOWED_SPLITS = {"development", "validation"}

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare labelled causal feature rows for hybrid_v2."
    )
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--ground-truth-frames", type=Path, default=DEFAULT_GT_FRAMES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--split",
        choices=("development", "validation"),
        required=True,
        help="Expected MERL split; test is intentionally unavailable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-video caches for this preparation run.",
    )
    return parser.parse_args()

def successful_records(batch, expected_split):
    records = []
    for record in batch.get("videos", []):
        if record.get("status") != "success":
            continue
        split = str(record.get("split", ""))
        if split != expected_split:
            continue
        feature_csv = record.get("stages", {}).get("features", {}).get("output_csv")
        if not feature_csv:
            raise ValueError(
                f"{record.get('video_id')}: successful batch row has no feature CSV"
            )
        records.append((record, Path(feature_csv)))
    if not records:
        raise ValueError(
            f"Batch contains no successful {expected_split!r} feature records"
        )
    return records

def main():
    args = parse_args()
    batch_path = args.batch_manifest.resolve()
    ground_truth_path = args.ground_truth_frames.resolve()
    output_root = args.output_root.resolve()
    for path in (batch_path, ground_truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.split not in ALLOWED_SPLITS:
        raise ValueError("Test rows may not be prepared for training or tuning")

    batch = load_json(batch_path)
    records = successful_records(batch, args.split)
    ground_truth = pd.read_csv(
        ground_truth_path,
        dtype={"video_id": str},
        low_memory=False,
    )
    available_splits = set(ground_truth["split"].dropna().astype(str))
    if args.split not in available_splits:
        raise ValueError(f"Ground truth does not contain split {args.split!r}")

    run_dir = output_root / f"{args.split}_{batch.get('batch_run_id', 'batch')}"
    cache_dir = run_dir / "videos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    all_counts = {}
    expected_columns = None

    for position, (record, feature_path) in enumerate(records, start=1):
        video_id = str(record["video_id"])
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        output_path = cache_dir / f"{video_id}.pkl.gz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Prepared cache already exists: {output_path}; pass --overwrite"
            )
        print(f"[{position}/{len(records)}] Preparing {video_id}", flush=True)
        raw = pd.read_csv(feature_path, low_memory=False)
        temporal = build_temporal_features(raw)
        labelled = attach_ground_truth(temporal, ground_truth, video_id)
        numeric, categorical = model_column_groups(labelled)
        current_columns = numeric + categorical
        if expected_columns is None:
            expected_columns = current_columns
        elif current_columns != expected_columns:
            missing = sorted(set(expected_columns) - set(current_columns))
            extra = sorted(set(current_columns) - set(expected_columns))
            raise ValueError(
                f"{video_id}: model feature schema differs; missing={missing}, extra={extra}"
            )
        labelled.to_pickle(output_path, compression="gzip")
        counts = label_counts(labelled)
        for label, count in counts.items():
            all_counts[label] = all_counts.get(label, 0) + count
        prepared.append({
            "video_id": video_id,
            "subject_id": int(record["subject_id"]),
            "session_id": int(record["session_id"]),
            "split": args.split,
            "source_feature_csv": str(feature_path.resolve()),
            "prepared_rows": len(labelled),
            "label_counts": counts,
            "cache_path": str(output_path.resolve()),
        })

    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "pipeline_version": MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "role": "model_training" if args.split == "development" else "threshold_tuning",
        "split": args.split,
        "contains_test_data": False,
        "source_batch_manifest": str(batch_path),
        "source_batch_run_id": batch.get("batch_run_id"),
        "ground_truth_frames": str(ground_truth_path),
        "video_count": len(prepared),
        "row_count": int(sum(row["prepared_rows"] for row in prepared)),
        "label_counts": all_counts,
        "causal_temporal_context_sec": [0.5, 1.0],
        "numeric_feature_columns": numeric,
        "categorical_feature_columns": categorical,
        "videos": prepared,
    }
    manifest_path = run_dir / "prepared_dataset_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print("MERL model-data preparation complete.")
    print("Split:", args.split)
    print("Videos:", len(prepared))
    print("Rows:", manifest["row_count"])
    print("Saved manifest:", manifest_path)

if __name__ == "__main__":
    main()
