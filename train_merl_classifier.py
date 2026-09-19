"""Train and validation-tune the hybrid_v2 two-stage MERL classifier."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ImportError as error:
    raise SystemExit(
        "hybrid_v2 training requires scikit-learn and joblib. Install them "
        "into the dissertation Python environment with: python -m pip "
        "install scikit-learn==1.6.1 joblib"
    ) from error

from merl_ml import (
    BEHAVIOURS,
    MODEL_SCHEMA_VERSION,
    MODEL_VERSION,
    RANDOM_SEED,
    balanced_sample_weights,
    event_f1,
    infer_sampling,
    load_cache_manifest,
    mask_runs,
    smooth_probability,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GT_EVENTS = PROJECT_DIR / "evaluation" / "merl_ground_truth_events.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "models" / MODEL_VERSION
THRESHOLDS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27, 0.32, 0.38, 0.45)
SMOOTHING_SEC = (0.13, 0.27, 0.40, 0.60)
MIN_DURATIONS = {
    "reach_to_shelf": (0.13, 0.20, 0.30, 0.40),
    "retract_from_shelf": (0.13, 0.20, 0.30, 0.40),
    "hand_in_shelf": (0.20, 0.30, 0.50, 0.70),
    "inspect_product": (0.40, 0.60, 0.80, 1.20),
    "inspect_shelf": (0.50, 0.80, 1.20, 1.60),
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the two-stage hybrid_v2 MERL behaviour model."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--ground-truth-events", type=Path, default=DEFAULT_GT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="hybrid_v2")
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Mark the validation-tuned model as frozen and eligible for test use.",
    )
    return parser.parse_args()

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)

def validate_manifest(manifest, expected_split, path):
    if manifest.get("split") != expected_split:
        raise ValueError(
            f"{path} is {manifest.get('split')!r}; expected {expected_split!r}"
        )
    if manifest.get("contains_test_data") is not False:
        raise ValueError(f"Refusing a dataset that may contain test rows: {path}")
    if manifest.get("pipeline_version") != MODEL_VERSION:
        raise ValueError(f"Dataset was not prepared for {MODEL_VERSION}: {path}")

def load_prepared(manifest):
    tables = []
    for video in manifest.get("videos", []):
        path = Path(video["cache_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pd.read_pickle(path, compression="gzip")
        table["video_id"] = str(video["video_id"])
        tables.append(table)
    if not tables:
        raise ValueError("Prepared dataset has no per-video tables")
    return pd.concat(tables, ignore_index=True)

def make_preprocessor(numeric_columns, categorical_columns):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)],
        remainder="drop",
        sparse_threshold=0.0,
    )

def make_estimator(numeric_columns, categorical_columns, max_iterations):
    return Pipeline([
        ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
        (
            "classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=max_iterations,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=15,
                random_state=RANDOM_SEED,
            ),
        ),
    ])

def class_probability(model, table, positive_class):
    probabilities = model.predict_proba(table)
    classes = list(model.named_steps["classifier"].classes_)
    if positive_class not in classes:
        return np.zeros(len(table), dtype=float)
    return probabilities[:, classes.index(positive_class)]

def combined_probabilities(stage1, stage2, table):
    event_probability = class_probability(stage1, table, 1)
    conditional = stage2.predict_proba(table)
    classes = [str(value) for value in stage2.named_steps["classifier"].classes_]
    result = {"event_probability": event_probability}
    for behaviour in BEHAVIOURS:
        if behaviour in classes:
            result[behaviour] = (
                event_probability * conditional[:, classes.index(behaviour)]
            )
        else:
            result[behaviour] = np.zeros(len(table), dtype=float)
    return result

def gt_event_records(events, video_id, behaviour):
    subset = events[
        (events["video_id"].astype(str) == str(video_id))
        & (events["behaviour"] == behaviour)
    ]
    return subset[["start_frame", "end_frame"]].to_dict("records")

def score_configuration(validation, probabilities, events, behaviour, config):
    all_ground_truth = []
    all_predictions = []
    for video_id, indices in validation.groupby("video_id", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        group = validation.loc[positions].sort_values("frame")
        ordered_positions = group.index.to_numpy(dtype=int)
        _, stride, sample_sec = infer_sampling(group)
        smoothing_rows = max(1, int(round(config["smoothing_sec"] / sample_sec)))
        smoothed = smooth_probability(
            probabilities[behaviour][ordered_positions], smoothing_rows
        )
        runs = mask_runs(
            smoothed >= config["threshold"],
            group["frame"].to_numpy(),
            fps=float(stride / sample_sec),
            min_duration_sec=config["min_duration_sec"],
            max_gap_sec=config["max_gap_sec"],
        )
        for run in runs:
            run["video_id"] = str(video_id)
        ground_truth = gt_event_records(events, video_id, behaviour)
        for row in ground_truth:
            row["video_id"] = str(video_id)
        video_number = int(str(video_id).split("_")[0]) * 10 + int(str(video_id).split("_")[1])
        offset = video_number * 1_000_000  # Stops matches from crossing video boundaries.
        all_ground_truth.extend({
            "start_frame": int(row["start_frame"]) + offset,
            "end_frame": int(row["end_frame"]) + offset,
        } for row in ground_truth)
        all_predictions.extend({
            "start_frame": int(row["start_frame"]) + offset,
            "end_frame": int(row["end_frame"]) + offset,
        } for row in runs)
    return event_f1(all_ground_truth, all_predictions, threshold=0.30)

def tune_decoder(validation, probabilities, events):
    tuned = {}
    report_rows = []
    for behaviour in BEHAVIOURS:
        best = None
        for smoothing_sec in SMOOTHING_SEC:
            for threshold in THRESHOLDS:
                for minimum in MIN_DURATIONS[behaviour]:
                    config = {
                        "threshold": threshold,
                        "smoothing_sec": smoothing_sec,
                        "min_duration_sec": minimum,
                        "max_gap_sec": 0.20,
                    }
                    metrics = score_configuration(
                        validation, probabilities, events, behaviour, config
                    )
                    candidate = {**config, **metrics}
                    if best is None or (
                        candidate["f1"], candidate["recall"], candidate["precision"], candidate["threshold"]
                    ) > (
                        best["f1"], best["recall"], best["precision"], best["threshold"]
                    ):
                        best = candidate
        tuned[behaviour] = best
        report_rows.append({"behaviour": behaviour, **best})
        print(
            f"Validation {behaviour}: F1={best['f1']:.3f}, "
            f"P={best['precision']:.3f}, R={best['recall']:.3f}, "
            f"threshold={best['threshold']:.2f}, "
            f"smooth={best['smoothing_sec']:.2f}s, "
            f"minimum={best['min_duration_sec']:.2f}s",
            flush=True,
        )
    return tuned, pd.DataFrame(report_rows)

def main():
    args = parse_args()
    training_path = args.training_data.resolve()
    validation_path = args.validation_data.resolve()
    events_path = args.ground_truth_events.resolve()
    output_dir = args.output_dir.resolve()
    for path in (training_path, validation_path, events_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    training_manifest = load_cache_manifest(training_path)
    validation_manifest = load_cache_manifest(validation_path)
    validate_manifest(training_manifest, "development", training_path)
    validate_manifest(validation_manifest, "validation", validation_path)
    training_ids = {row["video_id"] for row in training_manifest["videos"]}
    validation_ids = {row["video_id"] for row in validation_manifest["videos"]}
    if training_ids & validation_ids:
        raise ValueError("Training and validation manifests share video IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading prepared development rows...", flush=True)
    training = load_prepared(training_manifest)
    print("Loading prepared validation rows...", flush=True)
    validation = load_prepared(validation_manifest)
    numeric_columns = list(training_manifest["numeric_feature_columns"])
    categorical_columns = list(training_manifest["categorical_feature_columns"])
    if numeric_columns != validation_manifest["numeric_feature_columns"]:
        raise ValueError("Training/validation numeric feature schemas differ")
    if categorical_columns != validation_manifest["categorical_feature_columns"]:
        raise ValueError("Training/validation categorical feature schemas differ")
    model_columns = numeric_columns + categorical_columns

    print("Fitting Stage 1: behaviour versus background...", flush=True)
    stage1 = make_estimator(numeric_columns, categorical_columns, args.max_iterations)
    stage1.fit(
        training[model_columns],
        training["target_is_behaviour"].astype(int),
        classifier__sample_weight=balanced_sample_weights(
            training["target_is_behaviour"].astype(str)
        ),
    )

    stage2_rows = training[training["stage2_eligible"].astype(bool)].copy()
    missing_classes = set(BEHAVIOURS) - set(stage2_rows["target_label"].astype(str))
    if missing_classes:
        raise ValueError("Stage 2 training is missing: " + ", ".join(sorted(missing_classes)))
    print("Fitting Stage 2: MERL behaviour type...", flush=True)
    stage2 = make_estimator(numeric_columns, categorical_columns, args.max_iterations)
    stage2.fit(
        stage2_rows[model_columns],
        stage2_rows["target_label"].astype(str),
        classifier__sample_weight=balanced_sample_weights(stage2_rows["target_label"]),
    )

    print("Predicting validation rows and tuning temporal decoder...", flush=True)
    probabilities = combined_probabilities(stage1, stage2, validation[model_columns])
    ground_truth_events = pd.read_csv(events_path, dtype={"video_id": str})
    ground_truth_events = ground_truth_events[
        ground_truth_events["split"].astype(str) == "validation"
    ].copy()
    tuned, tuning_report = tune_decoder(
        validation.reset_index(drop=True), probabilities, ground_truth_events
    )

    model_path = output_dir / f"{args.run_id}_merl_model.joblib"
    manifest_path = output_dir / f"{args.run_id}_model_manifest.json"
    tuning_path = output_dir / f"{args.run_id}_validation_tuning.csv"
    bundle = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "pipeline_version": MODEL_VERSION,
        "stage1_model": stage1,
        "stage2_model": stage2,
        "numeric_feature_columns": numeric_columns,
        "categorical_feature_columns": categorical_columns,
        "decoder": tuned,
        "behaviours": BEHAVIOURS,
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(bundle, model_path, compress=3)
    tuning_report.to_csv(tuning_path, index=False, float_format="%.6f")
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "pipeline_version": MODEL_VERSION,
        "model_run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen" if args.freeze else "validation_tuned_not_frozen",
        "frozen_for_test": bool(args.freeze),
        "test_data_used_for_training_or_tuning": False,
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "training_data_manifest": str(training_path),
        "training_data_sha256": sha256(training_path),
        "validation_data_manifest": str(validation_path),
        "validation_data_sha256": sha256(validation_path),
        "ground_truth_events": str(events_path),
        "training_videos": len(training_ids),
        "validation_videos": len(validation_ids),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "stage2_training_rows": len(stage2_rows),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "classifier": "two_stage_hist_gradient_boosting",
        "stage1": "behaviour_vs_background",
        "stage2": "five_merl_behaviours",
        "causal_temporal_context_sec": [0.5, 1.0],
        "class_weighting": "balanced_sample_weights",
        "event_tiou_threshold": 0.30,
        "decoder": tuned,
        "validation_mean_class_f1": float(tuning_report["f1"].mean()),
        "validation_tuning_csv": str(tuning_path.resolve()),
        "random_seed": RANDOM_SEED,
        "python_packages": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json_atomic(manifest_path, manifest)
    print("MERL classifier training complete.")
    print("Model:", model_path)
    print("Manifest:", manifest_path)
    print("Validation mean class F1:", f"{manifest['validation_mean_class_f1']:.3f}")
    if not args.freeze:
        print("Model is validation-tuned but not frozen; test execution remains blocked.")

if __name__ == "__main__":
    main()
