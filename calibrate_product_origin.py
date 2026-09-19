"""Calibrate adaptive product-origin projection using validation annotations."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from merl_ml import as_boolean
from product_origin_v2 import (
    DEFAULT_CALIBRATION,
    expected_shelf,
    load_product_zones,
    origin_decision,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = PROJECT_DIR / "annotations" / "manual_event_annotations.csv"
DEFAULT_ZONES = PROJECT_DIR / "frames" / "zones.json"
DEFAULT_OUTPUT = PROJECT_DIR / "models" / "hybrid_v2" / "product_origin_calibration.json"
MULTIPLIERS = (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40)
PRE_EXIT_WINDOWS = (0.40, 0.60, 0.80, 1.00)
AMBIGUITY_RATIOS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
MINIMUM_TOP_SHARES = (0.20, 0.25, 0.30, 0.35)
ADAPTIVE_WEIGHTS = (0.0, 0.10, 0.25, 0.50, 1.00)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune hybrid_v2 product-origin projection on validation only."
    )
    parser.add_argument("--validation-batch", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--zones-json", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)

def validation_features(batch):
    paths = {}
    for record in batch.get("videos", []):
        if record.get("status") != "success" or record.get("split") != "validation":
            continue
        value = record.get("stages", {}).get("features", {}).get("output_csv")
        if not value:
            raise ValueError(f"{record.get('video_id')}: missing feature output")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[str(record["video_id"])] = path
    if not paths:
        raise ValueError("Batch contains no successful validation feature outputs")
    return paths

def make_examples(annotations, feature_paths):
    pickup = annotations[
        (annotations["split"].astype(str) == "validation")
        & (annotations["label"].astype(str) == "pickup")
        & (~as_boolean(annotations.get("uncertain", pd.Series(False, index=annotations.index))))
    ].copy()
    pickup = pickup[pickup["video_id"].astype(str).isin(feature_paths)]
    examples = []
    cache = {}
    for _, annotation in pickup.iterrows():
        video_id = str(annotation["video_id"])
        wrist = str(annotation.get("wrist", "")).strip().lower()
        actual = str(annotation.get("product_track_id", "")).strip()
        if wrist not in {"left", "right"} or not actual or "_Product_" not in actual:
            continue
        if video_id not in cache:
            cache[video_id] = pd.read_csv(feature_paths[video_id], low_memory=False)
        table = cache[video_id]
        start = int(annotation["start_frame"])
        end = int(annotation["end_frame"])
        subset = table[(table["frame"] >= start) & (table["frame"] <= end)].copy()
        if subset.empty:
            continue
        examples.append({
            "annotation_id": annotation["annotation_id"],
            "video_id": video_id,
            "wrist": wrist,
            "actual": actual,
            "shelf": expected_shelf(actual),
            "subset": subset,
        })
    if not examples:
        raise ValueError("No usable validation pickup annotations were found")
    return examples

def predict_examples(examples, zones, calibration):
    rows = []
    for example in examples:
        decision = origin_decision(
            example["subset"], example["wrist"], zones, calibration
        )
        rows.append({
            "annotation_id": example["annotation_id"],
            "video_id": example["video_id"],
            "wrist": example["wrist"],
            "actual": example["actual"],
            "shelf": example["shelf"],
            "predicted": decision["candidate"],
            "alternative": decision["alternative"],
            "ambiguous": bool(decision["ambiguous"]),
            "confidence": decision["confidence"],
            "margin": decision["margin"],
            "correct": decision["candidate"] == example["actual"],
        })
    return pd.DataFrame(rows)

def accuracy(table):
    return float(table["correct"].mean()) if len(table) else 0.0

def main():
    args = parse_args()
    batch_path = args.validation_batch.resolve()
    annotations_path = args.annotations.resolve()
    zones_path = args.zones_json.resolve()
    output_path = args.output.resolve()
    for path in (batch_path, annotations_path, zones_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    batch = load_json(batch_path)
    if batch.get("requested_split") not in {"validation", "all"}:
        raise ValueError("Product-origin calibration must use a validation batch")
    feature_paths = validation_features(batch)
    annotations = pd.read_csv(annotations_path, dtype={"video_id": str})
    examples = make_examples(annotations, feature_paths)
    zones = load_product_zones(zones_path)

    shelves = sorted({row["shelf"] for row in examples if row["shelf"]})
    best_global = None
    for pre_exit in PRE_EXIT_WINDOWS:
        candidate_calibration = {
            **DEFAULT_CALIBRATION,
            "pre_exit_sec": pre_exit,
            "shelf_multipliers": {shelf: 1.0 for shelf in shelves},
            "shelf_adaptive_weights": {shelf: 0.0 for shelf in shelves},
        }
        predictions = predict_examples(examples, zones, candidate_calibration)
        candidate = (accuracy(predictions), -abs(pre_exit - 0.8))
        if best_global is None or candidate > best_global[0]:
            best_global = (candidate, candidate_calibration)

    calibration = dict(best_global[1])
    shelf_multipliers = {}
    shelf_adaptive_weights = {}
    for shelf in shelves:
        shelf_examples = [row for row in examples if row["shelf"] == shelf]
        best = None
        for multiplier in MULTIPLIERS:
            for adaptive_weight in ADAPTIVE_WEIGHTS:
                candidate_calibration = {
                    **calibration,
                    "shelf_multipliers": {shelf: multiplier},
                    "shelf_adaptive_weights": {shelf: adaptive_weight},
                }
                predictions = predict_examples(shelf_examples, zones, candidate_calibration)
                candidate = (
                    accuracy(predictions),
                    -adaptive_weight,
                    -abs(multiplier - 1.0),
                )
                if best is None or candidate > best[0]:
                    best = (candidate, multiplier, adaptive_weight)
        shelf_multipliers[shelf] = best[1]
        shelf_adaptive_weights[shelf] = best[2]
    calibration["shelf_multipliers"] = shelf_multipliers
    calibration["shelf_adaptive_weights"] = shelf_adaptive_weights

    ranking_calibration = {  # Rank once, then test each ambiguity setting.
        **calibration,
        "ambiguity_ratio": 1.01,
        "minimum_top_share": 0.0,
    }
    ranked_predictions = predict_examples(examples, zones, ranking_calibration)
    best_selective = None
    for ratio in AMBIGUITY_RATIOS:
        for minimum_share in MINIMUM_TOP_SHARES:
            predictions = ranked_predictions.copy()
            runner_ratio = 1.0 - pd.to_numeric(
                predictions["margin"], errors="coerce"
            ).fillna(0.0)
            top_share = pd.to_numeric(
                predictions["confidence"], errors="coerce"
            ).fillna(0.0)
            predictions["ambiguous"] = (
                (runner_ratio >= ratio) | (top_share < minimum_share)
            )
            covered = predictions[~predictions["ambiguous"]]
            coverage = len(covered) / len(predictions)
            covered_accuracy = accuracy(covered)
            objective = covered_accuracy + 0.25 * coverage if coverage >= 0.50 else coverage  # Accuracy matters only with useful coverage.
            candidate = (objective, covered_accuracy, coverage, ratio)
            if best_selective is None or candidate > best_selective[0]:
                best_selective = (
                    candidate,
                    {
                        **calibration,
                        "ambiguity_ratio": ratio,
                        "minimum_top_share": minimum_share,
                    },
                    predictions,
                )
    calibration = best_selective[1]
    predictions = best_selective[2]
    covered = predictions[~predictions["ambiguous"]]
    calibration.update({
        "schema_version": 1,
        "pipeline_version": "hybrid_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tuned_split": "validation",
        "test_data_used": False,
        "validation_batch_manifest": str(batch_path),
        "annotations": str(annotations_path),
        "zones_json": str(zones_path),
        "validation_pickups": len(predictions),
        "exact_candidate_accuracy": accuracy(predictions),
        "non_ambiguous_coverage": len(covered) / len(predictions),
        "accuracy_when_non_ambiguous": accuracy(covered),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, calibration)
    report_path = output_path.with_name(output_path.stem + "_validation_predictions.csv")
    predictions.to_csv(report_path, index=False, float_format="%.4f")
    print("Product-origin calibration complete.")
    print("Validation pickups:", len(predictions))
    print("Exact candidate accuracy:", f"{calibration['exact_candidate_accuracy']:.3f}")
    print("Non-ambiguous coverage:", f"{calibration['non_ambiguous_coverage']:.3f}")
    print("Accuracy when non-ambiguous:", f"{calibration['accuracy_when_non_ambiguous']:.3f}")
    print("Saved calibration:", output_path)
    print("Saved validation predictions:", report_path)

if __name__ == "__main__":
    main()
