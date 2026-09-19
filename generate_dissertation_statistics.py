"""Generate reproducible dissertation statistics from a frozen batch run.

The script reads existing evaluation outputs and annotations.  It does not
change model predictions, zones, thresholds, or the batch manifest.  It writes
one JSON data package which is consumed by the statistics workbook builder.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_pipeline as evaluation

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH = (
    PROJECT_DIR
    / "evaluation"
    / "hybrid_v2_test"
    / "hybrid_v2_test_v1"
    / "batch_manifest.json"
)
DEFAULT_BASELINE_BATCH = (
    PROJECT_DIR
    / "evaluation"
    / "hybrid_test"
    / "hybrid_test_v1"
    / "batch_manifest.json"
)
DEFAULT_VALIDATION_SUMMARY = (
    PROJECT_DIR
    / "evaluation"
    / "hybrid_v2_validation"
    / "hybrid_v2_validation_v3"
    / "evaluation_validation"
    / "evaluation_summary.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "outputs" / "hybrid_v2_final_statistics"
)
BOOTSTRAP_SEED = 20260817
SHELF_PATTERN = re.compile(r"^([LR])(\d+)_Product_")
PRIORITY_WEIGHTS = {
    "exact_product_origin_recall": 0.40,
    "reach_to_shelf_f1": 0.25,
    "pickup_candidate_f1": 0.15,
    "product_held_f1": 0.08,
    "retract_from_shelf_f1": 0.04,
    "return_candidate_f1": 0.03,
    "hand_in_shelf_f1": 0.025,
    "inspect_product_f1": 0.015,
    "inspect_shelf_f1": 0.005,
    "comparison_candidate_f1": 0.005,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create dissertation statistics from a frozen evaluation."
    )
    parser.add_argument("--batch-manifest", type=Path, default=DEFAULT_BATCH)
    parser.add_argument(
        "--baseline-batch-manifest",
        type=Path,
        default=DEFAULT_BASELINE_BATCH,
        help="Frozen rule-based test batch used for V1-versus-V2 comparisons.",
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=DEFAULT_VALIDATION_SUMMARY,
        help="Frozen Hybrid V2 validation summary for generalisation checks.",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()

def clean_records(frame):
    return json.loads(frame.to_json(orient="records"))

def finite(value):
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None

def bool_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def shelf_code(value):
    match = SHELF_PATTERN.match(str(value).strip())
    return None if match is None else f"{match.group(1)}{match.group(2)}"

def shelf_sort_key(value):
    match = re.match(r"^([LR])(\d+)$", str(value))
    if not match:
        return (2, 999, str(value))
    return (0 if match.group(1) == "L" else 1, int(match.group(2)), "")

def metric_counts(metrics, labels):
    primary = metrics[
        metrics["iou_threshold"] == evaluation.PRIMARY_IOU_THRESHOLD
    ]
    indexed = primary.set_index("label")
    return np.array(
        [
            [
                int(indexed.loc[label, "tp"]),
                int(indexed.loc[label, "fp"]),
                int(indexed.loc[label, "fn"]),
            ]
            for label in labels
        ],
        dtype=int,
    )

def f1_from_counts(counts):
    tp, fp, fn = [float(value) for value in counts]
    precision = evaluation.safe_divide(tp, tp + fp)
    recall = evaluation.safe_divide(tp, tp + fn)
    return evaluation.safe_divide(
        2.0 * precision * recall, precision + recall
    )

def primary_metric_map(metrics):
    primary = metrics[
        metrics["iou_threshold"] == evaluation.PRIMARY_IOU_THRESHOLD
    ]
    return {
        str(row.label): row
        for row in primary.itertuples(index=False)
    }

def summary_attribute(summary, attribute):
    rows = summary["manual_product_interaction"].get(
        "matched_attribute_metrics", []
    )
    for row in rows:
        if row.get("attribute") == attribute:
            return row.get("accuracy")
    return None

def product_match_attributes(
    ground_truth,
    predictions,
    boundary_frames,
    collect_origin_rows=False,
):
    matches, _, _ = evaluation.product_definition_matches(
        ground_truth,
        predictions,
        evaluation.PRIMARY_IOU_THRESHOLD,
        boundary_frames,
    )
    counts = {
        "origin_eligible": 0,
        "origin_correct": 0,
        "returned_eligible": 0,
        "returned_correct": 0,
        "manual_pickups": int(
            (ground_truth["interaction"] == "pickup_candidate").sum()
        ),
        "predicted_pickups": int(
            (predictions["interaction"] == "pickup_candidate").sum()
        ),
        "matched_pickups": 0,
        "pickup_origin_eligible": 0,
        "pickup_origin_correct": 0,
    }
    origin_rows = []
    for gt_id, pred_id, iou, method, error, tolerance in matches:
        gt_row = ground_truth.loc[
            ground_truth["row_id"] == gt_id
        ].iloc[0]
        pred_row = predictions.loc[
            predictions["row_id"] == pred_id
        ].iloc[0]
        manual_origin = str(gt_row.get("product_track_id", "")).strip()
        predicted_origin = str(
            pred_row.get("origin_product_zone", "")
        ).strip()
        if "_Product_" in manual_origin:
            counts["origin_eligible"] += 1
            counts["origin_correct"] += int(
                manual_origin == predicted_origin
            )
        manual_returned = str(gt_row.get("returned", "")).strip().lower()
        predicted_returned = str(
            pred_row.get("returned", "")
        ).strip().lower()
        if (
            gt_row["interaction"] != "comparison_candidate"
            and manual_returned in {"true", "false"}
        ):
            counts["returned_eligible"] += 1
            counts["returned_correct"] += int(
                manual_returned == predicted_returned
            )
        if gt_row["interaction"] == "pickup_candidate":
            counts["matched_pickups"] += 1
            if "_Product_" in manual_origin:
                counts["pickup_origin_eligible"] += 1
                counts["pickup_origin_correct"] += int(
                    manual_origin == predicted_origin
                )
                if collect_origin_rows:
                    origin_rows.append(
                        {
                            "video_id": str(gt_row["video_id"]),
                            "actual_product_zone": manual_origin,
                            "predicted_product_zone": predicted_origin,
                            "actual_shelf": shelf_code(manual_origin),
                            "predicted_shelf": shelf_code(predicted_origin),
                            "exact_correct": manual_origin == predicted_origin,
                            "ambiguous": bool_value(
                                pred_row.get(
                                    "origin_product_zone_ambiguous", False
                                )
                            ),
                            "confidence": finite(
                                pred_row.get(
                                    "origin_product_zone_confidence", 0
                                )
                            ),
                            "match_method": method,
                            "iou": iou,
                            "boundary_error_frames": error,
                            "boundary_tolerance_frames": tolerance,
                        }
                    )
    return counts, origin_rows

def origin_confusion(origin_match_rows):
    origin_matches = pd.DataFrame(origin_match_rows)
    if origin_matches.empty:
        return {
            "row_labels": [],
            "column_labels": [],
            "counts": [],
            "row_percentages": [],
            "matched_pickups": 0,
            "error_categories": [],
        }
    origin_matches["predicted_shelf"] = origin_matches[
        "predicted_shelf"
    ].fillna("Unknown")
    origin_matches["actual_shelf"] = origin_matches[
        "actual_shelf"
    ].fillna("Unknown")
    shelf_labels = sorted(
        set(origin_matches["actual_shelf"])
        | set(origin_matches["predicted_shelf"]),
        key=shelf_sort_key,
    )
    shelf_counts = pd.crosstab(
        origin_matches["actual_shelf"],
        origin_matches["predicted_shelf"],
    ).reindex(index=shelf_labels, columns=shelf_labels, fill_value=0)
    shelf_percentages = shelf_counts.div(
        shelf_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0)
    categories = []
    for row in origin_matches.itertuples(index=False):
        if row.actual_product_zone == row.predicted_product_zone:
            categories.append("Exact product zone")
        elif row.actual_shelf == row.predicted_shelf:
            categories.append("Correct shelf, wrong product slot")
        else:
            actual_match = re.match(r"^([LR])(\d+)$", row.actual_shelf)
            predicted_match = re.match(
                r"^([LR])(\d+)$", row.predicted_shelf
            )
            if (
                actual_match
                and predicted_match
                and actual_match.group(1) == predicted_match.group(1)
                and abs(
                    int(actual_match.group(2))
                    - int(predicted_match.group(2))
                ) == 1
            ):
                categories.append("Adjacent shelf")
            else:
                categories.append("Other shelf/unknown")
    error_categories = [
        {"category": key, "count": int(value)}
        for key, value in pd.Series(categories).value_counts().items()
    ]
    return {
        "row_labels": shelf_labels,
        "column_labels": shelf_labels,
        "counts": shelf_counts.values.tolist(),
        "row_percentages": shelf_percentages.values.tolist(),
        "matched_pickups": int(len(origin_matches)),
        "error_categories": error_categories,
    }

def class_comparison(baseline_rows, hybrid_rows):
    baseline = {str(row["label"]): row for row in baseline_rows}
    hybrid = {str(row["label"]): row for row in hybrid_rows}
    rows = []
    for label in hybrid:
        old = baseline[label]
        new = hybrid[label]
        rows.append(
            {
                "label": label,
                "ground_truth_events": int(new["ground_truth_events"]),
                "baseline_predicted_events": int(old["predicted_events"]),
                "hybrid_v2_predicted_events": int(new["predicted_events"]),
                "baseline_precision": float(old["precision"]),
                "hybrid_v2_precision": float(new["precision"]),
                "precision_change": float(
                    new["precision"] - old["precision"]
                ),
                "baseline_recall": float(old["recall"]),
                "hybrid_v2_recall": float(new["recall"]),
                "recall_change": float(new["recall"] - old["recall"]),
                "baseline_f1": float(old["f1"]),
                "hybrid_v2_f1": float(new["f1"]),
                "f1_change": float(new["f1"] - old["f1"]),
            }
        )
    return rows

def duration_summary(table, label_column, domain):
    rows = []
    for label, group in table.groupby(label_column, sort=False):
        durations = pd.to_numeric(group["duration_sec"], errors="coerce")
        durations = durations[np.isfinite(durations) & (durations >= 0)]
        if not len(durations):
            continue
        rows.append(
            {
                "domain": domain,
                "label": str(label),
                "count": int(len(durations)),
                "mean_sec": float(durations.mean()),
                "median_sec": float(durations.median()),
                "std_sec": float(durations.std(ddof=1)) if len(durations) > 1 else 0.0,
                "q1_sec": float(durations.quantile(0.25)),
                "q3_sec": float(durations.quantile(0.75)),
                "min_sec": float(durations.min()),
                "max_sec": float(durations.max()),
            }
        )
    return rows

def row_normalize_confusion(table, row_label):
    frame = table.copy().set_index(row_label)
    numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0)
    denominators = numeric.sum(axis=1).replace(0, np.nan)
    normalized = numeric.div(denominators, axis=0).fillna(0)
    return {
        "row_labels": [str(value) for value in normalized.index],
        "column_labels": [str(value) for value in normalized.columns],
        "counts": numeric.astype(int).values.tolist(),
        "row_percentages": normalized.values.tolist(),
    }

def confidence_interval(values):
    return {
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
    }

def main():
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    batch_path = args.batch_manifest.resolve()
    batch = evaluation.load_json(batch_path)
    records = evaluation.successful_records(batch, args.split)
    if not records:
        raise ValueError(f"No successful {args.split} videos were found.")
    video_ids = {str(record["video_id"]) for record in records}
    fps_by_video = evaluation.video_fps_map(records)
    boundary_frames = {
        video_id: max(
            0,
            int(round(
                evaluation.PRODUCT_BOUNDARY_TOLERANCE_SEC * fps
            )),
        )
        for video_id, fps in fps_by_video.items()
    }
    evaluation_dir = batch_path.parent / f"evaluation_{args.split}"
    summary = evaluation.load_json(evaluation_dir / "evaluation_summary.json")

    baseline_path = args.baseline_batch_manifest.resolve()
    baseline_batch = evaluation.load_json(baseline_path)
    baseline_records = evaluation.successful_records(
        baseline_batch, args.split
    )
    baseline_video_ids = {
        str(record["video_id"]) for record in baseline_records
    }
    if baseline_video_ids != video_ids:
        raise ValueError(
            "Baseline and Hybrid V2 batches must contain the same videos. "
            f"Missing from baseline: {sorted(video_ids - baseline_video_ids)}; "
            f"extra in baseline: {sorted(baseline_video_ids - video_ids)}."
        )
    baseline_evaluation_dir = (
        baseline_path.parent / f"evaluation_{args.split}"
    )
    baseline_summary = evaluation.load_json(
        baseline_evaluation_dir / "evaluation_summary.json"
    )

    behaviour_pred, _ = evaluation.read_csv_with_video(
        records, "classifier", "event_output_csv"
    )
    behaviour_pred = evaluation.normalize_intervals(
        behaviour_pred, "behaviour"
    )
    behaviour_pred = behaviour_pred[
        behaviour_pred["behaviour"].isin(evaluation.MERL_BEHAVIOURS)
    ].copy()
    behaviour_pred = evaluation.consolidate_merl_hand_in_shelf(
        behaviour_pred,
        fps_by_video,
        evaluation.MERL_HAND_IN_SHELF_MERGE_GAP_SEC,
    )
    baseline_behaviour_pred, _ = evaluation.read_csv_with_video(
        baseline_records, "classifier", "event_output_csv"
    )
    baseline_behaviour_pred = evaluation.normalize_intervals(
        baseline_behaviour_pred, "behaviour"
    )
    baseline_behaviour_pred = baseline_behaviour_pred[
        baseline_behaviour_pred["behaviour"].isin(
            evaluation.MERL_BEHAVIOURS
        )
    ].copy()
    baseline_behaviour_pred = evaluation.consolidate_merl_hand_in_shelf(
        baseline_behaviour_pred,
        fps_by_video,
        evaluation.MERL_HAND_IN_SHELF_MERGE_GAP_SEC,
    )

    merl_gt = pd.read_csv(
        evaluation.DEFAULT_MERL_EVENTS, dtype={"video_id": str}
    )
    merl_gt = merl_gt[
        (merl_gt["split"] == args.split)
        & (merl_gt["video_id"].isin(video_ids))
    ].copy()
    merl_gt = evaluation.normalize_intervals(merl_gt, "behaviour")

    product_pred, _ = evaluation.read_csv_with_video(
        records, "product_interaction", "event_output_csv"
    )
    product_pred = evaluation.normalize_intervals(
        product_pred, "interaction"
    )
    product_pred = product_pred[
        product_pred["interaction"].isin(evaluation.PRODUCT_INTERACTIONS)
    ].copy()
    baseline_product_pred, _ = evaluation.read_csv_with_video(
        baseline_records, "product_interaction", "event_output_csv"
    )
    baseline_product_pred = evaluation.normalize_intervals(
        baseline_product_pred, "interaction"
    )
    baseline_product_pred = baseline_product_pred[
        baseline_product_pred["interaction"].isin(
            evaluation.PRODUCT_INTERACTIONS
        )
    ].copy()
    manual_gt = evaluation.prepare_manual_events(
        evaluation.DEFAULT_MANUAL_EVENTS, args.split, video_ids
    )

    predicted_frames, _ = evaluation.read_csv_with_video(
        records, "classifier", "frame_output_csv"
    )
    merl_frames = pd.read_csv(
        evaluation.DEFAULT_MERL_FRAMES, dtype={"video_id": str}
    )
    merl_frames = merl_frames[
        (merl_frames["split"] == args.split)
        & (merl_frames["video_id"].isin(video_ids))
    ].copy()
    _, merged_frames = evaluation.evaluate_merl_frames(
        merl_frames, predicted_frames
    )

    video_rows = []
    merl_video_counts = []
    baseline_merl_video_counts = []
    product_video_counts = []
    baseline_product_video_counts = []
    attribute_video_counts = []
    baseline_attribute_video_counts = []
    attention_video_counts = []
    origin_match_rows = []
    baseline_origin_match_rows = []
    priority_rows = []

    for record in records:
        video_id = str(record["video_id"])
        vg = merl_gt[merl_gt["video_id"] == video_id]
        vp = behaviour_pred[behaviour_pred["video_id"] == video_id]
        merl_metrics = evaluation.event_metric_rows(
            vg,
            vp,
            "behaviour",
            evaluation.MERL_BEHAVIOURS,
            "merl_behaviour",
        )
        merl_counts = metric_counts(
            merl_metrics, evaluation.MERL_BEHAVIOURS
        )
        merl_video_counts.append(merl_counts)
        baseline_vp = baseline_behaviour_pred[
            baseline_behaviour_pred["video_id"] == video_id
        ]
        baseline_merl_metrics = evaluation.event_metric_rows(
            vg,
            baseline_vp,
            "behaviour",
            evaluation.MERL_BEHAVIOURS,
            "merl_behaviour",
        )
        baseline_merl_video_counts.append(
            metric_counts(
                baseline_merl_metrics, evaluation.MERL_BEHAVIOURS
            )
        )

        pg = manual_gt[manual_gt["video_id"] == video_id]
        pp = product_pred[product_pred["video_id"] == video_id]
        product_metrics = evaluation.product_definition_metric_rows(
            pg,
            pp,
            boundary_frames,
            evaluation.PRODUCT_BOUNDARY_TOLERANCE_SEC,
        )
        product_counts = metric_counts(
            product_metrics, evaluation.PRODUCT_INTERACTIONS
        )
        product_video_counts.append(product_counts)
        baseline_pp = baseline_product_pred[
            baseline_product_pred["video_id"] == video_id
        ]
        baseline_product_metrics = evaluation.product_definition_metric_rows(
            pg,
            baseline_pp,
            boundary_frames,
            evaluation.PRODUCT_BOUNDARY_TOLERANCE_SEC,
        )
        baseline_product_video_counts.append(
            metric_counts(
                baseline_product_metrics,
                evaluation.PRODUCT_INTERACTIONS,
            )
        )
        attributes, video_origin_rows = product_match_attributes(
            pg, pp, boundary_frames, collect_origin_rows=True
        )
        origin_match_rows.extend(video_origin_rows)
        baseline_attributes, baseline_video_origin_rows = (
            product_match_attributes(
                pg,
                baseline_pp,
                boundary_frames,
                collect_origin_rows=True,
            )
        )
        baseline_origin_match_rows.extend(baseline_video_origin_rows)
        attribute_video_counts.append(
            [
                attributes["origin_correct"],
                attributes["origin_eligible"],
                attributes["returned_correct"],
                attributes["returned_eligible"],
            ]
        )
        baseline_attribute_video_counts.append(
            [
                baseline_attributes["origin_correct"],
                baseline_attributes["origin_eligible"],
                baseline_attributes["returned_correct"],
                baseline_attributes["returned_eligible"],
            ]
        )

        vm = merged_frames[merged_frames["video_id"] == video_id].copy()
        inspect_product = evaluation.bool_series(vm["inspect_product_gt"])
        inspect_shelf = evaluation.bool_series(vm["inspect_shelf_gt"])
        eligible = inspect_product ^ inspect_shelf
        attention = vm[eligible].copy()
        attention["expected"] = np.where(
            inspect_product[eligible], "hands", "shelf"
        )
        attention["predicted"] = (
            attention["attention_target"]
            .fillna("uncertain")
            .astype(str)
            .str.lower()
            .replace({"held_product": "hands"})
        )
        attention.loc[
            ~attention["predicted"].isin({"hands", "shelf"}),
            "predicted",
        ] = "uncertain"
        covered = attention["predicted"].isin({"hands", "shelf"})
        correct = attention["predicted"] == attention["expected"]
        attention_video_counts.append(
            [
                len(attention),
                int(covered.sum()),
                int((covered & correct).sum()),
                int(correct.sum()),
            ]
        )

        primary_merl = merl_metrics[
            merl_metrics["iou_threshold"]
            == evaluation.PRIMARY_IOU_THRESHOLD
        ]
        primary_product = product_metrics[
            product_metrics["iou_threshold"]
            == evaluation.PRIMARY_IOU_THRESHOLD
        ]
        baseline_primary_merl = baseline_merl_metrics[
            baseline_merl_metrics["iou_threshold"]
            == evaluation.PRIMARY_IOU_THRESHOLD
        ]
        baseline_primary_product = baseline_product_metrics[
            baseline_product_metrics["iou_threshold"]
            == evaluation.PRIMARY_IOU_THRESHOLD
        ]
        merl_by_label = primary_metric_map(merl_metrics)
        product_by_label = primary_metric_map(product_metrics)
        video_rows.append(
            {
                "video_id": video_id,
                "merl_mean_class_f1": float(primary_merl["f1"].mean()),
                "baseline_merl_mean_class_f1": float(
                    baseline_primary_merl["f1"].mean()
                ),
                "product_mean_class_f1": float(
                    primary_product["f1"].mean()
                ),
                "baseline_product_mean_class_f1": float(
                    baseline_primary_product["f1"].mean()
                ),
                "exact_origin_accuracy_matched": evaluation.safe_divide(
                    attributes["origin_correct"],
                    attributes["origin_eligible"],
                ),
                "returned_accuracy_matched": evaluation.safe_divide(
                    attributes["returned_correct"],
                    attributes["returned_eligible"],
                ),
                "attention_coverage": evaluation.safe_divide(
                    int(covered.sum()), len(attention)
                ),
                "attention_accuracy_when_covered": evaluation.safe_divide(
                    int((covered & correct).sum()), int(covered.sum())
                ),
            }
        )
        priority = {
            "video_id": video_id,
            "manual_pickups": attributes["manual_pickups"],
            "predicted_pickups": attributes["predicted_pickups"],
            "matched_pickups": attributes["matched_pickups"],
            "origin_evaluable_matched_pickups": attributes[
                "pickup_origin_eligible"
            ],
            "exact_origin_correct_pickups": attributes[
                "pickup_origin_correct"
            ],
            "exact_product_origin_recall": evaluation.safe_divide(
                attributes["pickup_origin_correct"],
                attributes["manual_pickups"],
            ),
            "exact_origin_accuracy_when_matched": evaluation.safe_divide(
                attributes["pickup_origin_correct"],
                attributes["pickup_origin_eligible"],
            ),
            "reach_to_shelf_f1": float(
                merl_by_label["reach_to_shelf"].f1
            ),
            "pickup_candidate_f1": float(
                product_by_label["pickup_candidate"].f1
            ),
            "product_held_f1": float(
                product_by_label["product_held"].f1
            ),
            "retract_from_shelf_f1": float(
                merl_by_label["retract_from_shelf"].f1
            ),
            "return_candidate_f1": float(
                product_by_label["return_candidate"].f1
            ),
            "hand_in_shelf_f1": float(
                merl_by_label["hand_in_shelf"].f1
            ),
            "inspect_product_f1": float(
                merl_by_label["inspect_product"].f1
            ),
            "inspect_shelf_f1": float(
                merl_by_label["inspect_shelf"].f1
            ),
            "comparison_candidate_f1": float(
                product_by_label["comparison_candidate"].f1
            ),
        }
        priority["priority_score"] = float(
            sum(
                PRIORITY_WEIGHTS[key] * priority[key]
                for key in PRIORITY_WEIGHTS
            )
        )
        important_weight = sum(
            PRIORITY_WEIGHTS[key]
            for key in (
                "exact_product_origin_recall",
                "reach_to_shelf_f1",
                "pickup_candidate_f1",
            )
        )
        priority["important_signal_score"] = float(
            sum(
                PRIORITY_WEIGHTS[key] * priority[key]
                for key in (
                    "exact_product_origin_recall",
                    "reach_to_shelf_f1",
                    "pickup_candidate_f1",
                )
            )
            / important_weight
        )
        priority_rows.append(priority)

    merl_video_counts = np.asarray(merl_video_counts, dtype=int)
    baseline_merl_video_counts = np.asarray(
        baseline_merl_video_counts, dtype=int
    )
    product_video_counts = np.asarray(product_video_counts, dtype=int)
    baseline_product_video_counts = np.asarray(
        baseline_product_video_counts, dtype=int
    )
    attribute_video_counts = np.asarray(attribute_video_counts, dtype=int)
    baseline_attribute_video_counts = np.asarray(
        baseline_attribute_video_counts, dtype=int
    )
    attention_video_counts = np.asarray(attention_video_counts, dtype=int)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = {
        "merl_mean_class_f1": [],
        "product_mean_class_f1": [],
        "origin_top_candidate_accuracy": [],
        "returned_boolean_accuracy": [],
        "attention_coverage": [],
        "attention_accuracy_when_covered": [],
    }
    paired_bootstrap = {
        "MERL mean class F1": [],
        "Product mean class F1": [],
        "Product origin top-candidate accuracy": [],
        "Returned-boolean accuracy": [],
    }
    for label in evaluation.MERL_BEHAVIOURS:
        paired_bootstrap[f"MERL F1: {label}"] = []
    for label in evaluation.PRODUCT_INTERACTIONS:
        paired_bootstrap[f"Product F1: {label}"] = []
    video_count = len(records)
    for _ in range(args.bootstrap_samples):
        indexes = rng.integers(0, video_count, size=video_count)
        merl_sum = merl_video_counts[indexes].sum(axis=0)
        baseline_merl_sum = baseline_merl_video_counts[indexes].sum(axis=0)
        product_sum = product_video_counts[indexes].sum(axis=0)
        baseline_product_sum = baseline_product_video_counts[
            indexes
        ].sum(axis=0)
        attribute_sum = attribute_video_counts[indexes].sum(axis=0)
        baseline_attribute_sum = baseline_attribute_video_counts[
            indexes
        ].sum(axis=0)
        attention_sum = attention_video_counts[indexes].sum(axis=0)
        merl_scores = [f1_from_counts(row) for row in merl_sum]
        baseline_merl_scores = [
            f1_from_counts(row) for row in baseline_merl_sum
        ]
        product_scores = [f1_from_counts(row) for row in product_sum]
        baseline_product_scores = [
            f1_from_counts(row) for row in baseline_product_sum
        ]
        bootstrap["merl_mean_class_f1"].append(
            np.mean(merl_scores)
        )
        bootstrap["product_mean_class_f1"].append(
            np.mean(product_scores)
        )
        bootstrap["origin_top_candidate_accuracy"].append(
            evaluation.safe_divide(attribute_sum[0], attribute_sum[1])
        )
        bootstrap["returned_boolean_accuracy"].append(
            evaluation.safe_divide(attribute_sum[2], attribute_sum[3])
        )
        bootstrap["attention_coverage"].append(
            evaluation.safe_divide(attention_sum[1], attention_sum[0])
        )
        bootstrap["attention_accuracy_when_covered"].append(
            evaluation.safe_divide(attention_sum[2], attention_sum[1])
        )
        paired_bootstrap["MERL mean class F1"].append(
            float(np.mean(merl_scores) - np.mean(baseline_merl_scores))
        )
        paired_bootstrap["Product mean class F1"].append(
            float(
                np.mean(product_scores)
                - np.mean(baseline_product_scores)
            )
        )
        paired_bootstrap["Product origin top-candidate accuracy"].append(
            evaluation.safe_divide(attribute_sum[0], attribute_sum[1])
            - evaluation.safe_divide(
                baseline_attribute_sum[0], baseline_attribute_sum[1]
            )
        )
        paired_bootstrap["Returned-boolean accuracy"].append(
            evaluation.safe_divide(attribute_sum[2], attribute_sum[3])
            - evaluation.safe_divide(
                baseline_attribute_sum[2], baseline_attribute_sum[3]
            )
        )
        for index, label in enumerate(evaluation.MERL_BEHAVIOURS):
            paired_bootstrap[f"MERL F1: {label}"].append(
                merl_scores[index] - baseline_merl_scores[index]
            )
        for index, label in enumerate(evaluation.PRODUCT_INTERACTIONS):
            paired_bootstrap[f"Product F1: {label}"].append(
                product_scores[index] - baseline_product_scores[index]
            )

    merl_full = merl_video_counts.sum(axis=0)
    baseline_merl_full = baseline_merl_video_counts.sum(axis=0)
    product_full = product_video_counts.sum(axis=0)
    baseline_product_full = baseline_product_video_counts.sum(axis=0)
    attribute_full = attribute_video_counts.sum(axis=0)
    baseline_attribute_full = baseline_attribute_video_counts.sum(axis=0)
    attention_full = attention_video_counts.sum(axis=0)
    point_estimates = {
        "merl_mean_class_f1": float(
            np.mean([f1_from_counts(row) for row in merl_full])
        ),
        "product_mean_class_f1": float(
            np.mean([f1_from_counts(row) for row in product_full])
        ),
        "origin_top_candidate_accuracy": evaluation.safe_divide(
            attribute_full[0], attribute_full[1]
        ),
        "returned_boolean_accuracy": evaluation.safe_divide(
            attribute_full[2], attribute_full[3]
        ),
        "attention_coverage": evaluation.safe_divide(
            attention_full[1], attention_full[0]
        ),
        "attention_accuracy_when_covered": evaluation.safe_divide(
            attention_full[2], attention_full[1]
        ),
    }
    bootstrap_rows = []
    friendly_names = {
        "merl_mean_class_f1": "MERL mean class F1",
        "product_mean_class_f1": "Product mean class F1",
        "origin_top_candidate_accuracy": "Product origin top-candidate accuracy",
        "returned_boolean_accuracy": "Returned-boolean accuracy",
        "attention_coverage": "Attention coverage",
        "attention_accuracy_when_covered": "Attention accuracy when covered",
    }
    for key, values in bootstrap.items():
        interval = confidence_interval(values)
        bootstrap_rows.append(
            {
                "metric": friendly_names[key],
                "estimate": point_estimates[key],
                **interval,
                "bootstrap_unit": "video",
                "samples": args.bootstrap_samples,
                "seed": BOOTSTRAP_SEED,
            }
        )

    current_merl_full_scores = [
        f1_from_counts(row) for row in merl_full
    ]
    baseline_merl_full_scores = [
        f1_from_counts(row) for row in baseline_merl_full
    ]
    current_product_full_scores = [
        f1_from_counts(row) for row in product_full
    ]
    baseline_product_full_scores = [
        f1_from_counts(row) for row in baseline_product_full
    ]
    paired_points = {
        "MERL mean class F1": (
            float(np.mean(baseline_merl_full_scores)),
            float(np.mean(current_merl_full_scores)),
        ),
        "Product mean class F1": (
            float(np.mean(baseline_product_full_scores)),
            float(np.mean(current_product_full_scores)),
        ),
        "Product origin top-candidate accuracy": (
            evaluation.safe_divide(
                baseline_attribute_full[0], baseline_attribute_full[1]
            ),
            evaluation.safe_divide(attribute_full[0], attribute_full[1]),
        ),
        "Returned-boolean accuracy": (
            evaluation.safe_divide(
                baseline_attribute_full[2], baseline_attribute_full[3]
            ),
            evaluation.safe_divide(attribute_full[2], attribute_full[3]),
        ),
    }
    for index, label in enumerate(evaluation.MERL_BEHAVIOURS):
        paired_points[f"MERL F1: {label}"] = (
            baseline_merl_full_scores[index],
            current_merl_full_scores[index],
        )
    for index, label in enumerate(evaluation.PRODUCT_INTERACTIONS):
        paired_points[f"Product F1: {label}"] = (
            baseline_product_full_scores[index],
            current_product_full_scores[index],
        )
    paired_change_rows = []
    for metric, values in paired_bootstrap.items():
        interval = confidence_interval(values)
        baseline_value, hybrid_value = paired_points[metric]
        paired_change_rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "hybrid_v2": hybrid_value,
                "absolute_change": hybrid_value - baseline_value,
                "relative_change": (
                    evaluation.safe_divide(
                        hybrid_value - baseline_value, baseline_value
                    )
                    if baseline_value
                    else None
                ),
                **interval,
                "bootstrap_probability_positive": float(
                    np.mean(np.asarray(values) > 0)
                ),
                "ci_excludes_zero": bool(
                    interval["lower_95"] > 0
                    or interval["upper_95"] < 0
                ),
                "samples": args.bootstrap_samples,
                "seed": BOOTSTRAP_SEED,
            }
        )

    merl_confusion = pd.read_csv(
        evaluation_dir / "merl_primary_confusion_matrix.csv"
    )
    baseline_merl_confusion = pd.read_csv(
        baseline_evaluation_dir / "merl_primary_confusion_matrix.csv"
    )
    attention_confusion = pd.read_csv(
        evaluation_dir / "attention_proxy_confusion_matrix.csv"
    )
    current_origin_confusion = origin_confusion(origin_match_rows)
    baseline_origin_confusion = origin_confusion(
        baseline_origin_match_rows
    )

    manual_raw = pd.read_csv(
        evaluation.DEFAULT_MANUAL_EVENTS, dtype={"video_id": str}
    )
    manual_raw = manual_raw[
        (manual_raw["split"] == args.split)
        & (manual_raw["video_id"].isin(video_ids))
        & (manual_raw["label"].isin(evaluation.PRODUCT_LABEL_MAP))
    ].copy()
    if "uncertain" in manual_raw:
        manual_raw = manual_raw[
            ~evaluation.bool_series(manual_raw["uncertain"])
        ].copy()
    manual_raw["duration_sec"] = pd.to_numeric(
        manual_raw["duration_sec"], errors="coerce"
    )
    merl_gt["duration_sec"] = (
        merl_gt["end_frame"] - merl_gt["start_frame"] + 1
    ) / 30.0

    pickup_origins = manual_raw[manual_raw["label"] == "pickup"].copy()
    pickup_origins["shelf"] = pickup_origins["product_track_id"].map(
        shelf_code
    )
    pickup_distribution = (
        pickup_origins.dropna(subset=["shelf"])
        .groupby("shelf")
        .size()
        .rename("pickup_count")
        .reset_index()
    )
    pickup_distribution["sort_key"] = pickup_distribution["shelf"].map(
        shelf_sort_key
    )
    pickup_distribution = pickup_distribution.sort_values(
        "sort_key", kind="stable"
    ).drop(columns="sort_key")

    priority_rows.sort(
        key=lambda row: (row["priority_score"], row["video_id"]),
        reverse=True,
    )
    for rank, row in enumerate(priority_rows, start=1):
        row["rank"] = rank
    median_priority = float(
        np.median([row["priority_score"] for row in priority_rows])
    )
    representative_pool = [
        row
        for row in priority_rows
        if row["exact_origin_correct_pickups"] > 0
    ] or priority_rows
    representative = min(
        representative_pool,
        key=lambda row: (
            abs(row["priority_score"] - median_priority),
            row["video_id"],
        ),
    )
    ranking = {
        "batch_manifest": str(batch_path),
        "split": args.split,
        "ranking_purpose": "qualitative example selection only",
        "model_or_threshold_changes": False,
        "priority_weights": PRIORITY_WEIGHTS,
        "exact_product_origin_definition": (
            "correct predicted top origin zone on a definition-aligned "
            "matched manual pickup, divided by all manual pickups"
        ),
        "representative_selection": (
            "closest to the all-video median priority score among videos "
            "with at least one exact correct product origin"
        ),
        "median_priority_score": median_priority,
        "selected_examples": {
            "best": priority_rows[0]["video_id"],
            "representative": representative["video_id"],
            "worst": priority_rows[-1]["video_id"],
        },
        "videos": priority_rows,
    }
    validation_summary_path = args.validation_summary.resolve()
    validation_summary = (
        evaluation.load_json(validation_summary_path)
        if validation_summary_path.is_file()
        else None
    )

    def overall_comparison_row(metric, baseline_value, hybrid_value):
        return {
            "metric": metric,
            "baseline": baseline_value,
            "hybrid_v2": hybrid_value,
            "absolute_change": hybrid_value - baseline_value,
            "relative_change": (
                evaluation.safe_divide(
                    hybrid_value - baseline_value, baseline_value
                )
                if baseline_value
                else None
            ),
        }

    baseline_v2_headline_comparison = [
        overall_comparison_row(
            "MERL mean class F1",
            baseline_summary["merl_behaviour"][
                "mean_class_f1_at_primary_iou"
            ],
            summary["merl_behaviour"]["mean_class_f1_at_primary_iou"],
        ),
        overall_comparison_row(
            "MERL mAP",
            baseline_summary["merl_behaviour"]["map_at_primary_iou"],
            summary["merl_behaviour"]["map_at_primary_iou"],
        ),
        overall_comparison_row(
            "Exact multilabel sampled-frame accuracy",
            baseline_summary["merl_behaviour"][
                "exact_multilabel_sampled_frame_accuracy"
            ],
            summary["merl_behaviour"][
                "exact_multilabel_sampled_frame_accuracy"
            ],
        ),
        overall_comparison_row(
            "Unambiguous primary-frame accuracy",
            baseline_summary["merl_behaviour"][
                "unambiguous_primary_frame_accuracy"
            ],
            summary["merl_behaviour"][
                "unambiguous_primary_frame_accuracy"
            ],
        ),
        overall_comparison_row(
            "Product definition-aligned mean F1",
            baseline_summary["manual_product_interaction"][
                "mean_class_f1_definition_aligned"
            ],
            summary["manual_product_interaction"][
                "mean_class_f1_definition_aligned"
            ],
        ),
        overall_comparison_row(
            "Product definition-aligned mAP",
            baseline_summary["manual_product_interaction"][
                "map_definition_aligned"
            ],
            summary["manual_product_interaction"][
                "map_definition_aligned"
            ],
        ),
        overall_comparison_row(
            "Product strict tIoU mean F1",
            baseline_summary["manual_product_interaction"][
                "mean_class_f1_tiou_only"
            ],
            summary["manual_product_interaction"][
                "mean_class_f1_tiou_only"
            ],
        ),
        overall_comparison_row(
            "Exact product-origin accuracy",
            summary_attribute(
                baseline_summary, "origin_product_zone_top_candidate"
            ),
            summary_attribute(
                summary, "origin_product_zone_top_candidate"
            ),
        ),
        overall_comparison_row(
            "Unambiguous product-origin accuracy",
            summary_attribute(
                baseline_summary, "origin_product_zone_unambiguous"
            ),
            summary_attribute(
                summary, "origin_product_zone_unambiguous"
            ),
        ),
        overall_comparison_row(
            "Product-origin decision coverage",
            summary_attribute(
                baseline_summary, "origin_product_zone_decision_coverage"
            ),
            summary_attribute(
                summary, "origin_product_zone_decision_coverage"
            ),
        ),
        overall_comparison_row(
            "Returned-boolean accuracy",
            summary_attribute(baseline_summary, "returned"),
            summary_attribute(summary, "returned"),
        ),
    ]
    merl_class_comparison = class_comparison(
        baseline_summary["merl_behaviour"]["per_class_primary_iou"],
        summary["merl_behaviour"]["per_class_primary_iou"],
    )
    product_class_comparison = class_comparison(
        baseline_summary["manual_product_interaction"][
            "per_class_definition_aligned"
        ],
        summary["manual_product_interaction"][
            "per_class_definition_aligned"
        ],
    )

    output = {
        "metadata": {
            "batch_manifest": str(batch_path),
            "batch_run_id": batch.get("batch_run_id"),
            "baseline_batch_manifest": str(baseline_path),
            "baseline_batch_run_id": baseline_batch.get("batch_run_id"),
            "validation_summary": str(validation_summary_path),
            "split": args.split,
            "video_count": len(records),
            "video_ids": [str(record["video_id"]) for record in records],
            "primary_tiou_threshold": evaluation.PRIMARY_IOU_THRESHOLD,
            "product_boundary_tolerance_sec": (
                evaluation.PRODUCT_BOUNDARY_TOLERANCE_SEC
            ),
            "hand_in_shelf_merge_gap_sec": (
                evaluation.MERL_HAND_IN_SHELF_MERGE_GAP_SEC
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interpretation": [
                "MERL and product prevalence/durations use ground truth annotations.",
                "Attention statistics are proxy agreement, not direct gaze accuracy.",
                "Product origin confusion is conditional on a definition-aligned matched pickup.",
                "returned=False does not prove purchase; it means no supported return was observed.",
                "Bootstrap confidence intervals resample videos, not correlated frames.",
                "V1-versus-V2 change intervals use paired resampling of the same 28 test videos.",
                "Test outputs are frozen; these comparisons do not change models or thresholds.",
            ],
        },
        "headline": {
            "merl_mean_class_f1": summary["merl_behaviour"][
                "mean_class_f1_at_primary_iou"
            ],
            "merl_map": summary["merl_behaviour"]["map_at_primary_iou"],
            "product_mean_class_f1": summary[
                "manual_product_interaction"
            ]["mean_class_f1_definition_aligned"],
            "product_map": summary["manual_product_interaction"][
                "map_definition_aligned"
            ],
            "product_strict_tiou_f1": summary[
                "manual_product_interaction"
            ]["mean_class_f1_tiou_only"],
            "origin_top_candidate_accuracy": point_estimates[
                "origin_top_candidate_accuracy"
            ],
            "returned_boolean_accuracy": point_estimates[
                "returned_boolean_accuracy"
            ],
            "attention_coverage": point_estimates["attention_coverage"],
            "attention_accuracy_when_covered": point_estimates[
                "attention_accuracy_when_covered"
            ],
        },
        "validation_test_comparison": (
            [
                {
                    "metric": "MERL mean class F1",
                    "validation": validation_summary["merl_behaviour"][
                        "mean_class_f1_at_primary_iou"
                    ],
                    "test": summary["merl_behaviour"][
                        "mean_class_f1_at_primary_iou"
                    ],
                },
                {
                    "metric": "MERL mAP",
                    "validation": validation_summary["merl_behaviour"][
                        "map_at_primary_iou"
                    ],
                    "test": summary["merl_behaviour"]["map_at_primary_iou"],
                },
                {
                    "metric": "Product mean class F1",
                    "validation": validation_summary[
                        "manual_product_interaction"
                    ]["mean_class_f1_definition_aligned"],
                    "test": summary["manual_product_interaction"][
                        "mean_class_f1_definition_aligned"
                    ],
                },
                {
                    "metric": "Product strict tIoU F1",
                    "validation": validation_summary[
                        "manual_product_interaction"
                    ]["mean_class_f1_tiou_only"],
                    "test": summary["manual_product_interaction"][
                        "mean_class_f1_tiou_only"
                    ],
                },
            ]
            if validation_summary is not None
            else []
        ),
        "baseline_v2_headline_comparison": (
            baseline_v2_headline_comparison
        ),
        "merl_class_comparison": merl_class_comparison,
        "product_class_comparison": product_class_comparison,
        "paired_change_confidence_intervals": paired_change_rows,
        "merl_performance": summary["merl_behaviour"][
            "per_class_primary_iou"
        ],
        "product_performance": summary["manual_product_interaction"][
            "per_class_definition_aligned"
        ],
        "merl_confusion": row_normalize_confusion(
            merl_confusion, "ground_truth"
        ),
        "baseline_merl_confusion": row_normalize_confusion(
            baseline_merl_confusion, "ground_truth"
        ),
        "attention_confusion": row_normalize_confusion(
            attention_confusion, "expected_attention"
        ),
        "origin_shelf_confusion": current_origin_confusion,
        "baseline_origin_shelf_confusion": baseline_origin_confusion,
        "duration_statistics": (
            duration_summary(
                merl_gt, "behaviour", "MERL ground truth"
            )
            + duration_summary(
                manual_raw, "label", "Manual product ground truth"
            )
        ),
        "duration_raw": {
            "merl": {
                label: pd.to_numeric(
                    group["duration_sec"], errors="coerce"
                ).dropna().tolist()
                for label, group in merl_gt.groupby("behaviour")
            },
            "product": {
                label: pd.to_numeric(
                    group["duration_sec"], errors="coerce"
                ).dropna().tolist()
                for label, group in manual_raw.groupby("label")
            },
        },
        "pickup_origin_distribution": clean_records(pickup_distribution),
        "per_video_statistics": video_rows,
        "priority_ranking": ranking,
        "bootstrap_confidence_intervals": bootstrap_rows,
        "source_files": [
            str(evaluation.DEFAULT_MERL_EVENTS.resolve()),
            str(evaluation.DEFAULT_MERL_FRAMES.resolve()),
            str(evaluation.DEFAULT_MANUAL_EVENTS.resolve()),
            str((evaluation_dir / "evaluation_summary.json").resolve()),
            str((evaluation_dir / "merl_primary_confusion_matrix.csv").resolve()),
            str((evaluation_dir / "attention_proxy_confusion_matrix.csv").resolve()),
            str((baseline_evaluation_dir / "evaluation_summary.json").resolve()),
            str((baseline_evaluation_dir / "merl_primary_confusion_matrix.csv").resolve()),
            str(validation_summary_path),
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "statistics_data.json"
    evaluation.write_json(output_path, output)
    evaluation.write_json(
        args.output_dir / "video_priority_ranking.json", ranking
    )
    csv_outputs = {
        "baseline_v2_headline_comparison.csv": (
            baseline_v2_headline_comparison
        ),
        "merl_class_comparison.csv": merl_class_comparison,
        "product_class_comparison.csv": product_class_comparison,
        "paired_change_confidence_intervals.csv": paired_change_rows,
        "bootstrap_confidence_intervals.csv": bootstrap_rows,
        "per_video_priority_ranking.csv": priority_rows,
        "duration_statistics.csv": output["duration_statistics"],
    }
    for filename, rows in csv_outputs.items():
        pd.DataFrame(rows).to_csv(
            args.output_dir / filename, index=False, encoding="utf-8"
        )
    print("Dissertation statistics generated.")
    print("Baseline:", baseline_batch.get("batch_run_id"))
    print("Hybrid V2:", batch.get("batch_run_id"))
    print("Videos:", len(records))
    print("MERL ground-truth events:", len(merl_gt))
    print("Manual product events:", len(manual_gt))
    print(
        "Origin-matched pickups:",
        current_origin_confusion["matched_pickups"],
    )
    print("Saved:", output_path.resolve())

if __name__ == "__main__":
    main()
