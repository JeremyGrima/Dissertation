import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MERL_EVENTS = (
    PROJECT_DIR / "evaluation" / "merl_ground_truth_events.csv" #Evaluates behaviour, product-interaction, and attention results against the available ground-truth annotations.
)
DEFAULT_MERL_FRAMES = (
    PROJECT_DIR / "evaluation" / "merl_ground_truth_frames.csv"
)
DEFAULT_MANUAL_EVENTS = (
    PROJECT_DIR / "annotations" / "manual_event_annotations.csv"
)
DEFAULT_BATCH_ROOT = PROJECT_DIR / "evaluation" / "batch_runs"

MERL_BEHAVIOURS = [
    "reach_to_shelf",
    "retract_from_shelf",
    "hand_in_shelf",
    "inspect_product",
    "inspect_shelf",
]
PRODUCT_LABEL_MAP = {
    "pickup": "pickup_candidate",
    "product_held": "product_held",
    "return": "return_candidate",
    "comparison": "comparison_candidate",
}
PRODUCT_INTERACTIONS = list(PRODUCT_LABEL_MAP.values())
IOU_THRESHOLDS = (0.10, 0.30, 0.50)
PRIMARY_IOU_THRESHOLD = 0.30
MERL_HAND_IN_SHELF_MERGE_GAP_SEC = 1.0
PRODUCT_BOUNDARY_TOLERANCE_SEC = 0.5
BOUNDARY_MATCH_INTERACTIONS = {
    "pickup_candidate": ("end_frame", "start_frame"),
    "return_candidate": ("start_frame", "start_frame"),
}

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)

def finite_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one completed batch without mixing validation and "
            "test results."
        )
    )
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        default=None,
        help="batch_manifest.json from run_pipeline_batch.py.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default=None,
        help="Required only if a batch contains both splits.",
    )
    parser.add_argument(
        "--merl-events", type=Path, default=DEFAULT_MERL_EVENTS
    )
    parser.add_argument(
        "--merl-frames", type=Path, default=DEFAULT_MERL_FRAMES
    )
    parser.add_argument(
        "--manual-events", type=Path, default=DEFAULT_MANUAL_EVENTS
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <batch directory>/evaluation_<split>.",
    )
    parser.add_argument(
        "--hand-in-shelf-merge-gap-sec",
        type=float,
        default=MERL_HAND_IN_SHELF_MERGE_GAP_SEC,
        help=(
            "Maximum gap used to consolidate wrist-level hand-in-shelf "
            "predictions into person-level MERL episodes (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--product-boundary-tolerance-sec",
        type=float,
        default=PRODUCT_BOUNDARY_TOLERANCE_SEC,
        help=(
            "Boundary tolerance for pickup/return matching in seconds "
            "(default: 0.5)."
        ),
    )
    return parser.parse_args()

def latest_batch_manifest():
    candidates = list(DEFAULT_BATCH_ROOT.glob("*/batch_manifest.json"))
    if not candidates:
        raise FileNotFoundError(
            "No batch manifest was found. Pass --batch-manifest explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)

def infer_split(batch, requested):
    completed_splits = {
        record.get("split")
        for record in batch.get("videos", [])
        if record.get("status") == "success"
    }
    completed_splits.discard(None)
    if requested:
        if requested not in completed_splits:
            raise ValueError(
                f"No successful {requested} videos exist in this batch."
            )
        return requested
    if len(completed_splits) == 1:
        return next(iter(completed_splits))
    raise ValueError(
        "This batch contains multiple splits. Pass --split validation or "
        "--split test so their results remain separate."
    )

def successful_records(batch, split):
    records = [
        record
        for record in batch.get("videos", [])
        if record.get("status") == "success"
        and record.get("split") == split
    ]
    records.sort(
        key=lambda record: (
            int(record.get("subject_id", 0)),
            int(record.get("session_id", 0)),
        )
    )
    return records

def read_csv_with_video(records, stage, key, required=True):
    tables = []
    missing = []
    for record in records:
        value = record.get("stages", {}).get(stage, {}).get(key)
        path = None if not value else Path(value)
        if path is None or not path.is_file():
            missing.append(record["video_id"])
            continue
        table = pd.read_csv(path)
        table.insert(0, "video_id", str(record["video_id"]))
        tables.append(table)
    if missing and required:
        raise FileNotFoundError(
            f"Missing {stage}.{key} outputs for: " + ", ".join(missing)
        )
    if not tables:
        return pd.DataFrame(), missing
    return pd.concat(tables, ignore_index=True), missing

def normalize_intervals(table, label_column):
    output = table.copy()
    output["video_id"] = output["video_id"].astype(str)
    output[label_column] = output[label_column].astype(str)
    output["start_frame"] = pd.to_numeric(
        output["start_frame"], errors="coerce"
    )
    output["end_frame"] = pd.to_numeric(
        output["end_frame"], errors="coerce"
    )
    output = output.dropna(subset=["start_frame", "end_frame"])
    output["start_frame"] = output["start_frame"].astype(int)
    output["end_frame"] = output["end_frame"].astype(int)
    output = output[output["end_frame"] >= output["start_frame"]].copy()
    output = output.reset_index(drop=True)
    output["row_id"] = np.arange(len(output), dtype=int)
    return output

def video_fps_map(records):
    output = {}
    for record in records:
        fps = record.get("stages", {}).get("pose", {}).get("fps", 30.0)
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 30.0
        output[str(record["video_id"])] = fps if fps > 0 else 30.0
    return output

def consolidate_merl_hand_in_shelf(events, fps_by_video, gap_sec):
    """Convert wrist episodes into the person-level action MERL annotates."""
    other = events[events["behaviour"] != "hand_in_shelf"].copy()
    hand = events[events["behaviour"] == "hand_in_shelf"].copy()
    merged_rows = []
    for video_id, group in hand.groupby("video_id", sort=False):
        fps = fps_by_video.get(str(video_id), 30.0)
        gap_frames = max(0, int(round(float(gap_sec) * fps)))
        current = None
        members = []
        for _, row in group.sort_values(
            ["start_frame", "end_frame"], kind="stable"
        ).iterrows():
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            if current is not None and start <= current["end_frame"] + gap_frames:
                current["end_frame"] = max(current["end_frame"], end)
                if "rule_score" in row:
                    current["rule_score"] = max(
                        finite_or_none(current.get("rule_score")) or 0.0,
                        finite_or_none(row.get("rule_score")) or 0.0,
                    )
                members.append(row)
                continue
            if current is not None:
                current["merged_wrist_event_count"] = len(members)
                current["merged_wrist_values"] = "|".join(sorted({
                    str(item.get("wrist", "")).strip()
                    for item in members
                    if str(item.get("wrist", "")).strip()
                }))
                current["wrist"] = "person"
                merged_rows.append(current)
            current = row.to_dict()
            current["start_frame"] = start
            current["end_frame"] = end
            members = [row]
        if current is not None:
            current["merged_wrist_event_count"] = len(members)
            current["merged_wrist_values"] = "|".join(sorted({
                str(item.get("wrist", "")).strip()
                for item in members
                if str(item.get("wrist", "")).strip()
            }))
            current["wrist"] = "person"
            merged_rows.append(current)

    merged_hand = pd.DataFrame(merged_rows)
    combined = pd.concat([other, merged_hand], ignore_index=True, sort=False)
    return normalize_intervals(combined, "behaviour")

def interval_iou(gt_row, pred_row):
    intersection = max(
        0,
        min(int(gt_row.end_frame), int(pred_row.end_frame))
        - max(int(gt_row.start_frame), int(pred_row.start_frame))
        + 1,
    )
    if not intersection:
        return 0.0
    gt_length = int(gt_row.end_frame) - int(gt_row.start_frame) + 1
    pred_length = int(pred_row.end_frame) - int(pred_row.start_frame) + 1
    return intersection / float(gt_length + pred_length - intersection)

def greedy_matches(gt, pred, label_column, threshold):
    matches = []
    matched_gt = set()
    matched_pred = set()
    group_keys = ["video_id", label_column]
    gt_groups = {
        key: group for key, group in gt.groupby(group_keys, sort=False)
    }
    pred_groups = {
        key: group for key, group in pred.groupby(group_keys, sort=False)
    }
    for key in set(gt_groups) | set(pred_groups):
        gt_group = gt_groups.get(key, gt.iloc[0:0])
        pred_group = pred_groups.get(key, pred.iloc[0:0])
        candidates = []
        for gt_row in gt_group.itertuples(index=False):
            for pred_row in pred_group.itertuples(index=False):
                iou = interval_iou(gt_row, pred_row)
                if iou >= threshold:
                    candidates.append(
                        (iou, int(gt_row.row_id), int(pred_row.row_id))
                    )
        candidates.sort(reverse=True)
        for iou, gt_id, pred_id in candidates:
            if gt_id in matched_gt or pred_id in matched_pred:
                continue
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
            matches.append((gt_id, pred_id, iou))
    return matches, matched_gt, matched_pred

def product_pair_match(gt_row, pred_row, label, threshold, tolerance_frames):
    iou = interval_iou(gt_row, pred_row)
    if iou >= threshold:
        return True, (2, iou), "temporal_iou", None
    fields = BOUNDARY_MATCH_INTERACTIONS.get(label)
    if fields is None:
        return False, (0, 0.0), None, None
    gt_field, pred_field = fields
    boundary_error = abs(
        int(getattr(gt_row, gt_field)) - int(getattr(pred_row, pred_field))
    )
    if boundary_error <= tolerance_frames:
        quality = 1.0 - safe_divide(boundary_error, tolerance_frames + 1)
        return True, (1, quality), "boundary_tolerance", boundary_error
    return False, (0, 0.0), None, boundary_error

def product_definition_matches(
    gt, pred, threshold, tolerance_frames_by_video
):
    matches = []
    matched_gt = set()
    matched_pred = set()
    group_keys = ["video_id", "interaction"]
    gt_groups = {
        key: group for key, group in gt.groupby(group_keys, sort=False)
    }
    pred_groups = {
        key: group for key, group in pred.groupby(group_keys, sort=False)
    }
    for key in set(gt_groups) | set(pred_groups):
        gt_group = gt_groups.get(key, gt.iloc[0:0])
        pred_group = pred_groups.get(key, pred.iloc[0:0])
        video_id, label = key
        tolerance_frames = tolerance_frames_by_video.get(str(video_id), 15)
        candidates = []
        for gt_row in gt_group.itertuples(index=False):
            for pred_row in pred_group.itertuples(index=False):
                valid, quality, method, boundary_error = product_pair_match(
                    gt_row,
                    pred_row,
                    label,
                    threshold,
                    tolerance_frames,
                )
                if not valid:
                    continue
                candidates.append(
                    (
                        quality[0],
                        quality[1],
                        int(gt_row.row_id),
                        int(pred_row.row_id),
                        interval_iou(gt_row, pred_row),
                        method,
                        boundary_error,
                        tolerance_frames,
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for (
            _,
            _,
            gt_id,
            pred_id,
            iou,
            method,
            boundary_error,
            tolerance_frames,
        ) in candidates:
            if gt_id in matched_gt or pred_id in matched_pred:
                continue
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
            matches.append(
                (
                    gt_id,
                    pred_id,
                    iou,
                    method,
                    boundary_error,
                    tolerance_frames,
                )
            )
    return matches, matched_gt, matched_pred

def product_definition_metric_rows(
    gt, pred, tolerance_frames_by_video, tolerance_sec
):
    rows = []
    for threshold in IOU_THRESHOLDS:
        matches, matched_gt, matched_pred = product_definition_matches(
            gt, pred, threshold, tolerance_frames_by_video
        )
        match_table = pd.DataFrame(
            matches,
            columns=[
                "gt_id",
                "pred_id",
                "iou",
                "match_method",
                "boundary_error_frames",
                "boundary_tolerance_frames",
            ],
        )
        for label in PRODUCT_INTERACTIONS:
            gt_ids = set(gt.loc[gt["interaction"] == label, "row_id"])
            pred_ids = set(
                pred.loc[pred["interaction"] == label, "row_id"]
            )
            label_matches = (
                match_table[match_table["gt_id"].isin(gt_ids)]
                if not match_table.empty
                else match_table
            )
            tp = len(label_matches)
            fp = len(pred_ids - matched_pred)
            fn = len(gt_ids - matched_gt)
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            f1 = safe_divide(2 * precision * recall, precision + recall)
            start_errors = []
            end_errors = []
            for item in label_matches.itertuples(index=False):
                gt_row = gt.loc[gt["row_id"] == item.gt_id].iloc[0]
                pred_row = pred.loc[pred["row_id"] == item.pred_id].iloc[0]
                start_errors.append(
                    abs(int(gt_row.start_frame) - int(pred_row.start_frame))
                )
                end_errors.append(
                    abs(int(gt_row.end_frame) - int(pred_row.end_frame))
                )
            boundary_values = pd.to_numeric(
                label_matches.get("boundary_error_frames"), errors="coerce"
            ).dropna()
            rows.append(
                {
                    "domain": "product_interaction_definition_aligned",
                    "label": label,
                    "iou_threshold": threshold,
                    "matching_policy": (
                        "temporal_iou_or_boundary_tolerance"
                        if label in BOUNDARY_MATCH_INTERACTIONS
                        else "temporal_iou"
                    ),
                    "boundary_tolerance_sec": (
                        tolerance_sec
                        if label in BOUNDARY_MATCH_INTERACTIONS
                        else np.nan
                    ),
                    "ground_truth_events": len(gt_ids),
                    "predicted_events": len(pred_ids),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "temporal_iou_matches": int(
                        (label_matches.get("match_method") == "temporal_iou").sum()
                    ) if not label_matches.empty else 0,
                    "boundary_tolerance_matches": int(
                        (label_matches.get("match_method") == "boundary_tolerance").sum()
                    ) if not label_matches.empty else 0,
                    "mean_matched_iou": (
                        float(label_matches["iou"].mean())
                        if tp else np.nan
                    ),
                    "mean_boundary_error_frames": (
                        float(boundary_values.mean())
                        if len(boundary_values) else np.nan
                    ),
                    "start_frame_mae": (
                        float(np.mean(start_errors)) if start_errors else np.nan
                    ),
                    "end_frame_mae": (
                        float(np.mean(end_errors)) if end_errors else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)

def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0

def event_metric_rows(gt, pred, label_column, labels, domain):
    rows = []
    for threshold in IOU_THRESHOLDS:
        matches, matched_gt, matched_pred = greedy_matches(
            gt, pred, label_column, threshold
        )
        match_table = pd.DataFrame(
            matches, columns=["gt_id", "pred_id", "iou"]
        )
        for label in labels:
            gt_ids = set(gt.loc[gt[label_column] == label, "row_id"])
            pred_ids = set(
                pred.loc[pred[label_column] == label, "row_id"]
            )
            label_matches = (
                match_table[match_table["gt_id"].isin(gt_ids)]
                if not match_table.empty
                else match_table
            )
            tp = len(label_matches)
            fp = len(pred_ids - matched_pred)
            fn = len(gt_ids - matched_gt)
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            f1 = safe_divide(2 * precision * recall, precision + recall)
            if tp:
                pairs = []
                for item in label_matches.itertuples(index=False):
                    gt_row = gt.loc[gt["row_id"] == item.gt_id].iloc[0]
                    pred_row = pred.loc[pred["row_id"] == item.pred_id].iloc[0]
                    pairs.append((gt_row, pred_row, float(item.iou)))
                mean_iou = float(np.mean([pair[2] for pair in pairs]))
                start_mae = float(
                    np.mean(
                        [
                            abs(
                                int(pair[0].start_frame)
                                - int(pair[1].start_frame)
                            )
                            for pair in pairs
                        ]
                    )
                )
                end_mae = float(
                    np.mean(
                        [
                            abs(
                                int(pair[0].end_frame)
                                - int(pair[1].end_frame)
                            )
                            for pair in pairs
                        ]
                    )
                )
            else:
                mean_iou = np.nan
                start_mae = np.nan
                end_mae = np.nan
            rows.append(
                {
                    "domain": domain,
                    "label": label,
                    "iou_threshold": threshold,
                    "ground_truth_events": len(gt_ids),
                    "predicted_events": len(pred_ids),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_matched_iou": mean_iou,
                    "start_frame_mae": start_mae,
                    "end_frame_mae": end_mae,
                }
            )
    return pd.DataFrame(rows)

def average_precision(recalls, precisions):
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(
        np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1])
    )

def ap_rows(gt, pred, label_column, score_column, labels, domain):
    output = []
    scores = pd.to_numeric(pred.get(score_column), errors="coerce").fillna(0.0)
    pred = pred.copy()
    pred["_score"] = scores
    for threshold in IOU_THRESHOLDS:
        class_aps = []
        for label in labels:
            class_gt = gt[gt[label_column] == label]
            class_pred = pred[pred[label_column] == label].sort_values(
                "_score", ascending=False, kind="stable"
            )
            gt_by_video = {
                video_id: list(group.itertuples(index=False))
                for video_id, group in class_gt.groupby("video_id")
            }
            used_gt = {video_id: set() for video_id in gt_by_video}
            true_positive = []
            false_positive = []
            for prediction in class_pred.itertuples(index=False):
                candidates = gt_by_video.get(str(prediction.video_id), [])
                best_index = None
                best_iou = 0.0
                for index, ground_truth in enumerate(candidates):
                    if index in used_gt[str(prediction.video_id)]:
                        continue
                    iou = interval_iou(ground_truth, prediction)
                    if iou > best_iou:
                        best_iou = iou
                        best_index = index
                if best_index is not None and best_iou >= threshold:
                    used_gt[str(prediction.video_id)].add(best_index)
                    true_positive.append(1.0)
                    false_positive.append(0.0)
                else:
                    true_positive.append(0.0)
                    false_positive.append(1.0)
            cumulative_tp = np.cumsum(true_positive)
            cumulative_fp = np.cumsum(false_positive)
            recalls = cumulative_tp / max(len(class_gt), 1)
            precisions = cumulative_tp / np.maximum(
                cumulative_tp + cumulative_fp, 1.0
            )
            ap = (
                average_precision(recalls, precisions)
                if len(class_gt)
                else np.nan
            )
            if np.isfinite(ap):
                class_aps.append(ap)
            output.append(
                {
                    "domain": domain,
                    "label": label,
                    "iou_threshold": threshold,
                    "ground_truth_events": len(class_gt),
                    "predicted_events": len(class_pred),
                    "average_precision": ap,
                }
            )
        output.append(
            {
                "domain": domain,
                "label": "mAP",
                "iou_threshold": threshold,
                "ground_truth_events": len(gt),
                "predicted_events": len(pred),
                "average_precision": (
                    float(np.mean(class_aps)) if class_aps else np.nan
                ),
            }
        )
    return pd.DataFrame(output)

def product_definition_ap_rows(
    gt, pred, score_column, tolerance_frames_by_video, tolerance_sec
):
    output = []
    scores = pd.to_numeric(pred.get(score_column), errors="coerce").fillna(0.0)
    pred = pred.copy()
    pred["_score"] = scores
    for threshold in IOU_THRESHOLDS:
        class_aps = []
        for label in PRODUCT_INTERACTIONS:
            class_gt = gt[gt["interaction"] == label]
            class_pred = pred[pred["interaction"] == label].sort_values(
                "_score", ascending=False, kind="stable"
            )
            gt_by_video = {
                str(video_id): list(group.itertuples(index=False))
                for video_id, group in class_gt.groupby("video_id")
            }
            used_gt = {video_id: set() for video_id in gt_by_video}
            true_positive = []
            false_positive = []
            for prediction in class_pred.itertuples(index=False):
                video_id = str(prediction.video_id)
                candidates = gt_by_video.get(video_id, [])
                tolerance_frames = tolerance_frames_by_video.get(
                    video_id, 15
                )
                best_index = None
                best_quality = (-1, -1.0)
                for index, ground_truth in enumerate(candidates):
                    if index in used_gt[video_id]:
                        continue
                    valid, quality, _, _ = product_pair_match(
                        ground_truth,
                        prediction,
                        label,
                        threshold,
                        tolerance_frames,
                    )
                    if valid and quality > best_quality:
                        best_quality = quality
                        best_index = index
                if best_index is not None:
                    used_gt[video_id].add(best_index)
                    true_positive.append(1.0)
                    false_positive.append(0.0)
                else:
                    true_positive.append(0.0)
                    false_positive.append(1.0)
            cumulative_tp = np.cumsum(true_positive)
            cumulative_fp = np.cumsum(false_positive)
            recalls = cumulative_tp / max(len(class_gt), 1)
            precisions = cumulative_tp / np.maximum(
                cumulative_tp + cumulative_fp, 1.0
            )
            ap = (
                average_precision(recalls, precisions)
                if len(class_gt)
                else np.nan
            )
            if np.isfinite(ap):
                class_aps.append(ap)
            output.append(
                {
                    "domain": "product_interaction_definition_aligned",
                    "label": label,
                    "iou_threshold": threshold,
                    "matching_policy": (
                        "temporal_iou_or_boundary_tolerance"
                        if label in BOUNDARY_MATCH_INTERACTIONS
                        else "temporal_iou"
                    ),
                    "boundary_tolerance_sec": (
                        tolerance_sec
                        if label in BOUNDARY_MATCH_INTERACTIONS
                        else np.nan
                    ),
                    "ground_truth_events": len(class_gt),
                    "predicted_events": len(class_pred),
                    "average_precision": ap,
                }
            )
        output.append(
            {
                "domain": "product_interaction_definition_aligned",
                "label": "mAP",
                "iou_threshold": threshold,
                "matching_policy": "class_specific",
                "boundary_tolerance_sec": tolerance_sec,
                "ground_truth_events": len(gt),
                "predicted_events": len(pred),
                "average_precision": (
                    float(np.mean(class_aps)) if class_aps else np.nan
                ),
            }
        )
    return pd.DataFrame(output)

def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )

def evaluate_merl_frames(gt_frames, predictions):
    gt_frames = gt_frames.copy()
    gt_frames["video_id"] = gt_frames["video_id"].astype(str)
    gt_frames["source_frame"] = pd.to_numeric(
        gt_frames["source_frame"], errors="coerce"
    ).astype("Int64")
    predictions = predictions.copy()
    predictions["video_id"] = predictions["video_id"].astype(str)
    predictions["frame"] = pd.to_numeric(
        predictions["frame"], errors="coerce"
    ).astype("Int64")
    merged = predictions.merge(
        gt_frames,
        left_on=["video_id", "frame"],
        right_on=["video_id", "source_frame"],
        how="inner",
        suffixes=("_pred", "_gt"),
    )

    rows = []
    exact = np.ones(len(merged), dtype=bool)
    for behaviour in MERL_BEHAVIOURS:
        pred_column = f"{behaviour}_pred"
        gt_column = f"{behaviour}_gt"
        predicted = bool_series(merged[pred_column])
        actual = bool_series(merged[gt_column])
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        tn = int((~predicted & ~actual).sum())
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        rows.append(
            {
                "label": behaviour,
                "evaluated_sampled_frames": len(merged),
                "positive_ground_truth_frames": int(actual.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": safe_divide(
                    2 * precision * recall, precision + recall
                ),
                "accuracy": safe_divide(tp + tn, len(merged)),
            }
        )
        exact &= predicted.to_numpy() == actual.to_numpy()

    rows.append(
        {
            "label": "exact_multilabel_set",
            "evaluated_sampled_frames": len(merged),
            "positive_ground_truth_frames": np.nan,
            "tp": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tn": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "accuracy": float(exact.mean()) if len(exact) else np.nan,
        }
    )
    return pd.DataFrame(rows), merged

def primary_confusion(merged):
    overlap = pd.to_numeric(merged["overlap_count"], errors="coerce")
    unambiguous = merged[overlap <= 1].copy()
    gt_labels = unambiguous["behaviour_labels_gt"].fillna("background")
    pred_labels = unambiguous["primary_behaviour"].fillna("uncertain")
    gt_labels = gt_labels.replace("", "background")
    pred_labels = pred_labels.replace("", "uncertain")
    confusion = pd.crosstab(
        gt_labels,
        pred_labels,
        rownames=["ground_truth"],
        colnames=["predicted"],
        dropna=False,
    )
    confusion = confusion.reset_index()
    accuracy = safe_divide(
        int((gt_labels.to_numpy() == pred_labels.to_numpy()).sum()),
        len(unambiguous),
    )
    return confusion, accuracy, len(unambiguous)

def attention_proxy(merged):
    inspect_product = bool_series(merged["inspect_product_gt"])
    inspect_shelf = bool_series(merged["inspect_shelf_gt"])
    eligible = inspect_product ^ inspect_shelf
    subset = merged[eligible].copy()
    subset["expected_attention"] = np.where(
        inspect_product[eligible], "hands", "shelf"
    )
    raw_target = subset["attention_target"].fillna("uncertain").astype(str)
    subset["predicted_attention"] = raw_target.str.lower().replace(
        {"held_product": "hands"}
    )
    valid_targets = {"hands", "shelf"}
    subset.loc[
        ~subset["predicted_attention"].isin(valid_targets),
        "predicted_attention",
    ] = "uncertain"
    subset["covered"] = subset["predicted_attention"].isin(valid_targets)
    subset["correct"] = (
        subset["predicted_attention"] == subset["expected_attention"]
    )

    rows = []
    for target in ("hands", "shelf", "all"):
        group = (
            subset
            if target == "all"
            else subset[subset["expected_attention"] == target]
        )
        covered = group[group["covered"]]
        rows.append(
            {
                "expected_target": target,
                "eligible_sampled_frames": len(group),
                "covered_frames": len(covered),
                "coverage": safe_divide(len(covered), len(group)),
                "accuracy_when_covered": safe_divide(
                    int(covered["correct"].sum()), len(covered)
                ),
                "overall_agreement_with_uncertain_as_incorrect": safe_divide(
                    int(group["correct"].sum()), len(group)
                ),
            }
        )
    confusion = pd.crosstab(
        subset["expected_attention"],
        subset["predicted_attention"],
        rownames=["expected_attention"],
        colnames=["predicted_attention"],
        dropna=False,
    ).reset_index()
    return pd.DataFrame(rows), confusion

def prepare_manual_events(path, split, video_ids):
    manual = pd.read_csv(path, dtype={"video_id": str})
    manual = manual[
        (manual["split"] == split)
        & (manual["video_id"].isin(video_ids))
    ].copy()
    if "uncertain" in manual:
        manual = manual[~bool_series(manual["uncertain"])].copy()
    manual = manual[manual["label"].isin(PRODUCT_LABEL_MAP)].copy()
    manual["interaction"] = manual["label"].map(PRODUCT_LABEL_MAP)
    return normalize_intervals(manual, "interaction")

def matched_attribute_accuracy(gt, pred, tolerance_frames_by_video):
    matches, _, _ = product_definition_matches(
        gt,
        pred,
        PRIMARY_IOU_THRESHOLD,
        tolerance_frames_by_video,
    )
    result = {
        "returned": {"eligible": 0, "correct": 0},
        "origin_product_zone_top_candidate": {"eligible": 0, "correct": 0},
        "origin_product_zone_unambiguous": {"eligible": 0, "correct": 0},
        "origin_product_zone_decision_coverage": {
            "eligible": 0,
            "correct": 0,
        },
    }
    match_rows = []
    for (
        gt_id,
        pred_id,
        iou,
        match_method,
        boundary_error_frames,
        boundary_tolerance_frames,
    ) in matches:
        gt_row = gt.loc[gt["row_id"] == gt_id].iloc[0]
        pred_row = pred.loc[pred["row_id"] == pred_id].iloc[0]
        manual_returned = str(gt_row.get("returned", "")).strip().lower()
        predicted_returned = str(pred_row.get("returned", "")).strip().lower()
        return_eligible = (
            gt_row["interaction"] != "comparison_candidate"
            and manual_returned in {"true", "false"}
        )
        return_correct = (
            return_eligible and manual_returned == predicted_returned
        )
        if return_eligible:
            result["returned"]["eligible"] += 1
            result["returned"]["correct"] += int(return_correct)

        manual_origin = str(gt_row.get("product_track_id", "")).strip()
        predicted_origin_value = pred_row.get("origin_product_zone", "")
        predicted_origin = (
            "" if pd.isna(predicted_origin_value)
            else str(predicted_origin_value).strip()
        )
        origin_ambiguous = bool_series(pd.Series([
            pred_row.get("origin_product_zone_ambiguous", False)
        ])).iloc[0]
        origin_eligible = "_Product_" in manual_origin
        origin_decided = (
            origin_eligible and bool(predicted_origin) and not origin_ambiguous
        )
        origin_correct = origin_eligible and manual_origin == predicted_origin
        if origin_eligible:
            result["origin_product_zone_top_candidate"]["eligible"] += 1
            result["origin_product_zone_top_candidate"]["correct"] += int(
                origin_correct
            )
            result["origin_product_zone_decision_coverage"]["eligible"] += 1
            result["origin_product_zone_decision_coverage"]["correct"] += int(
                origin_decided
            )
        if origin_decided:
            result["origin_product_zone_unambiguous"]["eligible"] += 1
            result["origin_product_zone_unambiguous"]["correct"] += int(
                origin_correct
            )

        match_rows.append(
            {
                "domain": "product",
                "video_id": gt_row["video_id"],
                "label": gt_row["interaction"],
                "iou": iou,
                "match_method": match_method,
                "boundary_error_frames": boundary_error_frames,
                "boundary_tolerance_frames": boundary_tolerance_frames,
                "ground_truth_start_frame": int(gt_row["start_frame"]),
                "ground_truth_end_frame": int(gt_row["end_frame"]),
                "predicted_start_frame": int(pred_row["start_frame"]),
                "predicted_end_frame": int(pred_row["end_frame"]),
                "manual_product_track_id": manual_origin,
                "predicted_track_id": pred_row.get("track_id"),
                "manual_returned": manual_returned,
                "predicted_returned": predicted_returned,
                "return_value_evaluated": return_eligible,
                "return_value_correct": (
                    bool(return_correct) if return_eligible else None
                ),
                "origin_zone_evaluated": origin_eligible,
                "origin_zone_correct": (
                    bool(origin_correct) if origin_eligible else None
                ),
                "origin_zone_ambiguous": bool(origin_ambiguous),
                "origin_zone_alternative": pred_row.get(
                    "origin_product_zone_alternative"
                ),
                "origin_zone_confidence": finite_or_none(
                    pred_row.get("origin_product_zone_confidence")
                ),
            }
        )

    rows = []
    for attribute, counts in result.items():
        rows.append(
            {
                "attribute": attribute,
                "eligible_matched_events": counts["eligible"],
                "correct": counts["correct"],
                "accuracy": safe_divide(
                    counts["correct"], counts["eligible"]
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(match_rows)

def merl_primary_matches(gt, pred):
    matches, _, _ = greedy_matches(
        gt, pred, "behaviour", PRIMARY_IOU_THRESHOLD
    )
    rows = []
    for gt_id, pred_id, iou in matches:
        gt_row = gt.loc[gt["row_id"] == gt_id].iloc[0]
        pred_row = pred.loc[pred["row_id"] == pred_id].iloc[0]
        rows.append(
            {
                "domain": "merl_behaviour",
                "video_id": gt_row["video_id"],
                "label": gt_row["behaviour"],
                "iou": iou,
                "ground_truth_start_frame": int(gt_row["start_frame"]),
                "ground_truth_end_frame": int(gt_row["end_frame"]),
                "predicted_start_frame": int(pred_row["start_frame"]),
                "predicted_end_frame": int(pred_row["end_frame"]),
                "rule_score": finite_or_none(pred_row.get("rule_score")),
            }
        )
    return pd.DataFrame(rows)

def dataframe_records(table):
    clean = table.replace({np.nan: None})
    return clean.to_dict(orient="records")

def main():
    args = parse_args()
    if args.hand_in_shelf_merge_gap_sec < 0:
        raise ValueError("--hand-in-shelf-merge-gap-sec cannot be negative.")
    if args.product_boundary_tolerance_sec < 0:
        raise ValueError("--product-boundary-tolerance-sec cannot be negative.")
    batch_path = (
        latest_batch_manifest()
        if args.batch_manifest is None
        else args.batch_manifest.resolve()
    )
    if not batch_path.is_file():
        raise FileNotFoundError(f"Batch manifest not found: {batch_path}")
    batch = load_json(batch_path)
    split = infer_split(batch, args.split)
    records = successful_records(batch, split)
    if not records:
        raise ValueError(f"No successful {split} runs were found.")
    video_ids = {str(record["video_id"]) for record in records}
    fps_by_video = video_fps_map(records)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else batch_path.parent / f"evaluation_{split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_behaviour_events, _ = read_csv_with_video(
        records, "classifier", "event_output_csv"
    )
    predicted_frames, _ = read_csv_with_video(
        records, "classifier", "frame_output_csv"
    )
    predicted_product_events, product_missing = read_csv_with_video(
        records,
        "product_interaction",
        "event_output_csv",
        required=False,
    )

    merl_gt = pd.read_csv(args.merl_events, dtype={"video_id": str})
    merl_gt = merl_gt[
        (merl_gt["split"] == split)
        & (merl_gt["video_id"].isin(video_ids))
    ].copy()
    merl_gt = normalize_intervals(merl_gt, "behaviour")
    predicted_behaviour_events = normalize_intervals(
        predicted_behaviour_events, "behaviour"
    )
    predicted_behaviour_events = predicted_behaviour_events[
        predicted_behaviour_events["behaviour"].isin(MERL_BEHAVIOURS)
    ].copy()
    raw_predicted_behaviour_event_count = len(predicted_behaviour_events)
    raw_hand_in_shelf_count = int(
        (predicted_behaviour_events["behaviour"] == "hand_in_shelf").sum()
    )
    predicted_behaviour_events = consolidate_merl_hand_in_shelf(
        predicted_behaviour_events,
        fps_by_video,
        args.hand_in_shelf_merge_gap_sec,
    )
    consolidated_hand_in_shelf_count = int(
        (predicted_behaviour_events["behaviour"] == "hand_in_shelf").sum()
    )

    merl_event_metrics = event_metric_rows(
        merl_gt,
        predicted_behaviour_events,
        "behaviour",
        MERL_BEHAVIOURS,
        "merl_behaviour",
    )
    merl_ap = ap_rows(
        merl_gt,
        predicted_behaviour_events,
        "behaviour",
        "rule_score",
        MERL_BEHAVIOURS,
        "merl_behaviour",
    )

    merl_frames = pd.read_csv(args.merl_frames, dtype={"video_id": str})
    merl_frames = merl_frames[
        (merl_frames["split"] == split)
        & (merl_frames["video_id"].isin(video_ids))
    ].copy()
    frame_metrics, merged_frames = evaluate_merl_frames(
        merl_frames, predicted_frames
    )
    primary_matrix, primary_accuracy, primary_count = primary_confusion(
        merged_frames
    )
    attention_metrics, attention_matrix = attention_proxy(merged_frames)

    manual_gt = prepare_manual_events(
        args.manual_events, split, video_ids
    )
    if predicted_product_events.empty:
        product_pred = pd.DataFrame(
            columns=[
                "video_id",
                "interaction",
                "start_frame",
                "end_frame",
                "confidence",
            ]
        )
    else:
        product_pred = normalize_intervals(
            predicted_product_events, "interaction"
        )
        product_pred = product_pred[
            product_pred["interaction"].isin(PRODUCT_INTERACTIONS)
        ].copy()
    product_tiou_metrics = event_metric_rows(
        manual_gt,
        product_pred,
        "interaction",
        PRODUCT_INTERACTIONS,
        "product_interaction",
    )
    product_tiou_ap = ap_rows(
        manual_gt,
        product_pred,
        "interaction",
        "confidence",
        PRODUCT_INTERACTIONS,
        "product_interaction",
    )
    boundary_tolerance_frames = {
        video_id: max(
            0,
            int(round(args.product_boundary_tolerance_sec * fps)),
        )
        for video_id, fps in fps_by_video.items()
    }
    product_metrics = product_definition_metric_rows(
        manual_gt,
        product_pred,
        boundary_tolerance_frames,
        args.product_boundary_tolerance_sec,
    )
    product_ap = product_definition_ap_rows(
        manual_gt,
        product_pred,
        "confidence",
        boundary_tolerance_frames,
        args.product_boundary_tolerance_sec,
    )
    attribute_metrics, product_matches = matched_attribute_accuracy(
        manual_gt, product_pred, boundary_tolerance_frames
    )
    merl_matches = merl_primary_matches(
        merl_gt, predicted_behaviour_events
    )
    event_matches = pd.concat(
        [merl_matches, product_matches], ignore_index=True, sort=False
    )

    merl_event_metrics.to_csv(
        output_dir / "merl_event_metrics.csv", index=False
    )
    merl_ap.to_csv(output_dir / "merl_ap_metrics.csv", index=False)
    predicted_behaviour_events.to_csv(
        output_dir / "merl_evaluation_events.csv", index=False
    )
    frame_metrics.to_csv(
        output_dir / "merl_frame_metrics.csv", index=False
    )
    primary_matrix.to_csv(
        output_dir / "merl_primary_confusion_matrix.csv", index=False
    )
    attention_metrics.to_csv(
        output_dir / "attention_proxy_metrics.csv", index=False
    )
    attention_matrix.to_csv(
        output_dir / "attention_proxy_confusion_matrix.csv", index=False
    )
    product_metrics.to_csv(
        output_dir / "product_event_metrics.csv", index=False
    )
    product_ap.to_csv(
        output_dir / "product_ap_metrics.csv", index=False
    )
    product_tiou_metrics.to_csv(
        output_dir / "product_event_metrics_tiou_only.csv", index=False
    )
    product_tiou_ap.to_csv(
        output_dir / "product_ap_metrics_tiou_only.csv", index=False
    )
    attribute_metrics.to_csv(
        output_dir / "product_attribute_metrics.csv", index=False
    )
    event_matches.to_csv(output_dir / "event_matches.csv", index=False)

    primary_merl = merl_event_metrics[
        merl_event_metrics["iou_threshold"] == PRIMARY_IOU_THRESHOLD
    ]
    primary_product = product_metrics[
        product_metrics["iou_threshold"] == PRIMARY_IOU_THRESHOLD
    ]
    primary_product_tiou = product_tiou_metrics[
        product_tiou_metrics["iou_threshold"] == PRIMARY_IOU_THRESHOLD
    ]
    merl_map_row = merl_ap[
        (merl_ap["label"] == "mAP")
        & (merl_ap["iou_threshold"] == PRIMARY_IOU_THRESHOLD)
    ]
    product_map_row = product_ap[
        (product_ap["label"] == "mAP")
        & (product_ap["iou_threshold"] == PRIMARY_IOU_THRESHOLD)
    ]
    product_tiou_map_row = product_tiou_ap[
        (product_tiou_ap["label"] == "mAP")
        & (product_tiou_ap["iou_threshold"] == PRIMARY_IOU_THRESHOLD)
    ]
    exact_row = frame_metrics[
        frame_metrics["label"] == "exact_multilabel_set"
    ]
    attention_all = attention_metrics[
        attention_metrics["expected_target"] == "all"
    ]

    summary = {
        "schema_version": 2,
        "created_at": now_iso(),
        "batch_manifest": str(batch_path.resolve()),
        "batch_run_id": batch.get("batch_run_id"),
        "split": split,
        "pipeline_configuration": {
            "classifier_mode": batch.get("settings", {}).get(
                "classifier_mode", "rules_v1"
            ),
            "model_manifest": batch.get("settings", {}).get(
                "model_manifest"
            ),
            "origin_calibration": batch.get("settings", {}).get(
                "origin_calibration"
            ),
            "source_sha256": batch.get("source_sha256", {}),
            "frozen_configuration_confirmed": bool(
                batch.get("frozen_configuration_confirmed", False)
            ),
        },
        "evaluated_video_count": len(records),
        "evaluated_video_ids": [record["video_id"] for record in records],
        "failed_or_unprocessed_selected_videos": max(
            0,
            int(batch.get("selected_video_count", len(records)))
            - len(records),
        ),
        "primary_event_iou_threshold": PRIMARY_IOU_THRESHOLD,
        "merl_behaviour": {
            "ground_truth_event_count": len(merl_gt),
            "predicted_event_count": len(predicted_behaviour_events),
            "raw_wrist_level_predicted_event_count": (
                raw_predicted_behaviour_event_count
            ),
            "hand_in_shelf_consolidation": {
                "unit": "person_level_for_merl_evaluation",
                "maximum_gap_sec": args.hand_in_shelf_merge_gap_sec,
                "raw_wrist_level_event_count": raw_hand_in_shelf_count,
                "consolidated_person_level_event_count": (
                    consolidated_hand_in_shelf_count
                ),
            },
            "mean_class_f1_at_primary_iou": finite_or_none(
                primary_merl["f1"].mean()
            ),
            "map_at_primary_iou": (
                finite_or_none(merl_map_row.iloc[0]["average_precision"])
                if len(merl_map_row)
                else None
            ),
            "exact_multilabel_sampled_frame_accuracy": (
                finite_or_none(exact_row.iloc[0]["accuracy"])
                if len(exact_row)
                else None
            ),
            "unambiguous_primary_frame_accuracy": primary_accuracy,
            "unambiguous_primary_frame_count": primary_count,
            "per_class_primary_iou": dataframe_records(primary_merl),
        },
        "manual_product_interaction": {
            "ground_truth_event_count": len(manual_gt),
            "predicted_event_count": len(product_pred),
            "definition_aligned_matching": {
                "pickup_candidate": (
                    "manual pickup end versus predicted pickup start"
                ),
                "return_candidate": (
                    "manual return start versus predicted return start"
                ),
                "boundary_tolerance_sec": (
                    args.product_boundary_tolerance_sec
                ),
                "product_held": "temporal IoU",
                "comparison_candidate": "temporal IoU",
            },
            "mean_class_f1_definition_aligned": finite_or_none(
                primary_product["f1"].mean()
            ),
            "map_definition_aligned": (
                finite_or_none(product_map_row.iloc[0]["average_precision"])
                if len(product_map_row)
                else None
            ),
            "per_class_definition_aligned": dataframe_records(
                primary_product
            ),
            "mean_class_f1_tiou_only": finite_or_none(
                primary_product_tiou["f1"].mean()
            ),
            "map_tiou_only": (
                finite_or_none(
                    product_tiou_map_row.iloc[0]["average_precision"]
                )
                if len(product_tiou_map_row)
                else None
            ),
            "per_class_tiou_only": dataframe_records(
                primary_product_tiou
            ),
            "matched_attribute_metrics": dataframe_records(
                attribute_metrics
            ),
            "videos_missing_product_output": product_missing,
        },
        "attention_proxy": (
            dataframe_records(attention_all)[0]
            if len(attention_all)
            else None
        ),
        "output_directory": str(output_dir.resolve()),
    }
    write_json(output_dir / "evaluation_summary.json", summary)

    print("Pipeline evaluation complete.")
    print("Batch:", batch.get("batch_run_id"))
    print("Split:", split)
    print("Videos evaluated:", len(records))
    print(
        f"MERL mean class F1 @ tIoU {PRIMARY_IOU_THRESHOLD:.2f}:",
        f"{summary['merl_behaviour']['mean_class_f1_at_primary_iou']:.3f}",
    )
    print(
        f"MERL mAP @ tIoU {PRIMARY_IOU_THRESHOLD:.2f}:",
        f"{summary['merl_behaviour']['map_at_primary_iou']:.3f}",
    )
    print(
        "Product mean class F1 (definition-aligned):",
        f"{summary['manual_product_interaction']['mean_class_f1_definition_aligned']:.3f}",
    )
    print(
        f"Product mean class F1 @ tIoU {PRIMARY_IOU_THRESHOLD:.2f} (audit):",
        f"{summary['manual_product_interaction']['mean_class_f1_tiou_only']:.3f}",
    )
    print("Saved evaluation directory:", output_dir)

if __name__ == "__main__":
    main()
