from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_VERSION = "hybrid_v2"
MODEL_SCHEMA_VERSION = 1
RANDOM_SEED = 20260820

BEHAVIOURS = [ #Utility functions used for training and running the MERL behaviour classifier.
    "reach_to_shelf",
    "retract_from_shelf",
    "hand_in_shelf",
    "inspect_product",
    "inspect_shelf",
]

PRIMARY_PRIORITY = [  # Same primary-label order as the rule baseline.
    "hand_in_shelf",
    "reach_to_shelf",
    "retract_from_shelf",
    "inspect_product",
    "inspect_shelf",
]

NUMERIC_FEATURES = [
    "person_box_width_px",
    "person_box_height_px",
    "person_box_diagonal_px",
    "body_to_shelf_distance_px",
    "body_to_shelf_norm",
    "body_near_shelf_dwell_sec",
    "mean_hip_angle_deg",
    "mean_knee_angle_deg",
    "left_hip_angle_deg",
    "right_hip_angle_deg",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "left_wrist_conf",
    "right_wrist_conf",
    "left_wrist_speed_px_sec",
    "right_wrist_speed_px_sec",
    "left_wrist_acceleration_px_sec2",
    "right_wrist_acceleration_px_sec2",
    "left_forearm_length_px",
    "right_forearm_length_px",
    "left_shoulder_to_wrist_px",
    "right_shoulder_to_wrist_px",
    "left_arm_extension_ratio",
    "right_arm_extension_ratio",
    "left_wrist_to_body_norm",
    "right_wrist_to_body_norm",
    "left_wrist_to_shelf_norm",
    "right_wrist_to_shelf_norm",
    "left_wrist_to_product_norm",
    "right_wrist_to_product_norm",
    "left_wrist_outward_speed_px_sec",
    "right_wrist_outward_speed_px_sec",
    "left_wrist_toward_body_speed_px_sec",
    "right_wrist_toward_body_speed_px_sec",
    "left_wrist_toward_shelf_speed_px_sec",
    "right_wrist_toward_shelf_speed_px_sec",
    "left_shelf_dwell_sec",
    "right_shelf_dwell_sec",
    "left_product_dwell_sec",
    "right_product_dwell_sec",
    "wrist_midpoint_to_body_norm",
    "inter_wrist_distance_norm",
    "wrist_midpoint_to_head_norm",
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
    "product_presentation_score",
    "product_presentation_dwell_sec",
    "multi_product_holding_score",
    "multi_product_holding_dwell_sec",
    "attention_direction_quality",
    "attention_shelf_distance_norm",
    "head_to_shelf_angle_deg",
    "torso_to_shelf_angle_deg",
    "attention_to_shelf_angle_deg",
    "head_to_hands_angle_deg",
    "torso_to_hands_angle_deg",
    "attention_to_hands_angle_deg",
    "attention_target_score",
    "attention_target_dwell_sec",
    "attention_to_shelf_dwell_sec",
    "attention_to_hands_dwell_sec",
]

BOOLEAN_FEATURES = [
    "body_near_shelf",
    "left_wrist_observed",
    "right_wrist_observed",
    "left_wrist_held",
    "right_wrist_held",
    "left_arm_extended",
    "right_arm_extended",
    "left_in_shelf",
    "right_in_shelf",
    "left_in_product",
    "right_in_product",
    "left_shelf_entry",
    "right_shelf_entry",
    "left_shelf_exit",
    "right_shelf_exit",
    "left_product_entry",
    "right_product_entry",
    "left_product_exit",
    "right_product_exit",
    "left_near_product",
    "right_near_product",
    "both_wrists_observed",
    "hands_near_body",
    "hands_together",
    "hands_near_head",
    "presentation_arm_support",
    "product_presentation_pose",
    "multi_product_holding_pose",
    "attention_to_shelf_proxy",
    "attention_to_hands_proxy",
    "held_product_available",
]

CATEGORICAL_FEATURES = [
    "posture_state",
    "attention_target",
    "attention_target_raw",
    "attention_direction_method",
    "head_direction_method",
    "left_wrist_motion_state",
    "right_wrist_motion_state",
    "left_zone_assignment_source",
    "right_zone_assignment_source",
    "left_shelf_zone",
    "right_shelf_zone",
]

DYNAMIC_NUMERIC_FEATURES = [
    "body_to_shelf_norm",
    "left_wrist_speed_px_sec",
    "right_wrist_speed_px_sec",
    "left_wrist_acceleration_px_sec2",
    "right_wrist_acceleration_px_sec2",
    "left_arm_extension_ratio",
    "right_arm_extension_ratio",
    "left_wrist_to_body_norm",
    "right_wrist_to_body_norm",
    "left_wrist_to_shelf_norm",
    "right_wrist_to_shelf_norm",
    "left_wrist_to_product_norm",
    "right_wrist_to_product_norm",
    "left_wrist_toward_body_speed_px_sec",
    "right_wrist_toward_body_speed_px_sec",
    "left_wrist_toward_shelf_speed_px_sec",
    "right_wrist_toward_shelf_speed_px_sec",
    "left_shelf_dwell_sec",
    "right_shelf_dwell_sec",
    "wrist_midpoint_to_body_norm",
    "inter_wrist_distance_norm",
    "product_presentation_score",
    "attention_to_shelf_angle_deg",
    "attention_to_hands_angle_deg",
    "attention_target_score",
]

RECENT_BOOLEAN_FEATURES = [
    "left_shelf_entry",
    "right_shelf_entry",
    "left_shelf_exit",
    "right_shelf_exit",
    "left_product_entry",
    "right_product_entry",
    "left_product_exit",
    "right_product_exit",
    "left_in_shelf",
    "right_in_shelf",
    "left_in_product",
    "right_in_product",
    "hands_near_body",
    "product_presentation_pose",
    "attention_to_shelf_proxy",
    "attention_to_hands_proxy",
]

CONTEXT_WINDOWS_SEC = (0.5, 1.0)

def as_boolean(series):
    """Convert CSV/object booleans without treating non-empty text as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )

def infer_sampling(frame_df):
    frame = pd.to_numeric(frame_df["frame"], errors="coerce")
    time = pd.to_numeric(frame_df["time_sec"], errors="coerce")
    frame_delta = frame.diff()
    time_delta = time.diff()
    valid = (frame_delta > 0) & (time_delta > 0)
    if not valid.any():
        return 30.0, 1, 1.0 / 30.0
    fps = float((frame_delta[valid] / time_delta[valid]).median())
    if abs(fps - round(fps)) < 0.1:
        fps = float(round(fps))
    stride = max(1, int(round(frame_delta[frame_delta > 0].median())))
    sample_sec = float(time_delta[valid].median())
    return fps, stride, sample_sec

def _existing(columns, available):
    return [column for column in columns if column in available]

def build_temporal_features(frame_df):
    """Create deterministic causal features for one video.

    Rolling windows only use the current and preceding sampled rows, preventing
    future information from leaking into a prediction.
    """
    required = {"frame", "time_sec"}
    missing = required.difference(frame_df.columns)
    if missing:
        raise ValueError("Feature CSV is missing: " + ", ".join(sorted(missing)))

    source = frame_df.sort_values("frame").reset_index(drop=True).copy()
    fps, stride, sample_sec = infer_sampling(source)
    columns = {
        "frame": pd.to_numeric(source["frame"], errors="raise").astype(int),
        "time_sec": pd.to_numeric(source["time_sec"], errors="coerce"),
    }

    numeric = _existing(NUMERIC_FEATURES, source.columns)
    boolean = _existing(BOOLEAN_FEATURES, source.columns)
    categorical = _existing(CATEGORICAL_FEATURES, source.columns)
    for column in numeric:
        columns[column] = pd.to_numeric(source[column], errors="coerce")
    for column in boolean:
        columns[column] = as_boolean(source[column]).astype(float)
    for column in categorical:
        columns[column] = source[column].fillna("missing").astype(str)

    for seconds in CONTEXT_WINDOWS_SEC:
        rows = max(2, int(round(seconds / max(sample_sec, 1e-6))))
        suffix = f"{int(round(seconds * 1000))}ms"
        for column in _existing(DYNAMIC_NUMERIC_FEATURES, source.columns):
            values = pd.to_numeric(source[column], errors="coerce")
            columns[f"{column}__mean_{suffix}"] = values.rolling(
                rows, min_periods=1
            ).mean()
            columns[f"{column}__delta_{suffix}"] = values - values.shift(rows)
            if seconds >= 1.0:
                columns[f"{column}__std_{suffix}"] = values.rolling(
                    rows, min_periods=2
                ).std()
        for column in _existing(RECENT_BOOLEAN_FEATURES, source.columns):
            values = as_boolean(source[column]).astype(float)
            columns[f"{column}__recent_{suffix}"] = values.rolling(
                rows, min_periods=1
            ).max()

    output = pd.DataFrame(columns)
    output.attrs.update({
        "fps": fps,
        "stride": stride,
        "sample_interval_sec": sample_sec,
        "causal_context": True,
    })
    return output

def primary_ground_truth(frame_rows):
    labels = []
    for _, row in frame_rows.iterrows():
        label = "background"
        for behaviour in PRIMARY_PRIORITY:
            if behaviour in row.index and bool(as_boolean(pd.Series([row[behaviour]])).iloc[0]):
                label = behaviour
                break
        labels.append(label)
    return pd.Series(labels, index=frame_rows.index, dtype="object")

def attach_ground_truth(model_rows, ground_truth_frames, video_id):
    ground_truth = ground_truth_frames[
        ground_truth_frames["video_id"].astype(str) == str(video_id)
    ].copy()
    if ground_truth.empty:
        raise ValueError(f"No MERL frame labels found for video {video_id}")
    ground_truth["source_frame"] = pd.to_numeric(
        ground_truth["source_frame"], errors="coerce"
    ).astype("Int64")
    keep = [
        "source_frame", "overlap_count", "background", *BEHAVIOURS,
    ]
    keep = [column for column in keep if column in ground_truth.columns]
    merged = model_rows.merge(
        ground_truth[keep],
        left_on="frame",
        right_on="source_frame",
        how="left",
        validate="one_to_one",
    )
    if merged["source_frame"].isna().any():
        missing = int(merged["source_frame"].isna().sum())
        raise ValueError(
            f"{video_id}: {missing} sampled feature rows have no MERL label"
        )
    merged["target_label"] = primary_ground_truth(merged)
    merged["target_is_behaviour"] = (
        merged["target_label"] != "background"
    ).astype(int)
    overlap = pd.to_numeric(merged.get("overlap_count", 0), errors="coerce")
    merged["stage2_eligible"] = (
        (merged["target_label"] != "background") & (overlap <= 1)
    )
    merged.insert(0, "video_id", str(video_id))
    return merged.drop(columns=["source_frame"], errors="ignore")

def model_column_groups(table):
    excluded = {
        "video_id", "frame", "time_sec", "overlap_count", "background",
        "target_label", "target_is_behaviour", "stage2_eligible", *BEHAVIOURS,
    }
    features = [column for column in table.columns if column not in excluded]
    categorical = [
        column for column in features
        if column in CATEGORICAL_FEATURES or table[column].dtype == object
    ]
    numeric = [column for column in features if column not in categorical]
    return numeric, categorical

def balanced_sample_weights(labels):
    labels = pd.Series(labels).astype(str)
    counts = labels.value_counts()
    if counts.empty:
        return np.asarray([], dtype=float)
    total = float(len(labels))
    classes = float(len(counts))
    weights = {label: total / (classes * count) for label, count in counts.items()}
    return labels.map(weights).astype(float).to_numpy()

def smooth_probability(values, rows):
    rows = max(1, int(rows))
    return pd.Series(values, dtype=float).rolling(rows, min_periods=1).mean().to_numpy()

def bridge_short_gaps(mask, max_gap_rows):
    values = np.asarray(mask, dtype=bool).copy()
    if max_gap_rows <= 0 or not values.any():
        return values
    index = 0
    while index < len(values):
        if values[index]:
            index += 1
            continue
        start = index
        while index < len(values) and not values[index]:
            index += 1
        if (
            start > 0 and index < len(values)
            and values[start - 1] and values[index]
            and index - start <= max_gap_rows
        ):
            values[start:index] = True
    return values

def mask_runs(mask, frames, fps, min_duration_sec, max_gap_sec=0.20):
    frames = np.asarray(frames, dtype=int)
    if len(frames) == 0:
        return []
    stride = max(1, int(round(np.median(np.diff(frames))))) if len(frames) > 1 else 1
    gap_rows = max(0, int(round(max_gap_sec * fps / stride)))
    values = bridge_short_gaps(mask, gap_rows)
    minimum_rows = max(1, int(np.ceil(min_duration_sec * fps / stride)))
    runs = []
    start = None
    for index, active in enumerate(values):
        if active and start is None:
            start = index
        at_end = index == len(values) - 1
        if start is not None and ((not active) or at_end):
            end = index if active and at_end else index - 1
            if end - start + 1 >= minimum_rows:
                runs.append({
                    "start_index": start,
                    "end_index": end,
                    "start_frame": int(frames[start]),
                    "end_frame": int(frames[end]),
                })
            start = None
    return runs

def interval_iou(first, second):
    intersection = max(
        0,
        min(int(first["end_frame"]), int(second["end_frame"]))
        - max(int(first["start_frame"]), int(second["start_frame"])) + 1,
    )
    first_length = int(first["end_frame"]) - int(first["start_frame"]) + 1
    second_length = int(second["end_frame"]) - int(second["start_frame"]) + 1
    union = first_length + second_length - intersection
    return intersection / union if union else 0.0

def event_f1(ground_truth, predictions, threshold=0.30):
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            iou = interval_iou(gt, pred)
            if iou >= threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    for _, gt_index, pred_index in candidates:
        if gt_index not in used_gt and pred_index not in used_pred:
            used_gt.add(gt_index)
            used_pred.add(pred_index)
    tp = len(used_gt)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }

def load_cache_manifest(path):
    import json
    with open(Path(path), "r", encoding="utf-8") as file:
        return json.load(file)

def label_counts(table):
    counts = Counter(table["target_label"].astype(str))
    return {label: int(counts.get(label, 0)) for label in ["background", *BEHAVIOURS]}
