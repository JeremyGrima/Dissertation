import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
POSE_OUTPUT_DIR = PROJECT_DIR / "pose_tests"
LEGACY_INPUT_CSV = (
    PROJECT_DIR / "pose_tests" / "zone_pose_wrists.csv"
)
LATEST_POSE_MANIFEST = POSE_OUTPUT_DIR / "latest_zone_pose_run.json"
DEFAULT_ZONES_JSON = PROJECT_DIR / "frames" / "zones.json"

ARM_EXTENDED_RATIO_THR = 0.82
STATIONARY_SPEED_PX_SEC = 35.0
DIRECTION_SPEED_PX_SEC = 50.0
NEAR_PRODUCT_DISTANCE_PX = 25.0
BODY_NEAR_SHELF_NORM_THR = 0.90
CROUCHING_KNEE_ANGLE_DEG = 125.0
BENDING_HIP_ANGLE_DEG = 145.0
ATTENTION_TARGET_ANGLE_DEG = 55.0
TORSO_FALLBACK_ANGLE_DEG = 45.0
ATTENTION_TARGET_MARGIN_DEG = 12.0
ATTENTION_ELSEWHERE_MIN_ANGLE_DEG = 85.0
ATTENTION_MAX_SHELF_NORM_THR = 1.80
ATTENTION_MAX_SHELF_DISTANCE_PX = 450.0
ATTENTION_HANDS_NEAR_BODY_NORM = 0.85
ATTENTION_SMOOTHING_ROWS = 5
ATTENTION_MIN_SUPPORT_ROWS = 3

PRESENTATION_MAX_INTER_WRIST_NORM = 0.45
MULTI_PRODUCT_MAX_INTER_WRIST_NORM = 0.60
PRESENTATION_MAX_HEAD_DISTANCE_NORM = 0.50
PRESENTATION_MIN_ARM_EXTENSION = 0.82
PRESENTATION_MAX_FLEXED_ELBOW_DEG = 135.0

HYBRID_CORRIDOR_FILL_MAX_SEC = 0.10  # Only bridges an existing contact on the same shelf.
HYBRID_CORRIDOR_FILL_MIN_PRIMARY_ROWS = 2

REQUIRED_COLUMNS = {
    "frame",
    "time_sec",
    "left_wrist_x",
    "left_wrist_y",
    "right_wrist_x",
    "right_wrist_y"
}

OPTIONAL_NUMERIC_COLUMNS = [
    "person_box_x1",
    "person_box_y1",
    "person_box_x2",
    "person_box_y2",
    "body_center_x",
    "body_center_y",
    "nose_x",
    "nose_y",
    "nose_conf",
    "left_eye_x",
    "left_eye_y",
    "left_eye_conf",
    "right_eye_x",
    "right_eye_y",
    "right_eye_conf",
    "left_ear_x",
    "left_ear_y",
    "left_ear_conf",
    "right_ear_x",
    "right_ear_y",
    "right_ear_conf",
    "shoulder_center_x",
    "shoulder_center_y",
    "left_shoulder_x",
    "left_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_y",
    "left_elbow_x",
    "left_elbow_y",
    "right_elbow_x",
    "right_elbow_y",
    "left_hip_x",
    "left_hip_y",
    "right_hip_x",
    "right_hip_y",
    "left_knee_x",
    "left_knee_y",
    "right_knee_x",
    "right_knee_y",
    "left_ankle_x",
    "left_ankle_y",
    "right_ankle_x",
    "right_ankle_y",
    "held_product_x",  # Reserved for later held-product detections.
    "held_product_y",
    "held_product_conf"
]

def rectangle_to_polygon(points):
    (x1, y1), (x2, y2) = points
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def load_zone_groups(zones_json_path):
    with open(zones_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shelf_zones = []
    product_zones = []

    for shape in data.get("shapes", []):
        label = shape["label"]
        points = shape["points"]
        shape_type = shape.get("shape_type", "polygon")

        if shape_type == "rectangle" and len(points) == 2:
            points = rectangle_to_polygon(points)
        elif shape_type != "polygon":
            continue

        polygon = np.asarray(points, dtype=float)
        target = shelf_zones if label.startswith("Shelf_") else product_zones
        target.append((label, polygon))

    return shelf_zones, product_zones

def point_on_segment(point, start, end, tolerance=1e-6):
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    if abs(cross) > tolerance:
        return False

    return (
        min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance and
        min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance
    )

def point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test with boundary inclusion."""
    x, y = point
    inside = False

    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]

        if point_on_segment(point, start, end):
            return True

        x1, y1 = start
        x2, y2 = end

        if (y1 > y) != (y2 > y):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection_x:
                inside = not inside

    return inside

def nearest_point_on_segment(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    denominator = float(np.dot(segment, segment))

    if denominator == 0:
        return start

    proportion = float(np.dot(point - start, segment) / denominator)
    proportion = min(1.0, max(0.0, proportion))
    return start + proportion * segment

def nearest_zone(point, zones):
    """Return distance, label and closest boundary point for a zone group."""
    if point is None or not np.all(np.isfinite(point)) or not zones:
        return np.nan, None, (np.nan, np.nan)

    best_distance = float("inf")
    best_label = None
    best_point = (np.nan, np.nan)

    for label, polygon in zones:
        if point_in_polygon(point, polygon):
            return 0.0, label, (float(point[0]), float(point[1]))

        for index in range(len(polygon)):
            boundary_point = nearest_point_on_segment(
                point,
                polygon[index],
                polygon[(index + 1) % len(polygon)]
            )
            candidate_distance = float(np.linalg.norm(
                np.asarray(point, dtype=float) - boundary_point
            ))

            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_label = label
                best_point = (
                    float(boundary_point[0]),
                    float(boundary_point[1])
                )

    return best_distance, best_label, best_point

def points_in_polygon(points, polygon):
    """Vectorized point-in-polygon test for an N×2 point array."""
    x = points[:, 0]
    y = points[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    inside = np.zeros(len(points), dtype=bool)
    boundary = np.zeros(len(points), dtype=bool)

    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        on_segment = (
            (np.abs(cross) <= 1e-6) &
            (x >= min(x1, x2) - 1e-6) &
            (x <= max(x1, x2) + 1e-6) &
            (y >= min(y1, y2) - 1e-6) &
            (y <= max(y1, y2) + 1e-6)
        )
        boundary |= valid & on_segment

        if y2 == y1:
            continue

        crossing = valid & ((y1 > y) != (y2 > y))
        intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crossing & (x < intersection_x)

    return (inside | boundary) & valid

def zone_features_for_points(x_values, y_values, zones):
    """Vectorized nearest-zone features for all rows in a dataframe."""
    points = np.column_stack((x_values, y_values)).astype(float)
    valid = np.all(np.isfinite(points), axis=1)
    best_distance = np.full(len(points), np.inf, dtype=float)
    best_label = np.full(len(points), None, dtype=object)
    nearest_x = np.full(len(points), np.nan, dtype=float)
    nearest_y = np.full(len(points), np.nan, dtype=float)

    for label, polygon in zones:
        inside = points_in_polygon(points, polygon)
        newly_inside = inside & (best_distance > 0)
        best_distance[newly_inside] = 0.0
        best_label[newly_inside] = label
        nearest_x[newly_inside] = points[newly_inside, 0]
        nearest_y[newly_inside] = points[newly_inside, 1]

        for index in range(len(polygon)):
            start = polygon[index]
            end = polygon[(index + 1) % len(polygon)]
            segment = end - start
            denominator = float(np.dot(segment, segment))

            if denominator == 0:
                closest = np.repeat(start[None, :], len(points), axis=0)
            else:
                proportion = np.sum((points - start) * segment, axis=1) / denominator
                proportion = np.clip(proportion, 0.0, 1.0)
                closest = start + proportion[:, None] * segment

            candidate_distance = np.linalg.norm(points - closest, axis=1)
            update = valid & (candidate_distance < best_distance)
            best_distance[update] = candidate_distance[update]
            best_label[update] = label
            nearest_x[update] = closest[update, 0]
            nearest_y[update] = closest[update, 1]

    best_distance[~valid] = np.nan
    best_label[~valid] = None
    return best_distance, best_label, nearest_x, nearest_y

def pair_distance(x1, y1, x2, y2):
    values = np.column_stack((x1, y1, x2, y2)).astype(float)
    valid = np.all(np.isfinite(values), axis=1)
    result = np.full(len(values), np.nan, dtype=float)
    result[valid] = np.hypot(
        values[valid, 0] - values[valid, 2],
        values[valid, 1] - values[valid, 3]
    )
    return pd.Series(result, index=x1.index)

def joint_angle(ax, ay, bx, by, cx, cy):
    """Calculate angle ABC in degrees for each row."""
    values = np.column_stack((ax, ay, bx, by, cx, cy)).astype(float)
    result = np.full(len(values), np.nan, dtype=float)

    for index, row in enumerate(values):
        if not np.all(np.isfinite(row)):
            continue

        point_a = row[0:2]
        point_b = row[2:4]
        point_c = row[4:6]
        vector_1 = point_a - point_b
        vector_2 = point_c - point_b
        denominator = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

        if denominator == 0:
            continue

        cosine = np.dot(vector_1, vector_2) / denominator
        result[index] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return pd.Series(result, index=ax.index)

def vector_angle(ax, ay, bx, by, cx, cy, dx, dy):
    """Angle between vectors A→B and C→D for each row."""
    values = np.column_stack((ax, ay, bx, by, cx, cy, dx, dy)).astype(float)
    result = np.full(len(values), np.nan, dtype=float)

    for index, row in enumerate(values):
        if not np.all(np.isfinite(row)):
            continue

        vector_1 = row[2:4] - row[0:2]
        vector_2 = row[6:8] - row[4:6]
        denominator = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

        if denominator == 0:
            continue

        cosine = np.dot(vector_1, vector_2) / denominator
        result[index] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return pd.Series(result, index=ax.index)

def safe_rate(series, time_sec):
    delta_time = time_sec.diff()
    rate = series.diff() / delta_time
    valid = (
        series.notna() &
        series.shift(1).notna() &
        delta_time.notna() &
        (delta_time > 0)
    )
    return rate.where(valid)

def run_duration_seconds(condition, time_sec):
    """Duration since the start of the current consecutive True run."""
    condition = pd.Series(condition, dtype="boolean").fillna(False).to_numpy(
        dtype=bool
    )
    times = pd.to_numeric(time_sec, errors="coerce").to_numpy(dtype=float)
    durations = np.zeros(len(condition), dtype=float)

    for index in range(1, len(condition)):
        if not condition[index]:
            continue

        if condition[index - 1] and np.isfinite(times[index - 1]):
            durations[index] = durations[index - 1] + max(
                0.0,
                times[index] - times[index - 1]
            )

    return pd.Series(durations, index=time_sec.index)

def ensure_optional_columns(frame_df):
    missing_columns = []

    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column not in frame_df.columns:
            frame_df[column] = np.nan
            missing_columns.append(column)

    for wrist in ("left", "right"):
        for suffix in ("shelf_zone", "product_zone", "track_note"):
            column = f"{wrist}_{suffix}"
            if column not in frame_df.columns:
                frame_df[column] = None
                missing_columns.append(column)

        for suffix in (
            "corridor_product_zone",
            "corridor_product_alternative",
            "corridor_product_candidates",
            "corridor_shelf_zone",
            "corridor_raw_shelf_zone",
        ):
            column = f"{wrist}_{suffix}"
            if column not in frame_df.columns:
                frame_df[column] = None
                missing_columns.append(column)
        for suffix in (
            "corridor_product_score",
            "corridor_product_margin",
        ):
            column = f"{wrist}_{suffix}"
            if column not in frame_df.columns:
                frame_df[column] = np.nan
                missing_columns.append(column)
        ambiguous_column = f"{wrist}_corridor_product_ambiguous"
        if ambiguous_column not in frame_df.columns:
            frame_df[ambiguous_column] = False
            missing_columns.append(ambiguous_column)

    if "held_product_detected" not in frame_df.columns:
        if {"held_product_x", "held_product_y"}.issubset(frame_df.columns):
            frame_df["held_product_detected"] = frame_df[
                ["held_product_x", "held_product_y"]
            ].notna().all(axis=1)
        else:
            frame_df["held_product_detected"] = False
        missing_columns.append("held_product_detected")

    if "held_product_origin_zone" not in frame_df.columns:
        frame_df["held_product_origin_zone"] = None
        missing_columns.append("held_product_origin_zone")

    return missing_columns

def fill_derived_centres(features):
    shoulder_x = features[["left_shoulder_x", "right_shoulder_x"]].mean(
        axis=1,
        skipna=True
    )
    shoulder_y = features[["left_shoulder_y", "right_shoulder_y"]].mean(
        axis=1,
        skipna=True
    )
    both_shoulders = (
        features["left_shoulder_x"].notna() &
        features["left_shoulder_y"].notna() &
        features["right_shoulder_x"].notna() &
        features["right_shoulder_y"].notna()
    )
    features.loc[
        features["shoulder_center_x"].isna() & both_shoulders,
        "shoulder_center_x"
    ] = shoulder_x
    features.loc[
        features["shoulder_center_y"].isna() & both_shoulders,
        "shoulder_center_y"
    ] = shoulder_y

    hip_x = features[["left_hip_x", "right_hip_x"]].mean(axis=1, skipna=True)
    hip_y = features[["left_hip_y", "right_hip_y"]].mean(axis=1, skipna=True)
    both_hips = (
        features["left_hip_x"].notna() &
        features["left_hip_y"].notna() &
        features["right_hip_x"].notna() &
        features["right_hip_y"].notna()
    )
    features["hip_center_x"] = hip_x.where(both_hips)
    features["hip_center_y"] = hip_y.where(both_hips)
    features.loc[
        features["body_center_x"].isna() & both_hips,
        "body_center_x"
    ] = features["hip_center_x"]
    features.loc[
        features["body_center_y"].isna() & both_hips,
        "body_center_y"
    ] = features["hip_center_y"]

    box_complete = features[
        ["person_box_x1", "person_box_y1", "person_box_x2", "person_box_y2"]
    ].notna().all(axis=1)
    box_center_x = (features["person_box_x1"] + features["person_box_x2"]) / 2
    box_center_y = (features["person_box_y1"] + features["person_box_y2"]) / 2
    features.loc[
        features["body_center_x"].isna() & box_complete,
        "body_center_x"
    ] = box_center_x
    features.loc[
        features["body_center_y"].isna() & box_complete,
        "body_center_y"
    ] = box_center_y

def add_zone_distance_features(features, x_col, y_col, prefix, zones):
    distance_values, labels, nearest_x, nearest_y = zone_features_for_points(
        features[x_col],
        features[y_col],
        zones
    )
    features[f"{prefix}_distance_px"] = distance_values
    features[f"{prefix}_nearest_zone"] = labels
    features[f"{prefix}_nearest_x"] = nearest_x
    features[f"{prefix}_nearest_y"] = nearest_y

def hybrid_contact_zones(
    single_shelf,
    single_product,
    corridor_shelf,
    corridor_product,
    time_sec,
):
    """Let single-point contact start episodes and corridor evidence fill gaps."""
    shelf_output = single_shelf.copy()
    product_output = single_product.copy()
    source = pd.Series(None, index=single_shelf.index, dtype=object)
    corridor_fill = pd.Series(False, index=single_shelf.index, dtype=bool)
    active_shelf = None
    last_single_time = None
    primary_streak = 0

    def present(value):
        return not pd.isna(value) and str(value).strip() not in {"", "None"}

    for index in single_shelf.index:
        shelf_value = single_shelf.at[index]
        product_value = single_product.at[index]
        timestamp = pd.to_numeric(time_sec.at[index], errors="coerce")
        primary_complete = present(shelf_value) and present(product_value)

        if primary_complete:
            if active_shelf == str(shelf_value):
                primary_streak += 1
            else:
                primary_streak = 1
            source.at[index] = "single_point"
            active_shelf = str(shelf_value)
            last_single_time = timestamp
            continue

        corridor_shelf_value = corridor_shelf.at[index]
        corridor_product_value = corridor_product.at[index]
        elapsed = (
            float(timestamp - last_single_time)
            if (
                active_shelf is not None
                and last_single_time is not None
                and pd.notna(timestamp)
                and pd.notna(last_single_time)
            )
            else np.inf
        )
        valid_fill = (
            active_shelf is not None
            and primary_streak >= HYBRID_CORRIDOR_FILL_MIN_PRIMARY_ROWS
            and elapsed >= 0.0
            and elapsed <= HYBRID_CORRIDOR_FILL_MAX_SEC + 1e-9
            and present(corridor_shelf_value)
            and present(corridor_product_value)
            and str(corridor_shelf_value) == active_shelf
        )
        if valid_fill:
            shelf_output.at[index] = corridor_shelf_value
            product_output.at[index] = corridor_product_value
            source.at[index] = "corridor_fill"
            corridor_fill.at[index] = True
            continue

        if present(shelf_value) or present(product_value):  # Keep partial values for distance and dwell checks.
            source.at[index] = "single_point_partial"
        else:
            shelf_output.at[index] = None
            product_output.at[index] = None
        active_shelf = None
        last_single_time = None
        primary_streak = 0

    return shelf_output, product_output, source, corridor_fill

def add_wrist_features(features, wrist, shelf_zones, product_zones):
    wrist_x = f"{wrist}_wrist_x"
    wrist_y = f"{wrist}_wrist_y"
    shoulder_x = f"{wrist}_shoulder_x"
    shoulder_y = f"{wrist}_shoulder_y"
    elbow_x = f"{wrist}_elbow_x"
    elbow_y = f"{wrist}_elbow_y"

    features[f"{wrist}_wrist_observed"] = features[
        f"{wrist}_track_note"
    ].isin(["tracked", "acquired", "reacquired"])
    features[f"{wrist}_wrist_held"] = features[
        f"{wrist}_track_note"
    ].fillna("").str.startswith("held")

    features[f"{wrist}_wrist_vx_px_sec"] = safe_rate(
        features[wrist_x], features["time_sec"]
    )
    features[f"{wrist}_wrist_vy_px_sec"] = safe_rate(
        features[wrist_y], features["time_sec"]
    )
    features[f"{wrist}_wrist_speed_px_sec"] = np.hypot(
        features[f"{wrist}_wrist_vx_px_sec"],
        features[f"{wrist}_wrist_vy_px_sec"]
    )
    features[f"{wrist}_wrist_acceleration_px_sec2"] = safe_rate(
        features[f"{wrist}_wrist_speed_px_sec"],
        features["time_sec"]
    )

    features[f"{wrist}_upper_arm_length_px"] = pair_distance(
        features[shoulder_x],
        features[shoulder_y],
        features[elbow_x],
        features[elbow_y]
    )
    features[f"{wrist}_forearm_length_px"] = pair_distance(
        features[elbow_x],
        features[elbow_y],
        features[wrist_x],
        features[wrist_y]
    )
    features[f"{wrist}_shoulder_to_wrist_px"] = pair_distance(
        features[shoulder_x],
        features[shoulder_y],
        features[wrist_x],
        features[wrist_y]
    )
    arm_path_length = (
        features[f"{wrist}_upper_arm_length_px"] +
        features[f"{wrist}_forearm_length_px"]
    )
    features[f"{wrist}_arm_extension_ratio"] = (
        features[f"{wrist}_shoulder_to_wrist_px"] / arm_path_length
    ).where(arm_path_length > 0)
    features[f"{wrist}_arm_extended"] = (
        features[f"{wrist}_arm_extension_ratio"] >= ARM_EXTENDED_RATIO_THR
    ).where(features[f"{wrist}_arm_extension_ratio"].notna())

    features[f"{wrist}_wrist_to_body_px"] = pair_distance(
        features[wrist_x],
        features[wrist_y],
        features["body_center_x"],
        features["body_center_y"]
    )
    features[f"{wrist}_wrist_outward_speed_px_sec"] = safe_rate(
        features[f"{wrist}_wrist_to_body_px"],
        features["time_sec"]
    )
    features[f"{wrist}_wrist_toward_body_speed_px_sec"] = -features[
        f"{wrist}_wrist_outward_speed_px_sec"
    ]

    add_zone_distance_features(
        features, wrist_x, wrist_y, f"{wrist}_wrist_to_shelf", shelf_zones
    )
    add_zone_distance_features(
        features, wrist_x, wrist_y, f"{wrist}_wrist_to_product", product_zones
    )
    features[f"{wrist}_wrist_toward_shelf_speed_px_sec"] = -safe_rate(
        features[f"{wrist}_wrist_to_shelf_distance_px"],
        features["time_sec"]
    )

    shelf_computed = features[f"{wrist}_wrist_to_shelf_nearest_zone"].where(
        features[f"{wrist}_wrist_to_shelf_distance_px"] == 0
    )
    product_computed = features[f"{wrist}_wrist_to_product_nearest_zone"].where(
        features[f"{wrist}_wrist_to_product_distance_px"] == 0
    )
    legacy_shelf = features[f"{wrist}_shelf_zone"].combine_first(
        shelf_computed
    )
    legacy_product = features[f"{wrist}_product_zone"].combine_first(
        product_computed
    )
    features[f"{wrist}_single_point_shelf_zone"] = legacy_shelf
    features[f"{wrist}_single_point_product_zone"] = legacy_product

    (  # The hand point starts contact; the corridor only fills a short gap.
        features[f"{wrist}_shelf_zone"],
        features[f"{wrist}_product_zone"],
        features[f"{wrist}_zone_assignment_source"],
        features[f"{wrist}_corridor_gap_fill"],
    ) = hybrid_contact_zones(
        legacy_shelf,
        legacy_product,
        features[f"{wrist}_corridor_shelf_zone"],
        features[f"{wrist}_corridor_product_zone"],
        features["time_sec"],
    )

    in_shelf = features[f"{wrist}_shelf_zone"].notna()
    in_product = features[f"{wrist}_product_zone"].notna()
    previous_in_shelf = in_shelf.shift(1, fill_value=False)
    previous_in_product = in_product.shift(1, fill_value=False)
    features[f"{wrist}_in_shelf"] = in_shelf
    features[f"{wrist}_in_product"] = in_product
    features[f"{wrist}_shelf_entry"] = in_shelf & ~previous_in_shelf
    features[f"{wrist}_shelf_exit"] = ~in_shelf & previous_in_shelf
    features[f"{wrist}_product_entry"] = in_product & ~previous_in_product
    features[f"{wrist}_product_exit"] = ~in_product & previous_in_product
    features[f"{wrist}_shelf_dwell_sec"] = run_duration_seconds(
        in_shelf, features["time_sec"]
    )
    features[f"{wrist}_product_dwell_sec"] = run_duration_seconds(
        in_product, features["time_sec"]
    )
    features[f"{wrist}_near_product"] = (
        features[f"{wrist}_wrist_to_product_distance_px"] <=
        NEAR_PRODUCT_DISTANCE_PX
    ).where(features[f"{wrist}_wrist_to_product_distance_px"].notna())

    speed = features[f"{wrist}_wrist_speed_px_sec"]
    toward_shelf = features[f"{wrist}_wrist_toward_shelf_speed_px_sec"]
    toward_body = features[f"{wrist}_wrist_toward_body_speed_px_sec"]
    motion_state = pd.Series("unknown", index=features.index, dtype="object")
    motion_state.loc[speed.notna() & (speed <= STATIONARY_SPEED_PX_SEC)] = (
        "stationary"
    )
    motion_state.loc[
        speed.notna() &
        (speed > STATIONARY_SPEED_PX_SEC) &
        (toward_shelf < DIRECTION_SPEED_PX_SEC) &
        (toward_body < DIRECTION_SPEED_PX_SEC)
    ] = "moving"
    motion_state.loc[
        speed.notna() & (toward_shelf >= DIRECTION_SPEED_PX_SEC)
    ] = "toward_shelf"
    motion_state.loc[
        speed.notna() &
        (toward_body >= DIRECTION_SPEED_PX_SEC) &
        (toward_body > toward_shelf.fillna(-np.inf))
    ] = "toward_body"
    features[f"{wrist}_wrist_motion_state"] = motion_state

def add_body_and_posture_features(features, shelf_zones):
    features["person_box_width_px"] = (
        features["person_box_x2"] - features["person_box_x1"]
    )
    features["person_box_height_px"] = (
        features["person_box_y2"] - features["person_box_y1"]
    )
    features["person_box_diagonal_px"] = np.hypot(
        features["person_box_width_px"],
        features["person_box_height_px"]
    )

    add_zone_distance_features(
        features,
        "body_center_x",
        "body_center_y",
        "body_to_shelf",
        shelf_zones
    )
    features["body_to_shelf_norm"] = (
        features["body_to_shelf_distance_px"] /
        features["person_box_height_px"]
    ).where(features["person_box_height_px"] > 0)
    features["body_near_shelf"] = (
        features["body_to_shelf_norm"] <= BODY_NEAR_SHELF_NORM_THR
    ).where(features["body_to_shelf_norm"].notna())
    features["body_near_shelf_dwell_sec"] = run_duration_seconds(
        features["body_near_shelf"],
        features["time_sec"]
    )

    features["left_hip_angle_deg"] = joint_angle(
        features["left_shoulder_x"],
        features["left_shoulder_y"],
        features["left_hip_x"],
        features["left_hip_y"],
        features["left_knee_x"],
        features["left_knee_y"]
    )
    features["right_hip_angle_deg"] = joint_angle(
        features["right_shoulder_x"],
        features["right_shoulder_y"],
        features["right_hip_x"],
        features["right_hip_y"],
        features["right_knee_x"],
        features["right_knee_y"]
    )
    features["left_knee_angle_deg"] = joint_angle(
        features["left_hip_x"],
        features["left_hip_y"],
        features["left_knee_x"],
        features["left_knee_y"],
        features["left_ankle_x"],
        features["left_ankle_y"]
    )
    features["right_knee_angle_deg"] = joint_angle(
        features["right_hip_x"],
        features["right_hip_y"],
        features["right_knee_x"],
        features["right_knee_y"],
        features["right_ankle_x"],
        features["right_ankle_y"]
    )
    features["mean_hip_angle_deg"] = features[
        ["left_hip_angle_deg", "right_hip_angle_deg"]
    ].mean(axis=1, skipna=True)
    features["mean_knee_angle_deg"] = features[
        ["left_knee_angle_deg", "right_knee_angle_deg"]
    ].mean(axis=1, skipna=True)
    hip_known = features[
        ["left_hip_angle_deg", "right_hip_angle_deg"]
    ].notna().any(axis=1)
    knee_known = features[
        ["left_knee_angle_deg", "right_knee_angle_deg"]
    ].notna().any(axis=1)
    posture_state = pd.Series("unknown", index=features.index, dtype="object")
    posture_state.loc[hip_known | knee_known] = "standing"
    posture_state.loc[
        hip_known & (features["mean_hip_angle_deg"] < BENDING_HIP_ANGLE_DEG)
    ] = "bending"
    posture_state.loc[
        knee_known &
        (features["mean_knee_angle_deg"] < CROUCHING_KNEE_ANGLE_DEG)
    ] = "crouching"
    features["posture_state"] = posture_state

def add_normalized_features(features):
    scale = features["person_box_height_px"].where(
        features["person_box_height_px"] > 0
    )

    for wrist in ("left", "right"):
        features[f"{wrist}_wrist_to_body_norm"] = (
            features[f"{wrist}_wrist_to_body_px"] / scale
        )
        features[f"{wrist}_wrist_to_shelf_norm"] = (
            features[f"{wrist}_wrist_to_shelf_distance_px"] / scale
        )
        features[f"{wrist}_wrist_to_product_norm"] = (
            features[f"{wrist}_wrist_to_product_distance_px"] / scale
        )

def row_point(row, x_column, y_column):
    x = row.get(x_column, np.nan)
    y = row.get(y_column, np.nan)
    if pd.isna(x) or pd.isna(y):
        return None
    return np.asarray([float(x), float(y)], dtype=float)

def categorical_run_duration(values, time_sec):
    """Duration of the current unchanged, non-uncertain target run."""
    labels = pd.Series(values, dtype="object").fillna("uncertain").to_numpy()
    times = pd.to_numeric(time_sec, errors="coerce").to_numpy(dtype=float)
    durations = np.zeros(len(labels), dtype=float)

    for index in range(1, len(labels)):
        if labels[index] == "uncertain" or labels[index] != labels[index - 1]:
            continue
        if np.isfinite(times[index]) and np.isfinite(times[index - 1]):
            durations[index] = durations[index - 1] + max(
                0.0, times[index] - times[index - 1]
            )

    return pd.Series(durations, index=time_sec.index)

def add_head_direction_features(features):
    """Estimate 2D head direction, using the torso only as a last fallback."""
    count = len(features)
    head_start = np.full((count, 2), np.nan, dtype=float)
    head_end = np.full((count, 2), np.nan, dtype=float)
    attention_start = np.full((count, 2), np.nan, dtype=float)
    attention_end = np.full((count, 2), np.nan, dtype=float)
    head_method = np.full(count, "unavailable", dtype=object)
    attention_method = np.full(count, "unavailable", dtype=object)
    quality = np.zeros(count, dtype=float)

    for index, row in features.iterrows():
        nose = row_point(row, "nose_x", "nose_y")
        left_eye = row_point(row, "left_eye_x", "left_eye_y")
        right_eye = row_point(row, "right_eye_x", "right_eye_y")
        left_ear = row_point(row, "left_ear_x", "left_ear_y")
        right_ear = row_point(row, "right_ear_x", "right_ear_y")
        shoulder = row_point(row, "shoulder_center_x", "shoulder_center_y")
        hip = row_point(row, "hip_center_x", "hip_center_y")
        box_height = row.get("person_box_height_px", np.nan)
        min_length = max(
            3.0,
            0.015 * float(box_height) if pd.notna(box_height) else 3.0
        )

        start = None
        end = None
        method = "unavailable"
        method_quality = 0.0

        if nose is not None and left_ear is not None and right_ear is not None:
            start = (left_ear + right_ear) / 2.0
            if left_eye is not None and right_eye is not None:
                eye_midpoint = (left_eye + right_eye) / 2.0
                end = 0.70 * nose + 0.30 * eye_midpoint
                method = "face_ears_eyes_nose"
                method_quality = 1.0
            else:
                end = nose
                method = "face_ears_nose"
                method_quality = 0.90
        elif nose is not None and left_eye is not None and right_eye is not None:
            start = (left_eye + right_eye) / 2.0
            end = nose
            method = "face_eyes_nose"
            method_quality = 0.82
        elif nose is not None:
            visible_face_points = [
                point for point in (left_eye, right_eye, left_ear, right_ear)
                if point is not None
            ]
            if len(visible_face_points) >= 2:
                start = np.mean(visible_face_points, axis=0)
                end = nose
                method = "face_partial_nose"
                method_quality = 0.70
            elif shoulder is not None:
                start = shoulder
                end = nose
                method = "shoulders_to_nose"
                method_quality = 0.58

        if (
            start is not None and end is not None and
            np.linalg.norm(end - start) >= min_length
        ):
            head_start[index] = start
            head_end[index] = end
            attention_start[index] = start
            attention_end[index] = end
            head_method[index] = method
            attention_method[index] = method
            quality[index] = method_quality
        elif hip is not None and shoulder is not None:
            attention_start[index] = hip  # Body orientation is a weaker cue than direct gaze.
            attention_end[index] = shoulder
            attention_method[index] = "torso_fallback"
            quality[index] = 0.35

    features["head_direction_start_x"] = head_start[:, 0]
    features["head_direction_start_y"] = head_start[:, 1]
    features["head_direction_end_x"] = head_end[:, 0]
    features["head_direction_end_y"] = head_end[:, 1]
    features["head_direction_method"] = head_method
    features["attention_direction_start_x"] = attention_start[:, 0]
    features["attention_direction_start_y"] = attention_start[:, 1]
    features["attention_direction_end_x"] = attention_end[:, 0]
    features["attention_direction_end_y"] = attention_end[:, 1]
    features["attention_direction_method"] = attention_method
    features["attention_direction_quality"] = quality

def smooth_attention_targets(raw_targets, raw_scores):
    """Use local persistent evidence without carrying a target indefinitely."""
    targets = pd.Series(raw_targets, dtype="object").fillna("uncertain")
    scores = pd.to_numeric(raw_scores, errors="coerce").fillna(0.0)
    radius = ATTENTION_SMOOTHING_ROWS // 2
    smoothed = []
    smoothed_scores = []

    for index in range(len(targets)):
        start = max(0, index - radius)
        end = min(len(targets), index + radius + 1)
        window_targets = targets.iloc[start:end]
        window_scores = scores.iloc[start:end]
        known = window_targets != "uncertain"

        if not known.any():
            smoothed.append("uncertain")
            smoothed_scores.append(0.0)
            continue

        candidates = window_targets[known].unique()
        totals = {
            candidate: float(window_scores[window_targets == candidate].sum())
            for candidate in candidates
        }
        winner = max(totals, key=totals.get)
        winner_rows = int((window_targets == winner).sum())
        total_weight = sum(totals.values())
        weight_share = totals[winner] / total_weight if total_weight > 0 else 0.0
        required_rows = min(ATTENTION_MIN_SUPPORT_ROWS, end - start)

        if winner_rows >= required_rows and weight_share >= 0.60:
            smoothed.append(winner)
            winner_scores = window_scores[window_targets == winner]
            smoothed_scores.append(float(winner_scores.mean()) * weight_share)
        else:
            smoothed.append("uncertain")
            smoothed_scores.append(0.0)

    return (
        pd.Series(smoothed, index=targets.index, dtype="object"),
        pd.Series(smoothed_scores, index=targets.index, dtype=float)
    )

def add_attention_features(features, shelf_zones):
    """Estimate shelf/hands/product attention targets from overhead pose."""
    left_valid = features[["left_wrist_x", "left_wrist_y"]].notna().all(axis=1)
    right_valid = features[["right_wrist_x", "right_wrist_y"]].notna().all(axis=1)
    both_valid = left_valid & right_valid
    features["wrist_midpoint_x"] = np.nan
    features["wrist_midpoint_y"] = np.nan
    features["wrist_midpoint_method"] = "unavailable"
    features.loc[both_valid, "wrist_midpoint_x"] = (
        features.loc[both_valid, "left_wrist_x"] +
        features.loc[both_valid, "right_wrist_x"]
    ) / 2.0
    features.loc[both_valid, "wrist_midpoint_y"] = (
        features.loc[both_valid, "left_wrist_y"] +
        features.loc[both_valid, "right_wrist_y"]
    ) / 2.0
    features.loc[both_valid, "wrist_midpoint_method"] = "both_wrists"

    only_left = left_valid & ~right_valid
    only_right = right_valid & ~left_valid
    features.loc[only_left, "wrist_midpoint_x"] = features.loc[
        only_left, "left_wrist_x"
    ]
    features.loc[only_left, "wrist_midpoint_y"] = features.loc[
        only_left, "left_wrist_y"
    ]
    features.loc[only_left, "wrist_midpoint_method"] = "left_wrist_only"
    features.loc[only_right, "wrist_midpoint_x"] = features.loc[
        only_right, "right_wrist_x"
    ]
    features.loc[only_right, "wrist_midpoint_y"] = features.loc[
        only_right, "right_wrist_y"
    ]
    features.loc[only_right, "wrist_midpoint_method"] = "right_wrist_only"

    features["wrist_midpoint_to_body_px"] = pair_distance(
        features["wrist_midpoint_x"],
        features["wrist_midpoint_y"],
        features["body_center_x"],
        features["body_center_y"]
    )
    scale = features["person_box_height_px"].where(
        features["person_box_height_px"] > 0
    )
    features["wrist_midpoint_to_body_norm"] = (
        features["wrist_midpoint_to_body_px"] / scale
    )
    features["hands_near_body"] = (
        features["wrist_midpoint_to_body_norm"] <=
        ATTENTION_HANDS_NEAR_BODY_NORM
    ).where(features["wrist_midpoint_to_body_norm"].notna())

    features["inter_wrist_distance_px"] = pair_distance(
        features["left_wrist_x"],
        features["left_wrist_y"],
        features["right_wrist_x"],
        features["right_wrist_y"]
    )
    features["inter_wrist_distance_norm"] = (
        features["inter_wrist_distance_px"] / scale
    )
    features["left_elbow_angle_deg"] = joint_angle(
        features["left_shoulder_x"],
        features["left_shoulder_y"],
        features["left_elbow_x"],
        features["left_elbow_y"],
        features["left_wrist_x"],
        features["left_wrist_y"]
    )
    features["right_elbow_angle_deg"] = joint_angle(
        features["right_shoulder_x"],
        features["right_shoulder_y"],
        features["right_elbow_x"],
        features["right_elbow_y"],
        features["right_wrist_x"],
        features["right_wrist_y"]
    )

    add_head_direction_features(features)
    nose_available = features[["nose_x", "nose_y"]].notna().all(axis=1)
    features["head_reference_x"] = features["nose_x"].where(
        nose_available, features["attention_direction_end_x"]
    )
    features["head_reference_y"] = features["nose_y"].where(
        nose_available, features["attention_direction_end_y"]
    )
    features["head_reference_method"] = np.where(
        nose_available, "nose", features["attention_direction_method"]
    )
    features["wrist_midpoint_to_head_px"] = pair_distance(
        features["wrist_midpoint_x"],
        features["wrist_midpoint_y"],
        features["head_reference_x"],
        features["head_reference_y"]
    )
    features["wrist_midpoint_to_head_norm"] = (
        features["wrist_midpoint_to_head_px"] / scale
    )

    both_wrists_observed = (
        features["left_wrist_observed"].fillna(False).astype(bool) &
        features["right_wrist_observed"].fillna(False).astype(bool) &
        both_valid
    )
    features["both_wrists_observed"] = both_wrists_observed
    features["hands_together"] = (
        features["inter_wrist_distance_norm"] <=
        PRESENTATION_MAX_INTER_WRIST_NORM
    ).where(features["inter_wrist_distance_norm"].notna())
    features["hands_near_head"] = (
        features["wrist_midpoint_to_head_norm"] <=
        PRESENTATION_MAX_HEAD_DISTANCE_NORM
    ).where(features["wrist_midpoint_to_head_norm"].notna())
    maximum_extension = features[[
        "left_arm_extension_ratio", "right_arm_extension_ratio"
    ]].max(axis=1, skipna=True)
    minimum_elbow_angle = features[[
        "left_elbow_angle_deg", "right_elbow_angle_deg"
    ]].min(axis=1, skipna=True)
    features["presentation_arm_support"] = (
        (maximum_extension >= PRESENTATION_MIN_ARM_EXTENSION) |
        (minimum_elbow_angle <= PRESENTATION_MAX_FLEXED_ELBOW_DEG)
    ).where(maximum_extension.notna() | minimum_elbow_angle.notna())
    hands_together = features["hands_together"].astype("boolean").fillna(False)
    hands_near_head = features["hands_near_head"].astype("boolean").fillna(False)
    presentation_arm_support = (
        features["presentation_arm_support"]
        .astype("boolean")
        .fillna(False)
    )
    features["product_presentation_pose"] = (
        both_wrists_observed &
        hands_together &
        hands_near_head &
        presentation_arm_support
    )
    features["multi_product_holding_pose"] = (  # Allows wider wrists when two pickups support it.
        both_wrists_observed &
        (features["inter_wrist_distance_norm"] <=
         MULTI_PRODUCT_MAX_INTER_WRIST_NORM).fillna(False) &
        hands_near_head &
        presentation_arm_support
    )

    wrist_closeness_support = np.clip(
        1.0 - (
            features["inter_wrist_distance_norm"] /
            PRESENTATION_MAX_INTER_WRIST_NORM
        ),
        0.0,
        1.0
    ).fillna(0.0)
    head_proximity_support = np.clip(
        1.0 - (
            features["wrist_midpoint_to_head_norm"] /
            PRESENTATION_MAX_HEAD_DISTANCE_NORM
        ),
        0.0,
        1.0
    ).fillna(0.0)
    extension_support = np.clip(
        (maximum_extension - 0.70) / 0.30, 0.0, 1.0
    ).fillna(0.0)
    flexion_support = np.clip(
        (150.0 - minimum_elbow_angle) / 60.0, 0.0, 1.0
    ).fillna(0.0)
    arm_support = np.maximum(extension_support, flexion_support)
    features["product_presentation_score"] = (
        0.30 * wrist_closeness_support +
        0.35 * head_proximity_support +
        0.20 * arm_support +
        0.15 * both_wrists_observed.astype(float)
    ).where(features["product_presentation_pose"], 0.0)
    features["product_presentation_dwell_sec"] = run_duration_seconds(
        features["product_presentation_pose"], features["time_sec"]
    )
    multi_product_closeness = np.clip(
        1.0 - (
            features["inter_wrist_distance_norm"] /
            MULTI_PRODUCT_MAX_INTER_WRIST_NORM
        ),
        0.0,
        1.0
    ).fillna(0.0)
    features["multi_product_holding_score"] = (
        0.25 * multi_product_closeness +
        0.35 * head_proximity_support +
        0.25 * arm_support +
        0.15 * both_wrists_observed.astype(float)
    ).where(features["multi_product_holding_pose"], 0.0)
    features["multi_product_holding_dwell_sec"] = run_duration_seconds(
        features["multi_product_holding_pose"], features["time_sec"]
    )

    add_zone_distance_features(
        features,
        "attention_direction_start_x",
        "attention_direction_start_y",
        "attention_origin_to_shelf",
        shelf_zones
    )
    features["attention_shelf_distance_norm"] = (
        features["attention_origin_to_shelf_distance_px"] / scale
    )

    target_arguments = {
        "shelf": (
            features["attention_origin_to_shelf_nearest_x"],
            features["attention_origin_to_shelf_nearest_y"]
        ),
        "hands": (
            features["wrist_midpoint_x"],
            features["wrist_midpoint_y"]
        ),
        "held_product": (
            features["held_product_x"],
            features["held_product_y"]
        )
    }
    for target, (target_x, target_y) in target_arguments.items():
        features[f"head_to_{target}_angle_deg"] = vector_angle(
            features["head_direction_start_x"],
            features["head_direction_start_y"],
            features["head_direction_end_x"],
            features["head_direction_end_y"],
            features["head_direction_start_x"],
            features["head_direction_start_y"],
            target_x,
            target_y
        )
        features[f"torso_to_{target}_angle_deg"] = vector_angle(
            features["hip_center_x"],
            features["hip_center_y"],
            features["shoulder_center_x"],
            features["shoulder_center_y"],
            features["hip_center_x"],
            features["hip_center_y"],
            target_x,
            target_y
        )
        features[f"attention_to_{target}_angle_deg"] = vector_angle(
            features["attention_direction_start_x"],
            features["attention_direction_start_y"],
            features["attention_direction_end_x"],
            features["attention_direction_end_y"],
            features["attention_direction_start_x"],
            features["attention_direction_start_y"],
            target_x,
            target_y
        )

    detected_values = features["held_product_detected"].map({
        True: True, False: False, 1: True, 0: False,
        "True": True, "False": False, "true": True, "false": False
    }).fillna(False)
    features["held_product_available"] = (
        detected_values.astype(bool) &
        features[["held_product_x", "held_product_y"]].notna().all(axis=1)
    )

    raw_targets = []
    raw_scores = []
    for index, row in features.iterrows():
        method = str(row["attention_direction_method"])
        quality = float(row["attention_direction_quality"])
        if method == "unavailable":
            raw_targets.append("uncertain")
            raw_scores.append(0.0)
            continue

        angle_threshold = (
            TORSO_FALLBACK_ANGLE_DEG
            if method == "torso_fallback"
            else ATTENTION_TARGET_ANGLE_DEG
        )
        shelf_angle = row["attention_to_shelf_angle_deg"]
        hands_angle = row["attention_to_hands_angle_deg"]
        held_angle = row["attention_to_held_product_angle_deg"]
        shelf_norm = row["attention_shelf_distance_norm"]
        shelf_px = row["attention_origin_to_shelf_distance_px"]
        hands_near = bool(row["hands_near_body"]) if pd.notna(
            row["hands_near_body"]
        ) else False
        held_available = bool(row["held_product_available"])
        shelf_distance_ok = (
            (pd.notna(shelf_norm) and
             float(shelf_norm) <= ATTENTION_MAX_SHELF_NORM_THR) or
            (pd.notna(shelf_px) and
             float(shelf_px) <= ATTENTION_MAX_SHELF_DISTANCE_PX)
        )
        shelf_ok = (
            shelf_distance_ok and pd.notna(shelf_angle) and
            float(shelf_angle) <= angle_threshold
        )
        hands_ok = (
            hands_near and pd.notna(hands_angle) and
            float(hands_angle) <= angle_threshold
        )
        held_ok = (
            held_available and pd.notna(held_angle) and
            float(held_angle) <= angle_threshold
        )

        target = "uncertain"
        target_angle = np.nan
        if held_ok and (
            not hands_ok or float(held_angle) <= float(hands_angle) + 5.0
        ):
            target = "held_product"
            target_angle = float(held_angle)
        elif hands_ok and (
            not shelf_ok or
            float(hands_angle) + ATTENTION_TARGET_MARGIN_DEG <=
            float(shelf_angle)
        ):
            target = "hands"
            target_angle = float(hands_angle)
        elif shelf_ok and (
            not hands_ok or
            float(shelf_angle) + ATTENTION_TARGET_MARGIN_DEG <=
            float(hands_angle)
        ):
            target = "shelf"
            target_angle = float(shelf_angle)
        elif hands_ok and shelf_ok:
            target = "uncertain"  # Both targets are plausible, so do not force one.
        else:
            known_angles = [
                float(angle) for angle in (shelf_angle, hands_angle, held_angle)
                if pd.notna(angle)
            ]
            if (
                method != "torso_fallback" and known_angles and
                min(known_angles) >= ATTENTION_ELSEWHERE_MIN_ANGLE_DEG
            ):
                target = "elsewhere"
                target_angle = min(known_angles)

        if target == "uncertain":
            score = 0.0
        elif target == "elsewhere":
            score = quality * min(target_angle / 120.0, 1.0)
        else:
            angular_support = max(
                0.0, 1.0 - target_angle / max(angle_threshold, 1.0)
            )
            score = quality * (0.55 + 0.45 * angular_support)

        raw_targets.append(target)
        raw_scores.append(float(np.clip(score, 0.0, 1.0)))

    features["attention_target_raw"] = raw_targets
    features["attention_target_score_raw"] = raw_scores
    smoothed_target, smoothed_score = smooth_attention_targets(
        features["attention_target_raw"],
        features["attention_target_score_raw"]
    )
    features["attention_target"] = smoothed_target
    features["attention_target_score"] = smoothed_score
    features["attention_target_dwell_sec"] = categorical_run_duration(
        features["attention_target"], features["time_sec"]
    )

    known_target = features["attention_target"] != "uncertain"
    shelf_proxy = pd.Series(pd.NA, index=features.index, dtype="boolean")
    hands_proxy = pd.Series(pd.NA, index=features.index, dtype="boolean")
    shelf_proxy.loc[known_target] = (
        features.loc[known_target, "attention_target"] == "shelf"
    )
    hands_proxy.loc[known_target] = features.loc[
        known_target, "attention_target"
    ].isin(["hands", "held_product"])
    features["attention_to_shelf_proxy"] = shelf_proxy
    features["attention_to_hands_proxy"] = hands_proxy
    features["attention_to_shelf_dwell_sec"] = run_duration_seconds(
        shelf_proxy, features["time_sec"]
    )
    features["attention_to_hands_dwell_sec"] = run_duration_seconds(
        hands_proxy, features["time_sec"]
    )

def extract_behaviour_features(frame_df, shelf_zones, product_zones):
    missing_required = REQUIRED_COLUMNS.difference(frame_df.columns)
    if missing_required:
        raise ValueError(
            "Input pose CSV is missing required columns: " +
            ", ".join(sorted(missing_required))
        )

    features = frame_df.copy()
    features = features.sort_values("frame").reset_index(drop=True)
    features["frame"] = pd.to_numeric(features["frame"], errors="coerce")
    features["time_sec"] = pd.to_numeric(features["time_sec"], errors="coerce")
    missing_optional = ensure_optional_columns(features)
    fill_derived_centres(features)
    add_body_and_posture_features(features, shelf_zones)

    for wrist in ("left", "right"):
        features = features.copy()  # Avoids pandas fragmentation warnings.
        add_wrist_features(features, wrist, shelf_zones, product_zones)

    add_normalized_features(features)
    features = features.copy()
    add_attention_features(features, shelf_zones)
    return features, missing_optional

def find_latest_pose_csv():
    """Resolve the newest pose CSV without overwriting earlier runs."""
    if LATEST_POSE_MANIFEST.exists():
        with open(LATEST_POSE_MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest_csv = Path(manifest.get("frame_csv", ""))
        if manifest_csv.is_file():
            return manifest_csv

    versioned_csvs = list(
        POSE_OUTPUT_DIR.glob("zone_pose_wrists_*.csv")
    )
    if versioned_csvs:
        return max(versioned_csvs, key=lambda path: path.stat().st_mtime)

    if LEGACY_INPUT_CSV.is_file():
        return LEGACY_INPUT_CSV

    raise FileNotFoundError(
        "No zone-pose frame CSV was found. Run zone_pose.py first or pass "
        "--input-csv explicitly."
    )

def pose_run_id_from_path(input_csv):
    prefix = "zone_pose_wrists_"
    stem = input_csv.stem

    if stem.startswith(prefix):
        return stem[len(prefix):]

    return "legacy"

def make_versioned_output_path(input_csv):
    pose_run_id = pose_run_id_from_path(input_csv)
    feature_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"behaviour_features_{pose_run_id}_{feature_run_id}.csv"
    return POSE_OUTPUT_DIR / filename, feature_run_id

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract posture, motion and shelf-interaction features."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Frame-level CSV created by zone_pose.py. If omitted, the newest "
            "pose run is selected automatically."
        )
    )
    parser.add_argument(
        "--zones-json",
        type=Path,
        default=DEFAULT_ZONES_JSON,
        help="LabelMe shelf/product zone file"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Destination feature CSV. If omitted, a new timestamped filename "
            "is created."
        )
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_csv = args.input_csv or find_latest_pose_csv()
    generated_output_csv, feature_run_id = make_versioned_output_path(input_csv)
    output_csv = args.output_csv or generated_output_csv
    shelf_zones, product_zones = load_zone_groups(args.zones_json)
    frame_df = pd.read_csv(input_csv)
    features, missing_optional = extract_behaviour_features(
        frame_df,
        shelf_zones,
        product_zones
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False, float_format="%.4f")

    feature_manifest = {
        "feature_run_id": feature_run_id,
        "pose_run_id": pose_run_id_from_path(input_csv),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(input_csv.resolve()),
        "zones_json": str(args.zones_json.resolve()),
        "output_csv": str(output_csv.resolve()),
        "input_rows": len(frame_df),
        "output_columns": len(features.columns),
        "zone_assignment": {
            "operational_method": "single_point_with_corridor_gap_fill",
            "event_start_method": "single_projected_hand_point",
            "corridor_fill_requires_existing_contact": True,
            "corridor_fill_minimum_primary_rows": (
                HYBRID_CORRIDOR_FILL_MIN_PRIMARY_ROWS
            ),
            "corridor_fill_requires_same_shelf": True,
            "single_point_columns_retained": True,
            "corridor_candidates_retained": True
        },
        "thresholds": {
            "arm_extended_ratio": ARM_EXTENDED_RATIO_THR,
            "stationary_speed_px_sec": STATIONARY_SPEED_PX_SEC,
            "direction_speed_px_sec": DIRECTION_SPEED_PX_SEC,
            "near_product_distance_px": NEAR_PRODUCT_DISTANCE_PX,
            "hybrid_corridor_fill_max_sec": HYBRID_CORRIDOR_FILL_MAX_SEC,
            "hybrid_corridor_fill_min_primary_rows": (
                HYBRID_CORRIDOR_FILL_MIN_PRIMARY_ROWS
            ),
            "body_near_shelf_norm": BODY_NEAR_SHELF_NORM_THR,
            "crouching_knee_angle_deg": CROUCHING_KNEE_ANGLE_DEG,
            "bending_hip_angle_deg": BENDING_HIP_ANGLE_DEG,
            "attention_target_angle_deg": ATTENTION_TARGET_ANGLE_DEG,
            "torso_fallback_angle_deg": TORSO_FALLBACK_ANGLE_DEG,
            "attention_target_margin_deg": ATTENTION_TARGET_MARGIN_DEG,
            "attention_max_shelf_norm": ATTENTION_MAX_SHELF_NORM_THR,
            "attention_hands_near_body_norm": ATTENTION_HANDS_NEAR_BODY_NORM,
            "attention_smoothing_rows": ATTENTION_SMOOTHING_ROWS,
            "presentation_max_inter_wrist_norm": (
                PRESENTATION_MAX_INTER_WRIST_NORM
            ),
            "multi_product_max_inter_wrist_norm": (
                MULTI_PRODUCT_MAX_INTER_WRIST_NORM
            ),
            "presentation_max_head_distance_norm": (
                PRESENTATION_MAX_HEAD_DISTANCE_NORM
            ),
            "presentation_min_arm_extension": (
                PRESENTATION_MIN_ARM_EXTENSION
            ),
            "presentation_max_flexed_elbow_deg": (
                PRESENTATION_MAX_FLEXED_ELBOW_DEG
            )
        }
    }
    manifest_path = output_csv.with_suffix(".json")
    latest_manifest_path = POSE_OUTPUT_DIR / "latest_behaviour_features_run.json"

    for target in (manifest_path, latest_manifest_path):
        with open(target, "w", encoding="utf-8") as f:
            json.dump(feature_manifest, f, indent=2)

    print("Behaviour feature extraction complete.")
    print("Pose input CSV:", input_csv)
    print("Feature run ID:", feature_run_id)
    print("Input rows:", len(frame_df))
    print("Output columns:", len(features.columns))
    print("Saved feature CSV:", output_csv)
    print("Saved feature manifest:", manifest_path)
    print(
        "Rows with known posture:",
        int((features["posture_state"] != "unknown").sum())
    )
    print(
        "Rows with body-to-shelf distance:",
        int(features["body_to_shelf_distance_px"].notna().sum())
    )

    missing_face_columns = sorted(
        set(missing_optional).intersection({
            "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
            "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y"
        })
    )
    if missing_face_columns:
        print(
            "This pose CSV predates eye/ear export. Attention used the "
            "shoulder-to-nose/torso fallbacks; rerun the updated zone_pose.py "
            "for the full head-direction estimate."
        )

if __name__ == "__main__":
    main()
