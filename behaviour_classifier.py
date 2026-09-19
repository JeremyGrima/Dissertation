import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "pose_tests"
LATEST_FEATURE_MANIFEST = OUTPUT_DIR / "latest_behaviour_features_run.json"
LATEST_CONFIRMATION_MANIFEST = OUTPUT_DIR / "latest_product_confirmation_run.json"
LATEST_CLASSIFIER_MANIFEST = OUTPUT_DIR / "latest_behaviour_classifier_run.json"
LATEST_POSE_MANIFEST = OUTPUT_DIR / "latest_zone_pose_run.json"

def same_path(left, right):
    if not left or not right:
        return False
    return Path(left).resolve() == Path(right).resolve()

BEHAVIOURS = [#Rule-based temporal classifier for the five MERL shopping behaviours.
    "reach_to_shelf",
    "hand_in_shelf",
    "retract_from_shelf",
    "inspect_product",
    "inspect_shelf"
]
PRIMARY_PRIORITY = [
    "hand_in_shelf",
    "reach_to_shelf",
    "retract_from_shelf",
    "inspect_product",
    "inspect_shelf"
]

BEHAVIOUR_COLOURS = {
    "reach_to_shelf": (0, 165, 255),
    "hand_in_shelf": (0, 0, 255),
    "retract_from_shelf": (255, 120, 0),
    "inspect_product": (255, 0, 255),
    "inspect_shelf": (0, 200, 0),
    "background": (190, 190, 190),
    "uncertain": (0, 215, 255),
    "shelf": (0, 220, 0),
    "hands": (255, 0, 255),
    "held_product": (255, 80, 255),
    "elsewhere": (190, 190, 190)
}

MOTION_SMOOTHING_ROWS = 5  # Smooths short motion spikes.
HAND_MIN_DURATION_SEC = 0.20
HAND_MAX_MISSING_ROWS = 1
MOTION_MIN_DURATION_SEC = 0.13
MOTION_MAX_MISSING_ROWS = 1
MOTION_LINK_GAP_SEC = 0.40
REACH_LOOKBACK_SEC = 1.20
RETRACT_LOOKAHEAD_SEC = 1.20
APPROACH_SPEED_THR = 50.0
RETURN_SPEED_THR = 50.0
MOVING_SPEED_THR = 35.0
MAX_REACH_SHELF_DISTANCE_PX = 200.0
MAX_REACH_SHELF_DISTANCE_NORM = 1.10
CROUCHED_RETRACT_PRESENTATION_MIN_SEC = 0.40
CROUCHED_RETRACT_MIN_POSTURE_SHARE = 0.50

INSPECT_PRODUCT_SEARCH_SEC = 7.00  # Pose-only proxy; no detector confirms the object here.
INSPECT_PRODUCT_MIN_DURATION_SEC = 0.50
INSPECT_PRODUCT_MAX_MISSING_ROWS = 1
INSPECT_PRODUCT_MAX_BODY_DISTANCE_NORM = 0.65
INSPECT_PRODUCT_STATIONARY_SPEED = 45.0
INSPECT_PRODUCT_MIN_HAND_ATTENTION_SHARE = 0.40
INSPECT_PRODUCT_HEAD_TO_HANDS_ANGLE_DEG = 55.0
INSPECT_PRODUCT_MIN_PRESENTATION_SHARE = 0.40
INSPECT_PRODUCT_PRESENTATION_ONLY_MIN_DURATION_SEC = 0.80

INSPECT_SHELF_MIN_DURATION_SEC = 0.80
INSPECT_SHELF_MAX_MISSING_ROWS = 2
INSPECT_SHELF_EVIDENCE_WARMUP_SEC = 0.40
INSPECT_SHELF_MAX_BODY_DISTANCE_NORM = 1.80
INSPECT_SHELF_MIN_ATTENTION_SCORE = 0.18

CORRIDOR_ORIGIN_PERSISTENCE_WEIGHT = 0.15  # Rewards zones that persist near shelf exit.
CORRIDOR_ORIGIN_AMBIGUITY_RATIO = 0.82
CORRIDOR_ORIGIN_MIN_TOP_SHARE = 0.28
CORRIDOR_ORIGIN_PRE_EXIT_SEC = 0.80
SINGLE_POINT_ORIGIN_SUPPORT_WEIGHT = 0.50

REQUIRED_COLUMNS = {
    "frame",
    "time_sec",
    "left_wrist_x",
    "left_wrist_y",
    "right_wrist_x",
    "right_wrist_y",
    "left_wrist_observed",
    "right_wrist_observed",
    "left_wrist_held",
    "right_wrist_held",
    "left_in_shelf",
    "right_in_shelf",
    "left_in_product",
    "right_in_product",
    "left_wrist_speed_px_sec",
    "right_wrist_speed_px_sec",
    "left_wrist_toward_shelf_speed_px_sec",
    "right_wrist_toward_shelf_speed_px_sec",
    "left_wrist_toward_body_speed_px_sec",
    "right_wrist_toward_body_speed_px_sec",
    "left_wrist_to_shelf_distance_px",
    "right_wrist_to_shelf_distance_px",
    "left_wrist_to_shelf_norm",
    "right_wrist_to_shelf_norm",
    "left_wrist_to_body_norm",
    "right_wrist_to_body_norm",
    "left_arm_extension_ratio",
    "right_arm_extension_ratio",
    "left_shelf_zone",
    "right_shelf_zone",
    "left_product_zone",
    "right_product_zone",
    "body_near_shelf",
    "body_to_shelf_norm",
    "attention_to_shelf_proxy",
    "attention_to_hands_proxy",
    "attention_target",
    "attention_target_score",
    "attention_target_dwell_sec",
    "attention_direction_method",
    "attention_to_hands_angle_deg",
    "hands_near_body",
    "wrist_midpoint_to_body_norm",
    "held_product_available",
    "held_product_origin_zone",
    "inter_wrist_distance_norm",
    "wrist_midpoint_to_head_norm",
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
    "product_presentation_pose",
    "product_presentation_score",
    "product_presentation_dwell_sec",
    "body_to_shelf_nearest_zone",
    "posture_state"
}

def as_boolean(series):
    """Convert CSV booleans safely without treating the string 'False' as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    mapped = series.map({
        True: True,
        False: False,
        1: True,
        0: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False
    })
    return mapped.astype("boolean")

def rolling_median(series, window=MOTION_SMOOTHING_ROWS):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rolling(
        window=window,
        center=True,
        min_periods=max(2, window // 2)
    ).median()

def bridge_short_false_gaps(condition, max_gap_rows):
    """Fill bounded False gaps while leaving leading/trailing gaps unchanged."""
    values = as_boolean(pd.Series(condition)).fillna(False).to_numpy(
        dtype=bool,
        copy=True
    )

    if max_gap_rows <= 0 or len(values) == 0:
        return pd.Series(values, index=getattr(condition, "index", None))

    index = 0
    while index < len(values):
        if values[index]:
            index += 1
            continue

        gap_start = index
        while index < len(values) and not values[index]:
            index += 1
        gap_end = index - 1
        gap_length = gap_end - gap_start + 1
        bounded = gap_start > 0 and index < len(values)

        if bounded and gap_length <= max_gap_rows:
            values[gap_start:index] = True

    return pd.Series(values, index=getattr(condition, "index", None))

def true_runs(condition, features, fps, min_duration_sec, max_gap_rows=0):
    bridged = bridge_short_false_gaps(condition, max_gap_rows)
    values = bridged.to_numpy(dtype=bool)
    runs = []
    index = 0

    while index < len(values):
        if not values[index]:
            index += 1
            continue

        start_index = index
        while index + 1 < len(values) and values[index + 1]:
            index += 1
        end_index = index
        start_frame = int(features.at[start_index, "frame"])
        end_frame = int(features.at[end_index, "frame"])
        duration_sec = (end_frame - start_frame) / fps

        if duration_sec + 1e-9 >= min_duration_sec:
            runs.append({
                "start_index": start_index,
                "end_index": end_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec": duration_sec,
                "observations": end_index - start_index + 1
            })

        index += 1

    return runs

def unique_join(series):
    values = [str(value) for value in series.dropna().unique()]
    return "|".join(sorted(values)) if values else None

def mode_or_none(series):
    values = series.dropna()
    if values.empty:
        return None
    return str(values.mode().iloc[0])

def legacy_contact_origin_zone(subset, wrist):
    """Choose the product zone at the reach apex, not during withdrawal."""
    single_point_column = f"{wrist}_single_point_product_zone"
    zone_column = (
        single_point_column
        if single_point_column in subset.columns
        else f"{wrist}_product_zone"
    )
    valid_zones = subset[zone_column].dropna()
    if valid_zones.empty:
        return None

    signals = []  # Peak wrist extension best represents the handled shelf region.
    for column in (
        f"{wrist}_wrist_to_body_norm",
        f"{wrist}_arm_extension_smoothed",
        f"{wrist}_arm_extension_ratio"
    ):
        if column in subset.columns:
            signal = pd.to_numeric(subset[column], errors="coerce").where(
                subset[zone_column].notna()
            )
            if signal.notna().any():
                signals.append(signal)

    if not signals:
        return mode_or_none(valid_zones)

    peak_index = signals[0].idxmax()
    peak_position = subset.index.get_loc(peak_index)
    contact_window = subset.iloc[
        max(0, peak_position - 1):min(len(subset), peak_position + 2)
    ][zone_column].dropna()
    if contact_window.empty:
        return str(subset.loc[peak_index, zone_column])

    counts = contact_window.value_counts()
    most_common = counts[counts == counts.max()].index.astype(str).tolist()
    peak_zone = str(subset.loc[peak_index, zone_column])
    return peak_zone if peak_zone in most_common else most_common[0]

def parse_corridor_candidates(value):
    if value is None or pd.isna(value):
        return {}
    candidates = {}
    for item in str(value).split("|"):
        if ":" not in item:
            continue
        label, score = item.rsplit(":", 1)
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if label and np.isfinite(numeric_score) and numeric_score > 0:
            candidates[label] = numeric_score
    return candidates

def product_zones_are_neighbours(first, second):
    first_match = re.match(
        r"^([LR])(\d+)_Product_(\d+)of(\d+)$", str(first or "")
    )
    second_match = re.match(
        r"^([LR])(\d+)_Product_(\d+)of(\d+)$", str(second or "")
    )
    if not first_match or not second_match:
        return False
    if first_match.group(1) != second_match.group(1):
        return False
    return (
        abs(int(first_match.group(2)) - int(second_match.group(2))) <= 1
        and abs(int(first_match.group(3)) - int(second_match.group(3))) <= 1
    )

def contact_origin_decision(subset, wrist):
    """Vote near shelf exit using all corridor candidates and point support."""
    candidate_column = f"{wrist}_corridor_product_candidates"
    if (
        candidate_column not in subset.columns
        or subset[candidate_column].dropna().empty
    ):
        fallback = legacy_contact_origin_zone(subset, wrist)
        return {
            "zone": fallback,
            "candidate": fallback,
            "alternative": None,
            "ambiguous": False,
            "confidence": None,
            "margin": None,
            "method": "legacy_reach_apex",
            "evidence": fallback,
        }

    voting_subset = subset
    if "time_sec" in subset.columns:
        times = pd.to_numeric(subset["time_sec"], errors="coerce")
        if times.notna().any():
            contact_end_time = float(times.max())
            recent = subset[
                times >= contact_end_time - CORRIDOR_ORIGIN_PRE_EXIT_SEC
            ]
            if not recent.empty:
                voting_subset = recent

    stable = voting_subset
    observed_column = f"{wrist}_wrist_observed"
    held_column = f"{wrist}_wrist_held"
    if observed_column in voting_subset.columns:
        trusted = as_boolean(
            voting_subset[observed_column]
        ).fillna(False)
        if held_column in voting_subset.columns:
            trusted &= ~as_boolean(
                voting_subset[held_column]
            ).fillna(False)
        trusted_subset = voting_subset[trusted]
        if len(trusted_subset) >= 2:
            stable = trusted_subset

    accumulated = defaultdict(float)
    hits = Counter()
    used_frames = 0
    ambiguous_column = f"{wrist}_corridor_product_ambiguous"
    single_point_column = f"{wrist}_single_point_product_zone"
    for _, row in stable.iterrows():
        candidates = parse_corridor_candidates(row.get(candidate_column))
        if not candidates:
            continue
        single_point_zone = row.get(single_point_column)
        if single_point_zone is not None and not pd.isna(single_point_zone):
            single_point_zone = str(single_point_zone).strip()
            if single_point_zone:
                candidates[single_point_zone] = (
                    candidates.get(single_point_zone, 0.0)
                    + SINGLE_POINT_ORIGIN_SUPPORT_WEIGHT
                )
        used_frames += 1
        frame_weight = 0.80 if bool(
            as_boolean(pd.Series([row.get(ambiguous_column, False)]))
            .fillna(False)
            .iloc[0]
        ) else 1.0
        for label, score in candidates.items():
            accumulated[label] += frame_weight * score
            hits[label] += 1

    if not accumulated:
        fallback = legacy_contact_origin_zone(voting_subset, wrist)
        return {
            "zone": fallback,
            "candidate": fallback,
            "alternative": None,
            "ambiguous": False,
            "confidence": None,
            "margin": None,
            "method": "legacy_reach_apex_no_corridor_evidence",
            "evidence": fallback,
        }

    combined = {
        label: score + CORRIDOR_ORIGIN_PERSISTENCE_WEIGHT * hits[label]
        for label, score in accumulated.items()
    }
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    alternative_label = ranked[1][0] if len(ranked) > 1 else None
    alternative_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total_score = sum(combined.values())
    top_share = best_score / total_score if total_score > 0 else 0.0
    runner_ratio = (
        alternative_score / best_score if best_score > 0 else 0.0
    )
    margin = 1.0 - runner_ratio
    neighbouring_tie = bool(
        alternative_label
        and product_zones_are_neighbours(best_label, alternative_label)
        and runner_ratio >= CORRIDOR_ORIGIN_AMBIGUITY_RATIO
    )
    ambiguous = neighbouring_tie or top_share < CORRIDOR_ORIGIN_MIN_TOP_SHARE
    evidence = "|".join(
        f"{label}:{score:.3f}"
        for label, score in ranked[:5]
    )
    return {
        "zone": best_label,
        "candidate": best_label,
        "alternative": alternative_label,
        "ambiguous": bool(ambiguous),
        "confidence": round(float(top_share), 4),
        "margin": round(float(margin), 4),
        "method": "hybrid_pre_exit_corridor_vote",
        "evidence": (
            f"pre_exit_sec={CORRIDOR_ORIGIN_PRE_EXIT_SEC:.2f}; "
            f"frames={used_frames}; ranked={evidence}"
        ),
    }

def infer_fps_and_stride(features):
    frame_delta = features["frame"].diff()
    time_delta = features["time_sec"].diff()
    valid = (frame_delta > 0) & (time_delta > 0)

    if not valid.any():
        return 30.0, 1

    fps = float((frame_delta[valid] / time_delta[valid]).median())
    rounded_fps = round(fps)
    if abs(fps - rounded_fps) < 0.10:
        fps = float(rounded_fps)

    stride = int(round(float(frame_delta[frame_delta > 0].median())))
    return fps, max(1, stride)

def add_smoothed_evidence(features):
    """Add robust motion evidence while retaining the original measurements."""
    result = features.copy()

    for wrist in ("left", "right"):
        observed = as_boolean(result[f"{wrist}_wrist_observed"]).fillna(False)
        held = as_boolean(result[f"{wrist}_wrist_held"]).fillna(False)
        trusted = observed & ~held
        result[f"{wrist}_motion_evidence_valid"] = trusted

        for source, destination in (
            ("wrist_speed_px_sec", "speed_smoothed_px_sec"),
            ("wrist_toward_shelf_speed_px_sec", "toward_shelf_smoothed_px_sec"),
            ("wrist_toward_body_speed_px_sec", "toward_body_smoothed_px_sec"),
            ("wrist_to_shelf_distance_px", "shelf_distance_smoothed_px"),
            ("wrist_to_shelf_norm", "shelf_distance_smoothed_norm"),
            ("wrist_to_body_norm", "body_distance_smoothed_norm"),
            ("arm_extension_ratio", "arm_extension_smoothed")
        ):
            result[f"{wrist}_{destination}"] = rolling_median(
                result[f"{wrist}_{source}"].where(trusted)
            )

    return result

def make_event(
    behaviour,
    run,
    fps,
    wrist=None,
    shelf_zone=None,
    product_zones=None,
    rule_score=0.0,
    evidence=None,
    source_hand_episode=None,
    product_origin_zone=None,
    product_origin_zone_candidate=None,
    product_origin_zone_alternative=None,
    product_origin_ambiguous=False,
    product_origin_confidence=None,
    product_origin_margin=None,
    product_origin_method=None,
    product_origin_evidence=None,
    held_product_origin_candidate=None,
    held_product_detection_confirmed=False,
    presentation_pose_share=None,
    attention_to_hands_share=None
):
    return {
        "behaviour": behaviour,
        "start_frame": int(run["start_frame"]),
        "end_frame": int(run["end_frame"]),
        "start_time_sec": run["start_frame"] / fps,
        "end_time_sec": run["end_frame"] / fps,
        "duration_sec": (run["end_frame"] - run["start_frame"]) / fps,
        "wrist": wrist,
        "shelf_zone": shelf_zone,
        "product_zones": product_zones,
        "rule_score": round(float(np.clip(rule_score, 0.0, 1.0)), 3),
        "evidence": evidence,
        "source_hand_episode": source_hand_episode,
        "product_origin_zone": product_origin_zone,
        "product_origin_zone_candidate": product_origin_zone_candidate,
        "product_origin_zone_alternative": product_origin_zone_alternative,
        "product_origin_ambiguous": bool(product_origin_ambiguous),
        "product_origin_confidence": product_origin_confidence,
        "product_origin_margin": product_origin_margin,
        "product_origin_method": product_origin_method,
        "product_origin_evidence": product_origin_evidence,
        "held_product_origin_candidate": held_product_origin_candidate,
        "held_product_detection_confirmed": bool(
            held_product_detection_confirmed
        ),
        "presentation_pose_share": presentation_pose_share,
        "attention_to_hands_share": attention_to_hands_share,
        "_start_index": int(run["start_index"]),
        "_end_index": int(run["end_index"])
    }

def inherited_origin_fields(event):
    return {
        "product_origin_zone": event.get("product_origin_zone"),
        "product_origin_zone_candidate": event.get(
            "product_origin_zone_candidate"
        ),
        "product_origin_zone_alternative": event.get(
            "product_origin_zone_alternative"
        ),
        "product_origin_ambiguous": event.get(
            "product_origin_ambiguous", False
        ),
        "product_origin_confidence": event.get(
            "product_origin_confidence"
        ),
        "product_origin_margin": event.get("product_origin_margin"),
        "product_origin_method": event.get("product_origin_method"),
        "product_origin_evidence": event.get("product_origin_evidence"),
    }

def detect_hand_episodes(features, fps):
    events = []
    episode_id = 0

    for wrist in ("left", "right"):
        in_shelf = as_boolean(features[f"{wrist}_in_shelf"]).fillna(False)
        in_product = as_boolean(features[f"{wrist}_in_product"]).fillna(False)
        has_wrist = (
            features[f"{wrist}_wrist_x"].notna() &
            features[f"{wrist}_wrist_y"].notna()
        )
        condition = in_shelf & in_product & has_wrist
        runs = true_runs(
            condition,
            features,
            fps,
            HAND_MIN_DURATION_SEC,
            HAND_MAX_MISSING_ROWS
        )

        for run in runs:
            episode_id += 1
            subset = features.iloc[run["start_index"]:run["end_index"] + 1]
            origin = contact_origin_decision(subset, wrist)
            observed_share = as_boolean(
                subset[f"{wrist}_wrist_observed"]
            ).fillna(False).mean()
            duration_support = min(run["duration_sec"] / 1.0, 1.0)
            score = 0.78 + 0.12 * observed_share + 0.08 * duration_support
            event = make_event(
                "hand_in_shelf",
                run,
                fps,
                wrist=wrist,
                shelf_zone=mode_or_none(subset[f"{wrist}_shelf_zone"]),
                product_zones=unique_join(subset[f"{wrist}_product_zone"]),
                rule_score=score,
                evidence=(
                    "continuous wrist occupancy in both shelf and product zones; "
                    "adjacent product-zone changes merged; product origin "
                    "selected by stable corridor evidence"
                ),
                source_hand_episode=episode_id,
                product_origin_zone=origin["zone"],
                product_origin_zone_candidate=origin["candidate"],
                product_origin_zone_alternative=origin["alternative"],
                product_origin_ambiguous=origin["ambiguous"],
                product_origin_confidence=origin["confidence"],
                product_origin_margin=origin["margin"],
                product_origin_method=origin["method"],
                product_origin_evidence=origin["evidence"],
            )
            events.append(event)

    return sorted(events, key=lambda event: (event["start_frame"], event["wrist"]))

def motion_candidate_runs(features, condition, fps):
    return true_runs(
        condition,
        features,
        fps,
        MOTION_MIN_DURATION_SEC,
        MOTION_MAX_MISSING_ROWS
    )

def detect_reach_for_hand(features, hand_event, fps, previous_hand_end=None):
    wrist = hand_event["wrist"]
    start_index = hand_event["_start_index"]
    lookback_frames = int(round(REACH_LOOKBACK_SEC * fps))
    earliest_frame = hand_event["start_frame"] - lookback_frames

    if previous_hand_end is not None:
        earliest_frame = max(earliest_frame, previous_hand_end + 1)

    in_window = (
        (features["frame"] >= earliest_frame) &
        (features["frame"] < hand_event["start_frame"])
    )
    valid_motion = as_boolean(
        features[f"{wrist}_motion_evidence_valid"]
    ).fillna(False)
    near_shelf = (
        (features[f"{wrist}_shelf_distance_smoothed_px"] <=
         MAX_REACH_SHELF_DISTANCE_PX) |
        (features[f"{wrist}_shelf_distance_smoothed_norm"] <=
         MAX_REACH_SHELF_DISTANCE_NORM)
    )
    condition = (
        in_window &
        valid_motion &
        near_shelf.fillna(False) &
        (features[f"{wrist}_speed_smoothed_px_sec"] >= MOVING_SPEED_THR) &
        (features[f"{wrist}_toward_shelf_smoothed_px_sec"] >=
         APPROACH_SPEED_THR)
    )
    candidates = motion_candidate_runs(features, condition, fps)

    if not candidates:
        return None

    candidate = candidates[-1]
    link_gap = (hand_event["start_frame"] - candidate["end_frame"]) / fps
    if link_gap > MOTION_LINK_GAP_SEC:
        return None

    run = dict(candidate)
    run["end_index"] = start_index
    run["end_frame"] = hand_event["start_frame"]
    subset = features.iloc[run["start_index"]:run["end_index"] + 1]
    approach_peak = subset[f"{wrist}_toward_shelf_smoothed_px_sec"].quantile(0.90)
    observed_share = as_boolean(
        subset[f"{wrist}_motion_evidence_valid"]
    ).fillna(False).mean()
    extension = subset[f"{wrist}_arm_extension_smoothed"].dropna()
    extension_support = 0.0
    if len(extension) >= 2:
        extension_support = float(
            extension.iloc[-1] - extension.iloc[0] >= 0.02 or
            extension.iloc[-1] >= 0.82
        )
    score = (
        0.50 +
        0.25 * min(max(float(approach_peak), 0.0) / 200.0, 1.0) +
        0.15 * observed_share +
        0.10 * extension_support
    )
    return make_event(
        "reach_to_shelf",
        run,
        fps,
        wrist=wrist,
        shelf_zone=hand_event["shelf_zone"],
        product_zones=hand_event["product_zones"],
        rule_score=score,
        evidence=(
            f"smoothed shelf-approach motion before hand episode; "
            f"90th-percentile approach speed={approach_peak:.1f}px/s; "
            "arm extension used only as supporting evidence"
        ),
        source_hand_episode=hand_event["source_hand_episode"],
        **inherited_origin_fields(hand_event),
    )

def detect_retract_for_hand(features, hand_event, fps, next_hand_start=None):
    wrist = hand_event["wrist"]
    end_index = hand_event["_end_index"]
    latest_frame = hand_event["end_frame"] + int(round(RETRACT_LOOKAHEAD_SEC * fps))

    if next_hand_start is not None:
        latest_frame = min(latest_frame, next_hand_start - 1)

    in_window = (
        (features["frame"] > hand_event["end_frame"]) &
        (features["frame"] <= latest_frame)
    )
    valid_motion = as_boolean(
        features[f"{wrist}_motion_evidence_valid"]
    ).fillna(False)
    condition = (
        in_window &
        valid_motion &
        (features[f"{wrist}_speed_smoothed_px_sec"] >= MOVING_SPEED_THR) &
        (features[f"{wrist}_toward_body_smoothed_px_sec"] >= RETURN_SPEED_THR)
    )
    candidates = motion_candidate_runs(features, condition, fps)

    if candidates:
        candidate = candidates[0]
        link_gap = (
            candidate["start_frame"] - hand_event["end_frame"]
        ) / fps
        if link_gap <= MOTION_LINK_GAP_SEC:
            run = dict(candidate)
            run["start_index"] = end_index
            run["start_frame"] = hand_event["end_frame"]
            subset = features.iloc[
                run["start_index"]:run["end_index"] + 1
            ]
            return_peak = subset[
                f"{wrist}_toward_body_smoothed_px_sec"
            ].quantile(0.90)
            observed_share = as_boolean(
                subset[f"{wrist}_motion_evidence_valid"]
            ).fillna(False).mean()
            score = (
                0.55 +
                0.25 * min(max(float(return_peak), 0.0) / 200.0, 1.0) +
                0.20 * observed_share
            )
            return make_event(
                "retract_from_shelf",
                run,
                fps,
                wrist=wrist,
                shelf_zone=hand_event["shelf_zone"],
                product_zones=hand_event["product_zones"],
                rule_score=score,
                evidence=(
                    f"smoothed movement toward body after shelf exit; "
                    f"90th-percentile return speed={return_peak:.1f}px/s"
                ),
                source_hand_episode=hand_event["source_hand_episode"],
                **inherited_origin_fields(hand_event),
                held_product_origin_candidate=hand_event["product_origin_zone"]
            )

    outside_shelf = ~as_boolean(  # A stable two-hand pose can recover a very short crouched retraction.
        features[f"{wrist}_in_shelf"]
    ).fillna(False)
    outside_product = ~as_boolean(
        features[f"{wrist}_in_product"]
    ).fillna(False)
    presentation = as_boolean(
        features["product_presentation_pose"]
    ).fillna(False)
    crouched = features["posture_state"].isin(["crouching", "bending"])
    fallback_condition = (
        in_window &
        valid_motion &
        outside_shelf &
        outside_product &
        presentation &
        crouched
    )
    fallback_runs = true_runs(
        fallback_condition,
        features,
        fps,
        CROUCHED_RETRACT_PRESENTATION_MIN_SEC,
        HAND_MAX_MISSING_ROWS
    )
    if not fallback_runs:
        return None

    candidate = fallback_runs[0]
    link_gap = (candidate["start_frame"] - hand_event["end_frame"]) / fps
    if link_gap > RETRACT_LOOKAHEAD_SEC:
        return None
    support_subset = features.iloc[
        end_index:candidate["end_index"] + 1
    ]
    posture_share = float(support_subset["posture_state"].isin(
        ["crouching", "bending"]
    ).mean())
    if posture_share < CROUCHED_RETRACT_MIN_POSTURE_SHARE:
        return None

    run = {
        "start_index": end_index,
        "end_index": candidate["start_index"],
        "start_frame": hand_event["end_frame"],
        "end_frame": candidate["start_frame"],
        "duration_sec": link_gap,
        "observations": candidate["start_index"] - end_index + 1
    }
    observed_share = as_boolean(
        support_subset[f"{wrist}_motion_evidence_valid"]
    ).fillna(False).mean()
    score = 0.58 + 0.12 * posture_share + 0.12 * observed_share
    return make_event(
        "retract_from_shelf",
        run,
        fps,
        wrist=wrist,
        shelf_zone=hand_event["shelf_zone"],
        product_zones=hand_event["product_zones"],
        rule_score=score,
        evidence=(
            "short crouched shelf exit followed by persistent two-hand "
            "product-presentation evidence; motion was below the normal "
            "minimum duration"
        ),
        source_hand_episode=hand_event["source_hand_episode"],
        **inherited_origin_fields(hand_event),
        held_product_origin_candidate=hand_event["product_origin_zone"]
    )

def detect_hand_sequences(features, hand_events, fps):
    events = list(hand_events)
    sequences = []

    for wrist in ("left", "right"):
        wrist_hands = [event for event in hand_events if event["wrist"] == wrist]

        for index, hand_event in enumerate(wrist_hands):
            previous_end = wrist_hands[index - 1]["end_frame"] if index > 0 else None
            next_start = (
                wrist_hands[index + 1]["start_frame"]
                if index + 1 < len(wrist_hands)
                else None
            )
            reach = detect_reach_for_hand(
                features,
                hand_event,
                fps,
                previous_hand_end=previous_end
            )
            retract = detect_retract_for_hand(
                features,
                hand_event,
                fps,
                next_hand_start=next_start
            )

            if reach is not None:
                events.append(reach)
            if retract is not None:
                events.append(retract)

            sequences.append({
                "hand": hand_event,
                "reach": reach,
                "retract": retract,
                "next_hand_start": next_start
            })

    return events, sequences

def detect_inspect_product(features, sequences, fps):
    events = []

    for sequence in sequences:
        retract = sequence["retract"]
        hand = sequence["hand"]
        if retract is None:
            continue

        wrist = hand["wrist"]
        latest_frame = retract["end_frame"] + int(
            round(INSPECT_PRODUCT_SEARCH_SEC * fps)
        )
        if sequence["next_hand_start"] is not None:
            latest_frame = min(latest_frame, sequence["next_hand_start"] - 1)

        valid_motion = as_boolean(
            features[f"{wrist}_motion_evidence_valid"]
        ).fillna(False)
        in_shelf = as_boolean(features[f"{wrist}_in_shelf"]).fillna(False)
        in_product = as_boolean(features[f"{wrist}_in_product"]).fillna(False)
        hands_near_body = as_boolean(features["hands_near_body"]).fillna(False)
        presentation_pose = as_boolean(
            features["product_presentation_pose"]
        ).fillna(False)
        in_window = (
            (features["frame"] > retract["end_frame"]) &
            (features["frame"] <= latest_frame)
        )
        stationary_active_wrist = (
            valid_motion &
            (features[f"{wrist}_speed_smoothed_px_sec"] <=
             INSPECT_PRODUCT_STATIONARY_SPEED) &
            (features[f"{wrist}_body_distance_smoothed_norm"] <=
             INSPECT_PRODUCT_MAX_BODY_DISTANCE_NORM)
        )
        condition = (
            in_window &
            ~in_shelf &
            ~in_product &
            hands_near_body &
            (stationary_active_wrist | presentation_pose)
        )
        candidates = true_runs(
            condition,
            features,
            fps,
            INSPECT_PRODUCT_MIN_DURATION_SEC,
            INSPECT_PRODUCT_MAX_MISSING_ROWS
        )

        if not candidates:
            continue

        for run in candidates:
            subset = features.iloc[run["start_index"]:run["end_index"] + 1]
            median_speed = subset[f"{wrist}_speed_smoothed_px_sec"].median()
            median_body_distance = subset[
                f"{wrist}_body_distance_smoothed_norm"
            ].median()
            attention_target = subset["attention_target"].fillna("uncertain")
            target_hands = attention_target.isin(["hands", "held_product"])
            direct_head_to_hands = (
                (subset["attention_direction_method"] != "torso_fallback") &
                (subset["attention_direction_method"] != "unavailable") &
                (subset["attention_to_hands_angle_deg"] <=
                 INSPECT_PRODUCT_HEAD_TO_HANDS_ANGLE_DEG)
            ).fillna(False)
            attention_support = target_hands | direct_head_to_hands
            attention_support_share = float(attention_support.mean())
            presentation_share = float(as_boolean(
                subset["product_presentation_pose"]
            ).fillna(False).mean())
            if (
                attention_support_share <
                INSPECT_PRODUCT_MIN_HAND_ATTENTION_SHARE and
                presentation_share < INSPECT_PRODUCT_MIN_PRESENTATION_SHARE
            ):
                continue  # Weak gaze without a stable presentation pose is not enough.
            if (
                attention_support_share <
                INSPECT_PRODUCT_MIN_HAND_ATTENTION_SHARE and
                run["duration_sec"] <
                INSPECT_PRODUCT_PRESENTATION_ONLY_MIN_DURATION_SEC
            ):
                continue  # Pose-only evidence must last longer than supported gaze.

            held_product_confirmed = as_boolean(
                subset["held_product_available"]
            ).fillna(False).any()
            duration_support = min(run["duration_sec"] / 1.0, 1.0)
            body_support = 0.0
            if pd.notna(median_body_distance):
                body_support = min(
                    max(
                        (
                            INSPECT_PRODUCT_MAX_BODY_DISTANCE_NORM -
                            median_body_distance
                        ) / INSPECT_PRODUCT_MAX_BODY_DISTANCE_NORM,
                        0.0
                    ),
                    1.0
                )
            stationary_support = 0.0
            if pd.notna(median_speed):
                stationary_support = min(
                    max(
                        (
                            INSPECT_PRODUCT_STATIONARY_SPEED - median_speed
                        ) / INSPECT_PRODUCT_STATIONARY_SPEED,
                        0.0
                    ),
                    1.0
                )
            score = (
                0.30 +
                0.15 * duration_support +
                0.10 * body_support +
                0.10 * stationary_support +
                0.15 * attention_support_share +
                0.15 * presentation_share +
                0.05 * float(held_product_confirmed)
            )
            evidence_parts = [
                "follows retraction",
                "hands remain near the body"
            ]
            if attention_support_share >= INSPECT_PRODUCT_MIN_HAND_ATTENTION_SHARE:
                evidence_parts.append("head direction supports the hands")
            if presentation_share >= INSPECT_PRODUCT_MIN_PRESENTATION_SHARE:
                evidence_parts.append(
                    "persistent two-hand product-presentation pose near the head"
                )
            evidence_parts.append(
                "held-object detection available"
                if held_product_confirmed
                else "pose-only proxy, held object not yet confirmed"
            )
            events.append(make_event(
                "inspect_product",
                run,
                fps,
                wrist=wrist,
                shelf_zone=hand["shelf_zone"],
                product_zones=hand["product_zones"],
                rule_score=score,
                evidence="; ".join(evidence_parts),
                source_hand_episode=hand["source_hand_episode"],
                **inherited_origin_fields(hand),
                held_product_origin_candidate=hand["product_origin_zone"],
                held_product_detection_confirmed=held_product_confirmed,
                presentation_pose_share=round(presentation_share, 3),
                attention_to_hands_share=round(attention_support_share, 3)
            ))
            break

    return events

def event_busy_mask(features, events):
    busy = pd.Series(False, index=features.index)
    for event in events:
        busy |= (
            (features["frame"] >= event["start_frame"]) &
            (features["frame"] <= event["end_frame"])
        )
    return busy

def detect_inspect_shelf(features, existing_events, fps):
    attention_target = features["attention_target"].fillna("uncertain")
    attention_score = pd.to_numeric(
        features["attention_target_score"], errors="coerce"
    ).fillna(0.0)
    attention_dwell = pd.to_numeric(
        features["attention_target_dwell_sec"], errors="coerce"
    ).fillna(0.0)
    broad_shelf_range = (
        features["body_to_shelf_norm"] <=
        INSPECT_SHELF_MAX_BODY_DISTANCE_NORM
    ).fillna(False)
    left_in_shelf = as_boolean(features["left_in_shelf"]).fillna(False)
    right_in_shelf = as_boolean(features["right_in_shelf"]).fillna(False)
    left_in_product = as_boolean(features["left_in_product"]).fillna(False)
    right_in_product = as_boolean(features["right_in_product"]).fillna(False)
    no_hand_interaction = ~(
        left_in_shelf | right_in_shelf | left_in_product | right_in_product
    )
    base_condition = (
        (attention_target == "shelf") &
        (attention_score >= INSPECT_SHELF_MIN_ATTENTION_SCORE) &
        (attention_dwell >= INSPECT_SHELF_EVIDENCE_WARMUP_SEC) &
        broad_shelf_range &
        no_hand_interaction
    )
    stable_attention = bridge_short_false_gaps(
        base_condition,
        INSPECT_SHELF_MAX_MISSING_ROWS
    )
    busy = event_busy_mask(features, existing_events)
    condition = stable_attention & ~busy
    runs = true_runs(
        condition,
        features,
        fps,
        INSPECT_SHELF_MIN_DURATION_SEC,
        max_gap_rows=0
    )
    events = []

    for run in runs:
        subset = features.iloc[run["start_index"]:run["end_index"] + 1]
        duration_support = min(run["duration_sec"] / 2.0, 1.0)
        mean_attention_score = float(
            subset["attention_target_score"].mean()
        )
        score = 0.55 + 0.20 * duration_support + 0.20 * mean_attention_score
        events.append(make_event(
            "inspect_shelf",
            run,
            fps,
            wrist=None,
            shelf_zone=mode_or_none(subset["body_to_shelf_nearest_zone"]),
            product_zones=None,
            rule_score=score,
            evidence=(
                "persistent smoothed shelf-directed head evidence within the "
                "broader attention range, without reaching or hand-in-shelf "
                "interaction"
            )
        ))

    return events

def create_frame_predictions(features, events):
    features = features.copy()
    optional_display_defaults = {
        "multi_product_holding_pose": False,
        "multi_product_holding_score": 0.0,
        "multi_product_holding_dwell_sec": 0.0
    }
    for column, default in optional_display_defaults.items():
        if column not in features.columns:
            features[column] = default

    selected_columns = [
        "frame",
        "time_sec",
        "posture_state",
        "body_near_shelf",
        "body_to_shelf_norm",
        "attention_to_shelf_proxy",
        "attention_to_hands_proxy",
        "attention_target_raw",
        "attention_target",
        "attention_target_score",
        "attention_target_dwell_sec",
        "attention_direction_method",
        "attention_direction_start_x",
        "attention_direction_start_y",
        "attention_direction_end_x",
        "attention_direction_end_y",
        "attention_to_shelf_angle_deg",
        "attention_to_hands_angle_deg",
        "attention_to_held_product_angle_deg",
        "attention_origin_to_shelf_nearest_x",
        "attention_origin_to_shelf_nearest_y",
        "wrist_midpoint_x",
        "wrist_midpoint_y",
        "held_product_x",
        "held_product_y",
        "hands_near_body",
        "wrist_midpoint_to_body_norm",
        "inter_wrist_distance_norm",
        "wrist_midpoint_to_head_norm",
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
        "product_presentation_pose",
        "product_presentation_score",
        "product_presentation_dwell_sec",
        "multi_product_holding_pose",
        "multi_product_holding_score",
        "multi_product_holding_dwell_sec",
        "held_product_available",
        "held_product_origin_zone",
        "left_motion_evidence_valid",
        "right_motion_evidence_valid",
        "left_speed_smoothed_px_sec",
        "right_speed_smoothed_px_sec",
        "left_toward_shelf_smoothed_px_sec",
        "right_toward_shelf_smoothed_px_sec",
        "left_toward_body_smoothed_px_sec",
        "right_toward_body_smoothed_px_sec",
        "left_arm_extension_smoothed",
        "right_arm_extension_smoothed"
    ]
    predictions = features[selected_columns].copy()

    for behaviour in BEHAVIOURS:
        predictions[behaviour] = False
        predictions[f"{behaviour}_rule_score"] = 0.0

    active_wrists = [set() for _ in range(len(predictions))]
    active_origin_candidates = [set() for _ in range(len(predictions))]

    for event in events:
        mask = (
            (predictions["frame"] >= event["start_frame"]) &
            (predictions["frame"] <= event["end_frame"])
        )
        behaviour = event["behaviour"]
        predictions.loc[mask, behaviour] = True
        predictions.loc[mask, f"{behaviour}_rule_score"] = np.maximum(
            predictions.loc[mask, f"{behaviour}_rule_score"],
            event["rule_score"]
        )

        if event["wrist"]:
            for index in predictions.index[mask]:
                active_wrists[index].add(event["wrist"])
        if event.get("held_product_origin_candidate"):
            for index in predictions.index[mask]:
                active_origin_candidates[index].add(
                    event["held_product_origin_candidate"]
                )

    predictions["active_wrists"] = [
        "|".join(sorted(wrists)) if wrists else None
        for wrists in active_wrists
    ]
    predictions["held_product_origin_candidates"] = [
        "|".join(sorted(origins)) if origins else None
        for origins in active_origin_candidates
    ]

    def labels_for_row(row):
        labels = [behaviour for behaviour in BEHAVIOURS if bool(row[behaviour])]
        if labels:
            return "|".join(labels)
        if (
            row["attention_target"] == "elsewhere" and
            float(row["attention_target_score"]) >= 0.35
        ):
            return "background"
        return "uncertain"

    predictions["behaviour_labels"] = predictions.apply(labels_for_row, axis=1)
    primary = []
    primary_scores = []

    for _, row in predictions.iterrows():
        label = (
            "background"
            if (
                row["attention_target"] == "elsewhere" and
                float(row["attention_target_score"]) >= 0.35
            )
            else "uncertain"
        )
        score = 0.0
        for behaviour in PRIMARY_PRIORITY:
            if bool(row[behaviour]):
                label = behaviour
                score = float(row[f"{behaviour}_rule_score"])
                break
        primary.append(label)
        primary_scores.append(score)

    predictions["primary_behaviour"] = primary
    predictions["primary_rule_score"] = primary_scores
    return predictions

def draw_text(frame, text, x, y, colour, scale=0.62, thickness=2):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA
    )

def annotate_behaviour_video(input_video, predictions, output_video):
    """Overlay temporal behaviour labels on the annotated pose video."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed, so the behaviour video cannot be created."
        )

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open annotated pose video: {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create behaviour video: {output_video}")

    written_frames = 0

    for row_index, row in predictions.iterrows():
        ok, frame = capture.read()
        if not ok:
            break

        active_labels = [
            behaviour
            for behaviour in BEHAVIOURS
            if bool(row[behaviour])
        ]
        if not active_labels:
            active_labels = [str(row["primary_behaviour"])]

        origin_text = row.get("held_product_origin_candidates")
        has_origin_candidate = pd.notna(origin_text)
        presentation_active = bool(row.get("product_presentation_pose", False))
        separated_products_pose = (
            bool(row.get("multi_product_holding_pose", False)) and
            not presentation_active
        )
        extra_origin_height = 24 if has_origin_candidate else 0
        extra_presentation_height = 24 if presentation_active else 0
        extra_multi_product_height = 24 if separated_products_pose else 0
        panel_height = (
            96 + extra_origin_height + extra_presentation_height +
            extra_multi_product_height +
            29 * len(active_labels)
        )
        panel_width = min(width - 20, 570)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

        draw_text(
            frame,
            f"Source frame {int(row['frame'])} | {float(row['time_sec']):.2f}s",
            24,
            38,
            (255, 255, 255),
            scale=0.58,
            thickness=1
        )
        wrist_text = row.get("active_wrists")
        if pd.notna(wrist_text):
            draw_text(
                frame,
                f"Active wrist: {str(wrist_text)}",
                330,
                38,
                (225, 225, 225),
                scale=0.52,
                thickness=1
            )

        attention_target = str(row.get("attention_target", "uncertain"))
        attention_score = float(row.get("attention_target_score", 0.0))
        attention_method = str(
            row.get("attention_direction_method", "unavailable")
        )
        draw_text(
            frame,
            (
                f"Attention: {attention_target.replace('_', ' ').upper()} "
                f"({attention_score:.2f}, {attention_method})"
            ),
            24,
            68,
            BEHAVIOUR_COLOURS.get(attention_target, (235, 235, 235)),
            scale=0.52,
            thickness=1
        )

        detail_y = 92
        if has_origin_candidate:
            draw_text(
                frame,
                f"Origin candidate: {str(origin_text)} (unconfirmed)",
                24,
                detail_y,
                (0, 215, 255),
                scale=0.48,
                thickness=1
            )
            detail_y += 24
        if presentation_active:
            presentation_score = float(
                row.get("product_presentation_score", 0.0)
            )
            draw_text(
                frame,
                f"Product-presentation pose: YES ({presentation_score:.2f})",
                24,
                detail_y,
                (255, 0, 255),
                scale=0.48,
                thickness=1
            )
            detail_y += 24
        if separated_products_pose:
            multi_product_score = float(
                row.get("multi_product_holding_score", 0.0)
            )
            draw_text(
                frame,
                (
                    "Separated-hand product pose: YES "
                    f"({multi_product_score:.2f}, candidate)"
                ),
                24,
                detail_y,
                (255, 120, 255),
                scale=0.48,
                thickness=1
            )
            detail_y += 24
        label_y_start = detail_y + 10

        start_x = row.get("attention_direction_start_x")
        start_y = row.get("attention_direction_start_y")
        end_x = row.get("attention_direction_end_x")
        end_y = row.get("attention_direction_end_y")
        if all(pd.notna(value) for value in (start_x, start_y, end_x, end_y)):
            direction = np.asarray(
                [float(end_x) - float(start_x), float(end_y) - float(start_y)]
            )
            length = float(np.linalg.norm(direction))
            if length > 0:
                direction /= length
                arrow_end = (
                    int(round(float(start_x) + 75.0 * direction[0])),
                    int(round(float(start_y) + 75.0 * direction[1]))
                )
                cv2.arrowedLine(
                    frame,
                    (int(round(float(start_x))), int(round(float(start_y)))),
                    arrow_end,
                    BEHAVIOUR_COLOURS.get(
                        attention_target, (235, 235, 235)
                    ),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.22
                )

        target_columns = {
            "shelf": (
                "attention_origin_to_shelf_nearest_x",
                "attention_origin_to_shelf_nearest_y"
            ),
            "hands": ("wrist_midpoint_x", "wrist_midpoint_y"),
            "held_product": ("held_product_x", "held_product_y")
        }
        if attention_target in target_columns:
            target_x_column, target_y_column = target_columns[attention_target]
            target_x = row.get(target_x_column)
            target_y = row.get(target_y_column)
            if pd.notna(target_x) and pd.notna(target_y):
                cv2.circle(
                    frame,
                    (int(round(float(target_x))), int(round(float(target_y)))),
                    8,
                    BEHAVIOUR_COLOURS[attention_target],
                    2,
                    cv2.LINE_AA
                )

        for label_index, behaviour in enumerate(active_labels):
            y = label_y_start + 29 * label_index
            display_label = behaviour.replace("_", " ").upper()
            score = (
                0.0
                if behaviour in ("background", "uncertain")
                else float(row[f"{behaviour}_rule_score"])
            )
            suffix = (
                ""
                if behaviour in ("background", "uncertain")
                else f"  score {score:.2f}"
            )
            draw_text(
                frame,
                display_label + suffix,
                24,
                y,
                BEHAVIOUR_COLOURS[behaviour],
                scale=0.67,
                thickness=2
            )

        primary = str(row["primary_behaviour"])  # Draw a thin colour strip for rapid state changes.
        cv2.rectangle(
            frame,
            (0, height - 8),
            (width, height - 1),
            BEHAVIOUR_COLOURS.get(primary, (190, 190, 190)),
            -1
        )
        writer.write(frame)
        written_frames += 1

    capture.release()
    writer.release()
    return written_frames

def classify_behaviours(feature_df):
    missing = REQUIRED_COLUMNS.difference(feature_df.columns)
    if missing:
        raise ValueError(
            "Feature CSV is missing required columns: " +
            ", ".join(sorted(missing))
        )

    features = feature_df.sort_values("frame").reset_index(drop=True)
    fps, stride = infer_fps_and_stride(features)
    features = add_smoothed_evidence(features)
    hand_events = detect_hand_episodes(features, fps)
    events, sequences = detect_hand_sequences(features, hand_events, fps)
    inspect_product_events = detect_inspect_product(features, sequences, fps)
    events.extend(inspect_product_events)
    inspect_shelf_events = detect_inspect_shelf(features, events, fps)
    events.extend(inspect_shelf_events)
    behaviour_order = {name: index for index, name in enumerate(BEHAVIOURS)}
    events.sort(key=lambda event: (
        event["start_frame"],
        behaviour_order[event["behaviour"]],
        event["wrist"] or ""
    ))

    public_columns = [
        "event_id",
        "behaviour",
        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "wrist",
        "shelf_zone",
        "product_zones",
        "rule_score",
        "evidence",
        "source_hand_episode",
        "product_origin_zone",
        "product_origin_zone_candidate",
        "product_origin_zone_alternative",
        "product_origin_ambiguous",
        "product_origin_confidence",
        "product_origin_margin",
        "product_origin_method",
        "product_origin_evidence",
        "held_product_origin_candidate",
        "held_product_detection_confirmed",
        "presentation_pose_share",
        "attention_to_hands_share"
    ]
    public_events = []
    for event_id, event in enumerate(events, start=1):
        public_event = {key: event.get(key) for key in public_columns}
        public_event["event_id"] = event_id
        public_events.append(public_event)

    event_df = pd.DataFrame(public_events, columns=public_columns)
    predictions = create_frame_predictions(features, events)
    return predictions, event_df, fps, stride

def find_latest_feature_csv():
    feature_manifest = None
    feature_path = None
    if LATEST_FEATURE_MANIFEST.exists():
        with open(LATEST_FEATURE_MANIFEST, "r", encoding="utf-8") as f:
            feature_manifest = json.load(f)
        candidate = Path(feature_manifest.get("output_csv", ""))
        if candidate.is_file():
            feature_path = candidate

    if LATEST_CONFIRMATION_MANIFEST.exists():  # Only reuse confirmations from this feature run.
        with open(
            LATEST_CONFIRMATION_MANIFEST, "r", encoding="utf-8"
        ) as f:
            confirmation_manifest = json.load(f)
        confirmed_path = Path(confirmation_manifest.get("output_csv", ""))
        confirmed_input = Path(
            confirmation_manifest.get("input_feature_csv", "")
        )
        confirmation_matches = (
            feature_path is None or
            (
                confirmed_input.is_file() and
                confirmed_input.resolve() == feature_path.resolve()
            )
        )
        if confirmed_path.is_file() and confirmation_matches:
            combined_manifest = dict(feature_manifest or {})
            combined_manifest.update({
                "pose_run_id": confirmation_manifest.get(
                    "pose_run_id",
                    combined_manifest.get("pose_run_id", "unknown")
                ),
                "output_csv": str(confirmed_path),
                "product_confirmation_run_id": confirmation_manifest.get(
                    "confirmation_run_id"
                )
            })
            return confirmed_path, combined_manifest

    if feature_path is not None:
        return feature_path, feature_manifest

    candidates = list(OUTPUT_DIR.glob("behaviour_features_*.csv"))
    if candidates:
        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
        return path, {}

    raise FileNotFoundError(
        "No behaviour feature CSV was found. Run behaviour_features.py first "
        "or pass --input-csv explicitly."
    )

def manifest_for_feature_csv(input_csv):
    """Load the sidecar manifest when an explicit feature CSV is supplied."""
    input_csv = Path(input_csv).resolve()
    candidates = [input_csv.with_suffix(".json")]
    candidates.extend(OUTPUT_DIR.glob("behaviour_features_*.json"))
    seen = set()
    for manifest_path in candidates:
        manifest_path = Path(manifest_path)
        resolved_manifest = manifest_path.resolve()
        if resolved_manifest in seen or not manifest_path.is_file():
            continue
        seen.add(resolved_manifest)
        try:
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        output_csv = manifest.get("output_csv")
        if output_csv and same_path(output_csv, input_csv):
            return manifest
    return {}

def find_pose_video(pose_run_id):
    manifest_candidates = []
    if pose_run_id and pose_run_id != "unknown":
        manifest_candidates.append(
            OUTPUT_DIR / f"zone_pose_run_{pose_run_id}.json"
        )
    manifest_candidates.append(LATEST_POSE_MANIFEST)

    for manifest_path in manifest_candidates:
        if not manifest_path.is_file():
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if (
            pose_run_id not in (None, "unknown") and
            str(manifest.get("run_id")) != str(pose_run_id)
        ):
            continue

        annotated_video = manifest.get("annotated_video")
        if not annotated_video:
            continue
        video_path = Path(annotated_video)
        if video_path.is_file():
            return video_path

    if pose_run_id and pose_run_id != "unknown":
        candidates = list(
            OUTPUT_DIR.glob(f"zone_pose_wrists_video_{pose_run_id}.mp4")
        )
        if candidates:
            return candidates[0]

    return None

def make_output_paths(input_csv, feature_manifest):
    pose_run_id = feature_manifest.get("pose_run_id", "unknown")
    classifier_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_path = OUTPUT_DIR / (
        f"behaviour_frame_labels_{pose_run_id}_{classifier_run_id}.csv"
    )
    event_path = OUTPUT_DIR / (
        f"behaviour_events_{pose_run_id}_{classifier_run_id}.csv"
    )
    video_path = OUTPUT_DIR / (
        f"behaviour_video_{pose_run_id}_{classifier_run_id}.mp4"
    )
    manifest_path = OUTPUT_DIR / (
        f"behaviour_classifier_run_{pose_run_id}_{classifier_run_id}.json"
    )
    return (
        frame_path,
        event_path,
        video_path,
        manifest_path,
        classifier_run_id,
        pose_run_id
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify temporal MERL shopping behaviours from features."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Feature CSV. If omitted, the newest feature run is selected."
    )
    parser.add_argument("--frame-output-csv", type=Path, default=None)
    parser.add_argument("--event-output-csv", type=Path, default=None)
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="Annotated pose video. The matching pose run is used by default."
    )
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Create CSV outputs without rendering a behaviour video."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.input_csv is None:
        input_csv, feature_manifest = find_latest_feature_csv()
    else:
        input_csv = args.input_csv.resolve()
        feature_manifest = manifest_for_feature_csv(input_csv)

    (
        default_frame_output,
        default_event_output,
        default_video_output,
        manifest_path,
        classifier_run_id,
        pose_run_id
    ) = make_output_paths(input_csv, feature_manifest)
    frame_output = args.frame_output_csv or default_frame_output
    event_output = args.event_output_csv or default_event_output
    video_output = args.video_output or default_video_output
    feature_df = pd.read_csv(input_csv)
    predictions, events, fps, stride = classify_behaviours(feature_df)
    frame_output.parent.mkdir(parents=True, exist_ok=True)
    event_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(frame_output, index=False, float_format="%.4f")
    events.to_csv(event_output, index=False, float_format="%.3f")

    input_video = (  # CSV-only batches intentionally leave this unset.
        None
        if args.skip_video
        else args.input_video or find_pose_video(pose_run_id)
    )
    rendered_video = None
    video_frames_written = 0

    if not args.skip_video:
        if input_video is None:
            print(
                "Warning: no matching annotated pose video was found; "
                "CSV outputs were still created."
            )
        elif cv2 is None:
            print(
                "Warning: OpenCV is unavailable; CSV outputs were created "
                "but behaviour video rendering was skipped."
            )
        else:
            video_frames_written = annotate_behaviour_video(
                input_video,
                predictions,
                video_output
            )
            rendered_video = video_output

    manifest = {
        "classifier_run_id": classifier_run_id,
        "pose_run_id": pose_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_feature_csv": str(input_csv.resolve()),
        "frame_output_csv": str(frame_output.resolve()),
        "event_output_csv": str(event_output.resolve()),
        "input_pose_video": (
            None if input_video is None else str(input_video.resolve())
        ),
        "behaviour_video": (
            None if rendered_video is None else str(rendered_video.resolve())
        ),
        "video_frames_written": video_frames_written,
        "fps": fps,
        "stride": stride,
        "input_rows": len(feature_df),
        "events_generated": len(events),
        "behaviour_counts": events["behaviour"].value_counts().to_dict(),
        "precautions": {
            "motion_smoothing_rows": MOTION_SMOOTHING_ROWS,
            "held_or_missing_wrists_excluded_from_motion": True,
            "product_zone_changes_merged_within_hand_episode": True,
            "legacy_reach_apex_available_as_fallback": True,
            "product_origin_selected_by_hybrid_pre_exit_vote": True,
            "product_origin_vote_uses_pre_exit_window": True,
            "single_point_supports_corridor_origin_vote": True,
            "ambiguous_product_origins_are_explicit": True,
            "crouched_retract_requires_persistent_presentation": True,
            "posture_used_in_core_rules": False,
            "inspect_product_is_pose_proxy": True,
            "product_presentation_pose_is_2d_proxy_not_true_arm_height": True,
            "weak_attention_outputs_uncertain": True,
            "torso_is_attention_fallback_only": True,
            "held_product_origin_is_candidate_until_object_confirmed": True,
            "rule_scores_are_calibrated_probabilities": False
        },
        "thresholds": {
            "hand_min_duration_sec": HAND_MIN_DURATION_SEC,
            "approach_speed_px_sec": APPROACH_SPEED_THR,
            "return_speed_px_sec": RETURN_SPEED_THR,
            "crouched_retract_presentation_min_sec": (
                CROUCHED_RETRACT_PRESENTATION_MIN_SEC
            ),
            "inspect_product_min_duration_sec": (
                INSPECT_PRODUCT_MIN_DURATION_SEC
            ),
            "inspect_shelf_min_duration_sec": INSPECT_SHELF_MIN_DURATION_SEC,
            "inspect_product_min_hand_attention_share": (
                INSPECT_PRODUCT_MIN_HAND_ATTENTION_SHARE
            ),
            "inspect_product_search_sec": INSPECT_PRODUCT_SEARCH_SEC,
            "inspect_product_min_presentation_share": (
                INSPECT_PRODUCT_MIN_PRESENTATION_SHARE
            ),
            "inspect_product_presentation_only_min_duration_sec": (
                INSPECT_PRODUCT_PRESENTATION_ONLY_MIN_DURATION_SEC
            ),
            "inspect_shelf_max_body_distance_norm": (
                INSPECT_SHELF_MAX_BODY_DISTANCE_NORM
            ),
            "corridor_origin_persistence_weight": (
                CORRIDOR_ORIGIN_PERSISTENCE_WEIGHT
            ),
            "corridor_origin_ambiguity_ratio": (
                CORRIDOR_ORIGIN_AMBIGUITY_RATIO
            ),
            "corridor_origin_minimum_top_share": (
                CORRIDOR_ORIGIN_MIN_TOP_SHARE
            ),
            "corridor_origin_pre_exit_sec": CORRIDOR_ORIGIN_PRE_EXIT_SEC,
            "single_point_origin_support_weight": (
                SINGLE_POINT_ORIGIN_SUPPORT_WEIGHT
            ),
        }
    }

    for path in (manifest_path, LATEST_CLASSIFIER_MANIFEST):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print("Behaviour classification complete.")
    print("Input feature CSV:", input_csv)
    print("Classifier run ID:", classifier_run_id)
    print("Saved frame labels:", frame_output)
    print("Saved behaviour events:", event_output)
    if rendered_video is not None:
        print("Saved behaviour video:", rendered_video)
        print("Video frames annotated:", video_frames_written)
    print("Saved run manifest:", manifest_path)
    print("Events generated:", len(events))
    print("\nBehaviour event counts:")
    for behaviour in BEHAVIOURS:
        print(f"{behaviour}: {int((events['behaviour'] == behaviour).sum())}")

if __name__ == "__main__":
    main()
