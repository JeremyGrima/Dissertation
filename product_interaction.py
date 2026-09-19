import argparse
import json
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
LATEST_CLASSIFIER_MANIFEST = (
    OUTPUT_DIR / "latest_behaviour_classifier_run.json"
)
LATEST_EMERGENCE_MANIFEST = (
    OUTPUT_DIR / "latest_product_emergence_run.json"
)
LATEST_PRODUCT_MANIFEST = OUTPUT_DIR / "latest_product_interaction_run.json"

HELD_CONFIRMATION_MIN_SEC = 0.50
HELD_CORE_EVIDENCE_MIN_SEC = 0.30
HELD_MAX_EVIDENCE_GAP_SEC = 1.50  # Bridges brief evidence gaps without extending the last support.
PRESENTATION_TRACK_MIN_SEC = 0.40
OBJECT_DETECTION_TRACK_MIN_SEC = 0.20
MULTI_PRODUCT_PRESENTATION_MIN_SEC = 0.40
MULTI_PRODUCT_SEQUENCE_MAX_GAP_SEC = 4.00
HELD_WRIST_TO_BODY_NORM = 0.75
DETECTED_OBJECT_TO_WRIST_NORM = 0.35
RETURN_MAX_UNOBSERVED_GAP_SEC = 1.50
COMPARISON_MIN_OVERLAP_SEC = 0.50

REQUIRED_FEATURE_COLUMNS = {
    "frame",
    "time_sec",
    "person_box_height_px",
    "left_wrist_x",
    "left_wrist_y",
    "right_wrist_x",
    "right_wrist_y",
    "left_wrist_to_body_norm",
    "right_wrist_to_body_norm",
    "left_wrist_observed",
    "right_wrist_observed",
    "left_in_shelf",
    "right_in_shelf",
    "left_in_product",
    "right_in_product",
    "product_presentation_pose",
    "product_presentation_score",
    "attention_target",
    "held_product_available",
    "held_product_x",
    "held_product_y"
}

REQUIRED_EVENT_COLUMNS = {
    "behaviour",
    "start_frame",
    "end_frame",
    "wrist",
    "shelf_zone",
    "source_hand_episode",
    "product_origin_zone",
    "rule_score"
}

TRACK_COLUMNS = [
    "track_id",
    "wrist",
    "source_hand_episode",
    "origin_product_zone",
    "origin_product_zone_candidate",
    "origin_product_zone_alternative",
    "origin_product_zone_ambiguous",
    "origin_product_zone_confidence",
    "origin_product_zone_margin",
    "origin_product_zone_method",
    "origin_product_zone_evidence",
    "origin_shelf_zone",
    "pickup_candidate_start_frame",
    "pickup_candidate_end_frame",
    "pickup_candidate_start_time_sec",
    "pickup_candidate_end_time_sec",
    "product_held",
    "held_confirmation_type",
    "held_start_frame",
    "held_end_frame",
    "held_duration_sec",
    "inspect_product",
    "inspect_start_frame",
    "inspect_end_frame",
    "inspect_duration_sec",
    "returned",
    "return_start_frame",
    "return_end_frame",
    "return_product_zone",
    "return_shelf_zone",
    "returned_to_origin_zone",
    "handling_duration_sec",
    "comparison_candidate",
    "compared_with_track_ids",
    "track_confidence",
    "track_end_reason",
    "evidence"
]

INTERACTION_EVENT_COLUMNS = [
    "event_id",
    "interaction",
    "track_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "wrist",
    "origin_product_zone",
    "origin_product_zone_candidate",
    "origin_product_zone_alternative",
    "origin_product_zone_ambiguous",
    "origin_product_zone_confidence",
    "origin_product_zone_margin",
    "origin_product_zone_method",
    "origin_shelf_zone",
    "returned",
    "return_product_zone",
    "return_shelf_zone",
    "related_track_ids",
    "confidence",
    "evidence"
]

def as_boolean(series):
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
    return mapped.fillna(False).astype(bool)

def scalar_boolean(value):
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def episode_key(value):
    if pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric

def text_or_none(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value)

def same_resolved_path(first, second):
    if first is None or second is None:
        return False
    if str(first).strip() == "" or str(second).strip() == "":
        return False
    return Path(first).resolve() == Path(second).resolve()

def infer_fps_and_stride(features):
    frame_delta = pd.to_numeric(features["frame"], errors="coerce").diff()
    time_delta = pd.to_numeric(features["time_sec"], errors="coerce").diff()
    valid = (frame_delta > 0) & (time_delta > 0)

    if not valid.any():
        return 30.0, 1

    fps = float((frame_delta[valid] / time_delta[valid]).median())
    rounded_fps = round(fps)
    if abs(fps - rounded_fps) < 0.10:
        fps = float(rounded_fps)
    stride = max(1, int(round(frame_delta[frame_delta > 0].median())))
    return fps, stride

def bridge_short_false_gaps(condition, max_gap_rows):
    values = as_boolean(pd.Series(condition)).to_numpy(dtype=bool, copy=True)
    if max_gap_rows <= 0:
        return pd.Series(values, index=getattr(condition, "index", None))

    index = 0
    while index < len(values):
        if values[index]:
            index += 1
            continue

        start = index
        while index < len(values) and not values[index]:
            index += 1
        bounded = start > 0 and index < len(values)
        if bounded and index - start <= max_gap_rows:
            values[start:index] = True

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
        start_frame = int(features.iloc[start_index]["frame"])
        end_frame = int(features.iloc[end_index]["frame"])
        duration_sec = (end_frame - start_frame) / fps
        if duration_sec + 1e-9 >= min_duration_sec:
            runs.append({
                "start_index": start_index,
                "end_index": end_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec": duration_sec
            })
        index += 1

    return runs

def persistent_true_mask(
    condition,
    features,
    fps,
    min_duration_sec,
    max_gap_rows=0
):
    """Keep complete True runs only when they persist for long enough."""
    mask = pd.Series(False, index=features.index)
    runs = true_runs(
        condition.reset_index(drop=True),
        features.reset_index(drop=True),
        fps,
        min_duration_sec,
        max_gap_rows=max_gap_rows
    )
    for run in runs:
        mask |= (
            (features["frame"] >= run["start_frame"]) &
            (features["frame"] <= run["end_frame"])
        )
    return mask

def event_for_episode(events, behaviour, source_episode, wrist=None):
    subset = events[events["behaviour"] == behaviour]
    keys = subset["source_hand_episode"].map(episode_key)
    subset = subset[keys == source_episode]
    if wrist is not None:
        subset = subset[subset["wrist"] == wrist]
    if subset.empty:
        return None
    return subset.sort_values("start_frame").iloc[0]

def next_hand_event(hand_events, retract, wrist):
    candidates = hand_events[
        (hand_events["wrist"] == wrist) &
        (hand_events["start_frame"] > int(retract["end_frame"]))
    ]
    if candidates.empty:
        return None
    return candidates.sort_values("start_frame").iloc[0]

def detected_object_near_wrist(
    features, wrist, source_hand_episode=None
):
    available_column = f"{wrist}_held_product_available"  # Falls back to the earlier shared fields.
    x_column = f"{wrist}_held_product_x"
    y_column = f"{wrist}_held_product_y"
    if {
        available_column, x_column, y_column
    }.issubset(features.columns):
        available = as_boolean(features[available_column])
    else:
        available_column = "held_product_available"
        x_column = "held_product_x"
        y_column = "held_product_y"
        available = as_boolean(features[available_column])
    scale = pd.to_numeric(
        features["person_box_height_px"], errors="coerce"
    ).where(lambda values: values > 0)
    distance = np.hypot(
        pd.to_numeric(features[x_column], errors="coerce") -
        pd.to_numeric(features[f"{wrist}_wrist_x"], errors="coerce"),
        pd.to_numeric(features[y_column], errors="coerce") -
        pd.to_numeric(features[f"{wrist}_wrist_y"], errors="coerce")
    )
    near_wrist = (
        (distance / scale) <= DETECTED_OBJECT_TO_WRIST_NORM
    )
    episode_column = f"{wrist}_product_emergence_episode"
    if (
        source_hand_episode is not None
        and episode_column in features.columns
    ):
        detected_episode = pd.to_numeric(
            features[episode_column], errors="coerce"
        )
        available &= detected_episode == float(source_hand_episode)
    return available & near_wrist

def multi_product_holding_pose(features):
    if "multi_product_holding_pose" in features.columns:
        return as_boolean(features["multi_product_holding_pose"])

    required = {
        "left_wrist_observed",
        "right_wrist_observed",
        "inter_wrist_distance_norm",
        "wrist_midpoint_to_head_norm",
        "presentation_arm_support"
    }
    if not required.issubset(features.columns):
        return pd.Series(False, index=features.index)

    return (
        as_boolean(features["left_wrist_observed"]) &
        as_boolean(features["right_wrist_observed"]) &
        (pd.to_numeric(
            features["inter_wrist_distance_norm"], errors="coerce"
        ) <= 0.60).fillna(False) &
        (pd.to_numeric(
            features["wrist_midpoint_to_head_norm"], errors="coerce"
        ) <= 0.50).fillna(False) &
        as_boolean(features["presentation_arm_support"])
    )

def held_evidence_for_window(
    window,
    wrist,
    inspect_event,
    fps,
    stride,
    allow_multi_product_pose=False,
    source_hand_episode=None
):
    raw_actual_object = detected_object_near_wrist(
        window, wrist, source_hand_episode=source_hand_episode
    )
    raw_presentation = as_boolean(window["product_presentation_pose"])
    actual_object = persistent_true_mask(
        raw_actual_object,
        window,
        fps,
        OBJECT_DETECTION_TRACK_MIN_SEC,
        max_gap_rows=1
    )
    presentation = persistent_true_mask(
        raw_presentation,
        window,
        fps,
        PRESENTATION_TRACK_MIN_SEC,
        max_gap_rows=1
    )
    raw_multi_product = pd.Series(False, index=window.index)
    if allow_multi_product_pose:
        raw_multi_product = multi_product_holding_pose(window)
    multi_product = persistent_true_mask(
        raw_multi_product,
        window,
        fps,
        MULTI_PRODUCT_PRESENTATION_MIN_SEC,
        max_gap_rows=1
    )
    observed = as_boolean(window[f"{wrist}_wrist_observed"])
    near_body = (
        pd.to_numeric(
            window[f"{wrist}_wrist_to_body_norm"], errors="coerce"
        ) <= HELD_WRIST_TO_BODY_NORM
    ).fillna(False)
    attention_hands = window["attention_target"].fillna("uncertain").isin(
        ["hands", "held_product"]
    )
    attention_support = observed & near_body & attention_hands
    inspection = pd.Series(False, index=window.index)
    if inspect_event is not None:
        inspection = (
            (window["frame"] >= int(inspect_event["start_frame"])) &
            (window["frame"] <= int(inspect_event["end_frame"]))
        )

    core = actual_object | presentation | multi_product | inspection  # Attention cannot create or extend a track.
    combined = core
    max_gap_rows = max(
        1, int(round(HELD_MAX_EVIDENCE_GAP_SEC * fps / stride))
    )
    runs = true_runs(
        combined.reset_index(drop=True),
        window.reset_index(drop=True),
        fps,
        HELD_CONFIRMATION_MIN_SEC,
        max_gap_rows=max_gap_rows
    )

    for run in runs:
        run_mask = (
            (window["frame"] >= run["start_frame"]) &
            (window["frame"] <= run["end_frame"])
        )
        core_rows = int(core[run_mask].sum())
        core_duration = max(0.0, (core_rows - 1) * stride / fps)
        if core_duration + 1e-9 < HELD_CORE_EVIDENCE_MIN_SEC:
            continue

        presentation_share = float(presentation[run_mask].mean())
        multi_product_share = float(multi_product[run_mask].mean())
        attention_share = float(attention_support[run_mask].mean())
        actual_share = float(actual_object[run_mask].mean())
        actual_object_source = None
        source_column = f"{wrist}_held_product_source"
        if actual_share > 0 and source_column in window.columns:
            source_values = window.loc[
                run_mask & actual_object, source_column
            ].dropna()
            if not source_values.empty:
                actual_object_source = str(
                    source_values.value_counts().index[0]
                )
        inspection_share = float(inspection[run_mask].mean())
        maximum_presentation_score = float(pd.to_numeric(
            window.loc[run_mask, "product_presentation_score"],
            errors="coerce"
        ).fillna(0.0).max())
        if "multi_product_holding_score" in window.columns:
            maximum_presentation_score = max(
                maximum_presentation_score,
                float(pd.to_numeric(
                    window.loc[run_mask, "multi_product_holding_score"],
                    errors="coerce"
                ).fillna(0.0).max())
            )

        if (
            actual_share > 0
            and actual_object_source == "shelf_exit_emergence"
        ):
            confirmation_type = "shelf_exit_emergence"
        elif actual_share > 0:
            confirmation_type = "object_detector"
        elif multi_product_share > 0 and presentation_share > 0:
            confirmation_type = "single_and_multi_product_pose_proxy"
        elif multi_product_share > 0:
            confirmation_type = "multi_product_pose_proxy"
        elif inspection_share > 0 and presentation_share > 0:
            confirmation_type = "inspection_and_presentation_pose_proxy"
        elif presentation_share > 0:
            confirmation_type = "presentation_pose_proxy"
        else:
            confirmation_type = "inspection_pose_proxy"

        return {
            **run,
            "confirmation_type": confirmation_type,
            "actual_object_source": actual_object_source,
            "actual_object_share": actual_share,
            "presentation_share": presentation_share,
            "multi_product_share": multi_product_share,
            "attention_share": attention_share,
            "inspection_share": inspection_share,
            "maximum_presentation_score": maximum_presentation_score
        }

    return None

def make_interaction_event(
    interaction,
    track,
    start_frame,
    end_frame,
    fps,
    confidence,
    evidence,
    related_track_ids=None
):
    return {
        "event_id": None,
        "interaction": interaction,
        "track_id": track["track_id"],
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time_sec": int(start_frame) / fps,
        "end_time_sec": int(end_frame) / fps,
        "duration_sec": (int(end_frame) - int(start_frame)) / fps,
        "wrist": track["wrist"],
        "origin_product_zone": track["origin_product_zone"],
        "origin_product_zone_candidate": track[
            "origin_product_zone_candidate"
        ],
        "origin_product_zone_alternative": track[
            "origin_product_zone_alternative"
        ],
        "origin_product_zone_ambiguous": track[
            "origin_product_zone_ambiguous"
        ],
        "origin_product_zone_confidence": track[
            "origin_product_zone_confidence"
        ],
        "origin_product_zone_margin": track[
            "origin_product_zone_margin"
        ],
        "origin_product_zone_method": track[
            "origin_product_zone_method"
        ],
        "origin_shelf_zone": track["origin_shelf_zone"],
        "returned": bool(track["returned"]),
        "return_product_zone": track["return_product_zone"],
        "return_shelf_zone": track["return_shelf_zone"],
        "related_track_ids": related_track_ids,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
        "evidence": evidence
    }

def build_product_tracks(features, behaviour_events):
    missing_features = REQUIRED_FEATURE_COLUMNS.difference(features.columns)
    if missing_features:
        raise ValueError(
            "Feature CSV is missing required columns: " +
            ", ".join(sorted(missing_features))
        )
    missing_events = REQUIRED_EVENT_COLUMNS.difference(
        behaviour_events.columns
    )
    if missing_events:
        raise ValueError(
            "Behaviour-event CSV is missing required columns: " +
            ", ".join(sorted(missing_events))
        )

    features = features.sort_values("frame").reset_index(drop=True).copy()
    events = behaviour_events.sort_values("start_frame").reset_index(
        drop=True
    ).copy()
    events["start_frame"] = pd.to_numeric(
        events["start_frame"], errors="coerce"
    )
    events["end_frame"] = pd.to_numeric(
        events["end_frame"], errors="coerce"
    )
    fps, stride = infer_fps_and_stride(features)
    last_frame = int(features["frame"].max())
    hand_events = events[events["behaviour"] == "hand_in_shelf"]
    retract_events = events[events["behaviour"] == "retract_from_shelf"]
    tracks = []

    for track_number, (_, retract) in enumerate(
        retract_events.iterrows(), start=1
    ):
        wrist = str(retract["wrist"])
        source_episode = episode_key(retract["source_hand_episode"])
        hand = event_for_episode(
            events, "hand_in_shelf", source_episode, wrist=wrist
        )
        inspect = event_for_episode(
            events, "inspect_product", source_episode, wrist=wrist
        )
        origin_product = text_or_none(retract.get("product_origin_zone"))
        if origin_product is None and hand is not None:
            origin_product = text_or_none(hand.get("product_origin_zone"))
        origin_candidate = text_or_none(
            retract.get("product_origin_zone_candidate")
        )
        if origin_candidate is None and hand is not None:
            origin_candidate = text_or_none(
                hand.get("product_origin_zone_candidate")
            )
        origin_candidate = origin_candidate or origin_product
        origin_alternative = text_or_none(
            retract.get("product_origin_zone_alternative")
        )
        if origin_alternative is None and hand is not None:
            origin_alternative = text_or_none(
                hand.get("product_origin_zone_alternative")
            )
        origin_ambiguous = scalar_boolean(
            retract.get("product_origin_ambiguous")
        )
        if hand is not None:
            origin_ambiguous = origin_ambiguous or scalar_boolean(
                hand.get("product_origin_ambiguous")
            )
        origin_confidence = pd.to_numeric(
            retract.get("product_origin_confidence"), errors="coerce"
        )
        origin_margin = pd.to_numeric(
            retract.get("product_origin_margin"), errors="coerce"
        )
        origin_method = text_or_none(
            retract.get("product_origin_method")
        )
        origin_evidence = text_or_none(
            retract.get("product_origin_evidence")
        )
        if hand is not None:
            if pd.isna(origin_confidence):
                origin_confidence = pd.to_numeric(
                    hand.get("product_origin_confidence"), errors="coerce"
                )
            if pd.isna(origin_margin):
                origin_margin = pd.to_numeric(
                    hand.get("product_origin_margin"), errors="coerce"
                )
            origin_method = origin_method or text_or_none(
                hand.get("product_origin_method")
            )
            origin_evidence = origin_evidence or text_or_none(
                hand.get("product_origin_evidence")
            )
        origin_shelf = text_or_none(retract.get("shelf_zone"))
        if origin_shelf is None and hand is not None:
            origin_shelf = text_or_none(hand.get("shelf_zone"))

        upcoming_hand = next_hand_event(hand_events, retract, wrist)
        upcoming_retract = None
        upcoming_origin = None
        upcoming_origin_ambiguous = False
        if upcoming_hand is not None:
            upcoming_episode = episode_key(
                upcoming_hand["source_hand_episode"]
            )
            upcoming_retract = event_for_episode(
                events,
                "retract_from_shelf",
                upcoming_episode,
                wrist=wrist
            )
            upcoming_origin = text_or_none(
                upcoming_hand.get("product_origin_zone")
            )
            upcoming_origin_ambiguous = scalar_boolean(
                upcoming_hand.get("product_origin_ambiguous")
            )
        upcoming_is_distinct_pickup = (
            upcoming_retract is not None and
            not origin_ambiguous and
            not upcoming_origin_ambiguous and
            origin_product is not None and
            upcoming_origin is not None and
            upcoming_origin != origin_product and
            (
                int(upcoming_hand["start_frame"]) -
                int(retract["end_frame"])
            ) / fps <= MULTI_PRODUCT_SEQUENCE_MAX_GAP_SEC
        )

        earlier_retracts = retract_events[
            (retract_events["wrist"] == wrist) &
            (retract_events["end_frame"] < int(retract["start_frame"]))
        ].sort_values("end_frame")
        preceded_by_distinct_pickup = False
        if not earlier_retracts.empty:
            previous_retract = earlier_retracts.iloc[-1]
            previous_origin = text_or_none(
                previous_retract.get("product_origin_zone")
            )
            previous_origin_ambiguous = scalar_boolean(
                previous_retract.get("product_origin_ambiguous")
            )
            previous_gap_sec = (
                int(retract["start_frame"]) -
                int(previous_retract["end_frame"])
            ) / fps
            preceded_by_distinct_pickup = (
                not origin_ambiguous and
                not previous_origin_ambiguous and
                previous_origin is not None and
                origin_product is not None and
                previous_origin != origin_product and
                previous_gap_sec <= MULTI_PRODUCT_SEQUENCE_MAX_GAP_SEC
            )

        multi_product_context = (
            upcoming_is_distinct_pickup or preceded_by_distinct_pickup
        )
        following_hand = None
        if upcoming_is_distinct_pickup:
            following_hand = next_hand_event(
                hand_events, upcoming_retract, wrist
            )
        window_end = (
            last_frame
            if (
                (upcoming_is_distinct_pickup and following_hand is None) or
                (not upcoming_is_distinct_pickup and upcoming_hand is None)
            )
            else (
                int(following_hand["start_frame"]) - stride
                if upcoming_is_distinct_pickup
                else int(upcoming_hand["start_frame"]) - stride
            )
        )
        window = features[
            (features["frame"] > int(retract["end_frame"])) &
            (features["frame"] <= window_end)
        ].copy()
        held = None
        if not window.empty:
            held = held_evidence_for_window(
                window,
                wrist,
                inspect,
                fps,
                stride,
                allow_multi_product_pose=multi_product_context,
                source_hand_episode=source_episode
            )

        returned = False
        return_product = None
        return_shelf = None
        return_start = None
        return_end = None
        return_gap_sec = None
        if (
            held is not None and
            upcoming_hand is not None and
            not upcoming_is_distinct_pickup
        ):
            return_gap_sec = (
                int(upcoming_hand["start_frame"]) - held["end_frame"]
            ) / fps
            returned = return_gap_sec <= RETURN_MAX_UNOBSERVED_GAP_SEC
            if returned:
                return_product = text_or_none(
                    upcoming_hand.get("product_origin_zone")
                )
                return_shelf = text_or_none(upcoming_hand.get("shelf_zone"))
                return_start = int(upcoming_hand["start_frame"])
                return_end = int(upcoming_hand["end_frame"])

        inspect_start = None if inspect is None else int(inspect["start_frame"])
        inspect_end = None if inspect is None else int(inspect["end_frame"])
        inspect_duration = (
            None
            if inspect is None
            else (inspect_end - inspect_start) / fps
        )
        pickup_score = float(retract.get("rule_score", 0.0) or 0.0)
        inspect_score = (
            0.0
            if inspect is None
            else float(inspect.get("rule_score", 0.0) or 0.0)
        )
        if held is None:
            track_confidence = 0.35 * pickup_score
            evidence = (
                "retraction from a product zone, but no persistent held-item "
                "or presentation evidence was found"
            )
            end_reason = "pickup_not_confirmed"
        else:
            track_confidence = (
                0.25 * pickup_score +
                0.30 * held["maximum_presentation_score"] +
                0.20 * inspect_score +
                0.15 * held["attention_share"] +
                0.10 * min(held["duration_sec"] / 2.0, 1.0)
            )
            if held["actual_object_share"] > 0:
                track_confidence = max(track_confidence, 0.85)
            evidence = (
                f"post-retraction held evidence via "
                f"{held['confirmation_type']}; "
                f"presentation share={held['presentation_share']:.2f}; "
                f"multi-product pose share="
                f"{held['multi_product_share']:.2f}; "
                f"attention-to-hands share={held['attention_share']:.2f}"
            )
            if returned:
                evidence += (
                    f"; later same-wrist shelf entry after "
                    f"{return_gap_sec:.2f}s supports return"
                )
                end_reason = "returned"
            else:
                if upcoming_is_distinct_pickup:
                    evidence += (
                        "; a different-origin pickup followed, so the next "
                        "shelf entry was not forced to be a return"
                    )
                else:
                    evidence += "; no supported return observed"
                end_reason = (
                    "observation_ended_not_returned"
                    if upcoming_hand is None and following_hand is None
                    else "continued_through_distinct_pickup"
                    if upcoming_is_distinct_pickup
                    else "later_return_not_supported"
                )

        held_start = None if held is None else int(held["start_frame"])
        held_end = None if held is None else int(held["end_frame"])
        held_duration = None if held is None else float(held["duration_sec"])
        handling_duration = held_duration  # Excludes any unseen gap before a later return.

        tracks.append({
            "track_id": f"Product_{track_number:02d}",
            "wrist": wrist,
            "source_hand_episode": source_episode,
            "origin_product_zone": origin_product,
            "origin_product_zone_candidate": origin_candidate,
            "origin_product_zone_alternative": origin_alternative,
            "origin_product_zone_ambiguous": origin_ambiguous,
            "origin_product_zone_confidence": (
                None if pd.isna(origin_confidence) else float(origin_confidence)
            ),
            "origin_product_zone_margin": (
                None if pd.isna(origin_margin) else float(origin_margin)
            ),
            "origin_product_zone_method": origin_method,
            "origin_product_zone_evidence": origin_evidence,
            "origin_shelf_zone": origin_shelf,
            "pickup_candidate_start_frame": int(retract["start_frame"]),
            "pickup_candidate_end_frame": int(retract["end_frame"]),
            "pickup_candidate_start_time_sec": int(retract["start_frame"]) / fps,
            "pickup_candidate_end_time_sec": int(retract["end_frame"]) / fps,
            "product_held": held is not None,
            "held_confirmation_type": (
                None if held is None else held["confirmation_type"]
            ),
            "held_start_frame": held_start,
            "held_end_frame": held_end,
            "held_duration_sec": held_duration,
            "inspect_product": inspect is not None,
            "inspect_start_frame": inspect_start,
            "inspect_end_frame": inspect_end,
            "inspect_duration_sec": inspect_duration,
            "returned": returned,
            "return_start_frame": return_start,
            "return_end_frame": return_end,
            "return_product_zone": return_product,
            "return_shelf_zone": return_shelf,
            "returned_to_origin_zone": (
                None
                if not returned or origin_product is None or return_product is None
                else origin_product == return_product
            ),
            "handling_duration_sec": handling_duration,
            "comparison_candidate": False,
            "compared_with_track_ids": None,
            "track_confidence": round(
                float(np.clip(track_confidence, 0.0, 1.0)), 3
            ),
            "track_end_reason": end_reason,
            "evidence": evidence,
            "_inspect_score": inspect_score,
            "_pickup_score": pickup_score
        })

    interaction_events = []
    for track in tracks:
        interaction_events.append(make_interaction_event(
            "pickup_candidate",
            track,
            track["pickup_candidate_start_frame"],
            track["pickup_candidate_end_frame"],
            fps,
            track["_pickup_score"],
            "hand retracts after occupying matching shelf and product zones"
        ))
        if track["product_held"]:
            interaction_events.append(make_interaction_event(
                "product_held",
                track,
                track["held_start_frame"],
                track["held_end_frame"],
                fps,
                track["track_confidence"],
                track["evidence"]
            ))
        if track["inspect_product"]:
            interaction_events.append(make_interaction_event(
                "inspect_product",
                track,
                track["inspect_start_frame"],
                track["inspect_end_frame"],
                fps,
                track["_inspect_score"],
                "linked behaviour-classifier inspection interval"
            ))
        if track["returned"]:
            short_gap_support = max(
                0.0,
                1.0 - (
                    (track["return_start_frame"] - track["held_end_frame"]) /
                    fps / RETURN_MAX_UNOBSERVED_GAP_SEC
                )
            )
            return_confidence = (
                0.55 * track["track_confidence"] +
                0.25 * short_gap_support +
                0.20 * float(track["returned_to_origin_zone"] is True)
            )
            interaction_events.append(make_interaction_event(
                "return_candidate",
                track,
                track["return_start_frame"],
                track["return_end_frame"],
                fps,
                return_confidence,
                (
                    "confirmed held candidate later enters a shelf/product "
                    "zone with the same wrist"
                )
            ))

    confirmed_tracks = [track for track in tracks if track["product_held"]]
    comparison_events = []
    for first_index, first in enumerate(confirmed_tracks):
        for second in confirmed_tracks[first_index + 1:]:
            if first["origin_product_zone"] == second["origin_product_zone"]:
                continue
            use_inspection_overlap = False
            overlap_start = None
            overlap_end = None
            overlap_sec = -1.0
            if first["inspect_product"] and second["inspect_product"]:
                inspect_overlap_start = max(
                    first["inspect_start_frame"],
                    second["inspect_start_frame"]
                )
                inspect_overlap_end = min(
                    first["inspect_end_frame"],
                    second["inspect_end_frame"]
                )
                inspect_overlap_sec = (
                    inspect_overlap_end - inspect_overlap_start
                ) / fps
                if (
                    inspect_overlap_sec + 1e-9 >=
                    COMPARISON_MIN_OVERLAP_SEC
                ):
                    use_inspection_overlap = True
                    overlap_start = inspect_overlap_start
                    overlap_end = inspect_overlap_end
                    overlap_sec = inspect_overlap_sec

            if not use_inspection_overlap:
                overlap_start = max(
                    first["held_start_frame"], second["held_start_frame"]
                )
                overlap_end = min(
                    first["held_end_frame"], second["held_end_frame"]
                )
                overlap_sec = (overlap_end - overlap_start) / fps
            if overlap_sec + 1e-9 < COMPARISON_MIN_OVERLAP_SEC:
                continue

            first["comparison_candidate"] = True
            second["comparison_candidate"] = True
            first_links = set(
                (first["compared_with_track_ids"] or "").split("|")
            )
            second_links = set(
                (second["compared_with_track_ids"] or "").split("|")
            )
            first_links.discard("")
            second_links.discard("")
            first_links.add(second["track_id"])
            second_links.add(first["track_id"])
            first["compared_with_track_ids"] = "|".join(sorted(first_links))
            second["compared_with_track_ids"] = "|".join(sorted(second_links))
            comparison_confidence = (
                0.40 * first["track_confidence"] +
                0.40 * second["track_confidence"] +
                0.20 * min(overlap_sec / 2.0, 1.0)
            )
            comparison_track = dict(first)
            comparison_track["track_id"] = (
                f"{first['track_id']}|{second['track_id']}"
            )
            comparison_track["wrist"] = (
                f"{first['wrist']}|{second['wrist']}"
            )
            comparison_track["origin_product_zone"] = (
                f"{first['origin_product_zone']}|"
                f"{second['origin_product_zone']}"
            )
            comparison_track["origin_shelf_zone"] = (
                f"{first['origin_shelf_zone']}|{second['origin_shelf_zone']}"
            )
            comparison_events.append(make_interaction_event(
                "comparison_candidate",
                comparison_track,
                overlap_start,
                overlap_end,
                fps,
                comparison_confidence,
                (
                    "different-origin product candidates are inspected "
                    "concurrently"
                    if use_inspection_overlap
                    else "different-origin product candidates are held concurrently"
                ),
                related_track_ids=(
                    f"{first['track_id']}|{second['track_id']}"
                )
            ))

    interaction_events.extend(comparison_events)
    interaction_events.sort(key=lambda event: (
        event["start_frame"], event["interaction"], event["track_id"]
    ))
    for event_id, event in enumerate(interaction_events, start=1):
        event["event_id"] = event_id

    public_tracks = [
        {column: track.get(column) for column in TRACK_COLUMNS}
        for track in tracks
    ]
    track_df = pd.DataFrame(public_tracks, columns=TRACK_COLUMNS)
    event_df = pd.DataFrame(
        interaction_events, columns=INTERACTION_EVENT_COLUMNS
    )
    return track_df, event_df, fps, stride

TRACK_COLOURS = [
    (255, 80, 255),
    (0, 215, 255),
    (255, 170, 0),
    (0, 220, 120),
    (180, 100, 255),
    (255, 220, 80)
]

def draw_text(frame, text, x, y, colour, scale=0.48, thickness=1):
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

def track_state_at_frame(track, frame_number):
    if (
        track["returned"] and
        track["return_start_frame"] <= frame_number <= track["return_end_frame"]
    ):
        return "RETURN CANDIDATE"
    if (
        track["inspect_product"] and
        track["inspect_start_frame"] <= frame_number <= track["inspect_end_frame"]
    ):
        return "INSPECTING"
    if track["product_held"]:
        display_end = track["held_end_frame"]
        if track["held_start_frame"] <= frame_number <= display_end:
            return "HELD CANDIDATE"
    if (
        track["pickup_candidate_start_frame"] <= frame_number <=
        track["pickup_candidate_end_frame"]
    ):
        return "PICKUP CANDIDATE"
    return None

def annotate_product_video(
    input_video,
    features,
    tracks,
    interaction_events,
    output_video
):
    if cv2 is None:
        raise RuntimeError("OpenCV is required to render the product video.")

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    track_rows = tracks.to_dict("records")
    comparison_rows = interaction_events[
        interaction_events["interaction"] == "comparison_candidate"
    ].to_dict("records")
    written = 0

    for _, feature_row in features.sort_values("frame").iterrows():
        ok, frame = capture.read()
        if not ok:
            break
        frame_number = int(feature_row["frame"])
        active = []
        for index, track in enumerate(track_rows):
            state = track_state_at_frame(track, frame_number)
            if state is not None:
                active.append((index, track, state))

        active_comparisons = [
            event for event in comparison_rows
            if event["start_frame"] <= frame_number <= event["end_frame"]
        ]
        if active or active_comparisons:
            line_count = len(active) + len(active_comparisons) + 1
            panel_height = 28 + line_count * 24
            panel_y = max(10, height - panel_height - 14)
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (10, panel_y),
                (min(width - 10, 650), height - 10),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
            draw_text(
                frame,
                "PRODUCT INTERACTION (candidate evidence)",
                24,
                panel_y + 24,
                (255, 255, 255),
                scale=0.52,
                thickness=1
            )
            y = panel_y + 50
            for index, track, state in active:
                colour = TRACK_COLOURS[index % len(TRACK_COLOURS)]
                return_suffix = (
                    " | RETURN OBSERVED"
                    if state == "RETURN CANDIDATE"
                    else ""
                )
                draw_text(
                    frame,
                    (
                        f"{track['track_id']} | "
                        f"{'AMBIG ' if track.get('origin_product_zone_ambiguous') else ''}"
                        f"{track['origin_product_zone']}"
                        f"{' / ' + str(track.get('origin_product_zone_alternative')) if track.get('origin_product_zone_ambiguous') and track.get('origin_product_zone_alternative') else ''} | "
                        f"{state}{return_suffix}"
                    ),
                    24,
                    y,
                    colour,
                    scale=0.47,
                    thickness=1
                )
                wrist = track["wrist"]
                wrist_x = feature_row.get(f"{wrist}_wrist_x")
                wrist_y = feature_row.get(f"{wrist}_wrist_y")
                if pd.notna(wrist_x) and pd.notna(wrist_y):
                    point = (int(round(wrist_x)), int(round(wrist_y)))
                    cv2.circle(frame, point, 13, colour, 2, cv2.LINE_AA)
                    draw_text(
                        frame,
                        track["track_id"],
                        point[0] + 14,
                        point[1] - 10,
                        colour,
                        scale=0.42,
                        thickness=1
                    )
                y += 24
            for comparison in active_comparisons:
                draw_text(
                    frame,
                    (
                        "COMPARISON CANDIDATE: " +
                        str(comparison["related_track_ids"])
                    ),
                    24,
                    y,
                    (0, 215, 255),
                    scale=0.47,
                    thickness=1
                )
                y += 24

        writer.write(frame)
        written += 1

    capture.release()
    writer.release()
    return written

def resolve_inputs(args):
    classifier_manifest = {}
    if LATEST_CLASSIFIER_MANIFEST.is_file():
        with open(LATEST_CLASSIFIER_MANIFEST, "r", encoding="utf-8") as file:
            classifier_manifest = json.load(file)

    event_csv = args.behaviour_events or Path(
        classifier_manifest.get("event_output_csv", "")
    )
    base_feature_csv = Path(
        classifier_manifest.get("input_feature_csv", "")
    )
    feature_csv = args.features or base_feature_csv
    input_video = args.input_video
    if input_video is None:
        video_value = classifier_manifest.get("behaviour_video")
        if video_value:
            input_video = Path(video_value)

    emergence_manifest = {}
    if args.features is None and LATEST_EMERGENCE_MANIFEST.is_file():
        with open(
            LATEST_EMERGENCE_MANIFEST, "r", encoding="utf-8"
        ) as file:
            candidate_manifest = json.load(file)
        augmented_value = candidate_manifest.get(
            "augmented_feature_csv"
        )
        compatible = (
            candidate_manifest.get("classifier_run_id")
            == classifier_manifest.get("classifier_run_id")
            and str(candidate_manifest.get("pose_run_id", ""))
            == str(classifier_manifest.get("pose_run_id", ""))
            and same_resolved_path(
                candidate_manifest.get("input_features"),
                base_feature_csv,
            )
            and same_resolved_path(
                candidate_manifest.get("input_behaviour_events"),
                event_csv,
            )
            and augmented_value
            and Path(augmented_value).is_file()
        )
        if compatible:
            emergence_manifest = candidate_manifest
            feature_csv = Path(augmented_value)
            if args.input_video is None:
                emergence_video = candidate_manifest.get(
                    "annotated_video"
                )
                if emergence_video and Path(emergence_video).is_file():
                    input_video = Path(emergence_video)
            classifier_manifest[
                "product_emergence_run_id"
            ] = candidate_manifest.get("product_emergence_run_id")
            classifier_manifest[
                "product_emergence_event_csv"
            ] = candidate_manifest.get("emergence_event_csv")

    if not event_csv.is_file():
        raise FileNotFoundError(
            "No behaviour event CSV found. Run behaviour_classifier.py first "
            "or pass --behaviour-events."
        )
    if not feature_csv.is_file():
        raise FileNotFoundError(
            "No behaviour feature CSV found. Run behaviour_features.py first "
            "or pass --features."
        )
    return event_csv, feature_csv, input_video, classifier_manifest

def make_output_paths(classifier_manifest):
    pose_run_id = str(classifier_manifest.get("pose_run_id", "unknown"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{pose_run_id}_{run_id}"
    return {
        "run_id": run_id,
        "pose_run_id": pose_run_id,
        "tracks": OUTPUT_DIR / f"product_tracks_{stem}.csv",
        "events": OUTPUT_DIR / f"product_interaction_events_{stem}.csv",
        "video": OUTPUT_DIR / f"product_interaction_video_{stem}.mp4",
        "manifest": OUTPUT_DIR / f"product_interaction_run_{stem}.json"
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build candidate product tracks from behaviour outputs."
    )
    parser.add_argument("--behaviour-events", type=Path, default=None)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--tracks-output", type=Path, default=None)
    parser.add_argument("--events-output", type=Path, default=None)
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Generate CSV outputs without rendering an annotated video."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    event_csv, feature_csv, input_video, classifier_manifest = resolve_inputs(
        args
    )
    paths = make_output_paths(classifier_manifest)
    tracks_output = args.tracks_output or paths["tracks"]
    events_output = args.events_output or paths["events"]
    video_output = args.video_output or paths["video"]
    features = pd.read_csv(feature_csv, low_memory=False)
    behaviour_events = pd.read_csv(event_csv)
    tracks, interaction_events, fps, stride = build_product_tracks(
        features, behaviour_events
    )
    tracks_output.parent.mkdir(parents=True, exist_ok=True)
    events_output.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(tracks_output, index=False, float_format="%.3f")
    interaction_events.to_csv(
        events_output, index=False, float_format="%.3f"
    )

    rendered_video = None
    frames_written = 0
    if not args.skip_video:
        if input_video is None or not input_video.is_file():
            print(
                "Warning: no matching behaviour video was found; CSV outputs "
                "were still generated."
            )
        elif cv2 is None:
            print(
                "Warning: OpenCV is unavailable; CSV outputs were generated "
                "but video rendering was skipped."
            )
        else:
            video_output.parent.mkdir(parents=True, exist_ok=True)
            frames_written = annotate_product_video(
                input_video,
                features,
                tracks,
                interaction_events,
                video_output
            )
            rendered_video = video_output

    event_counts = {
        str(key): int(value)
        for key, value in interaction_events["interaction"].value_counts().items()
    }
    manifest = {
        "product_interaction_run_id": paths["run_id"],
        "pose_run_id": paths["pose_run_id"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_behaviour_events": str(event_csv.resolve()),
        "input_features": str(feature_csv.resolve()),
        "input_video": (
            None if input_video is None else str(input_video.resolve())
        ),
        "product_emergence_run_id": classifier_manifest.get(
            "product_emergence_run_id"
        ),
        "product_emergence_event_csv": classifier_manifest.get(
            "product_emergence_event_csv"
        ),
        "track_output_csv": str(tracks_output.resolve()),
        "event_output_csv": str(events_output.resolve()),
        "annotated_video": (
            None if rendered_video is None else str(rendered_video.resolve())
        ),
        "video_frames_written": frames_written,
        "fps": fps,
        "stride": stride,
        "tracks_generated": len(tracks),
        "held_tracks": int(tracks["product_held"].sum()),
        "returned_tracks": int(tracks["returned"].sum()),
        "comparison_tracks": int(tracks["comparison_candidate"].sum()),
        "ambiguous_origin_tracks": int(
            tracks["origin_product_zone_ambiguous"].sum()
        ),
        "event_counts": event_counts,
        "interpretation": {
            "returned_false_means": "no supported return observed",
            "returned_false_does_not_prove_purchase": True,
            "product_held_is_candidate_without_object_detector": True,
            "shelf_exit_emergence_is_object_confirmation": True,
            "distinct_following_pickup_is_not_forced_to_be_return": True,
            "multi_product_pose_requires_distinct_pickup_context": True,
            "shelf_fields_retained_for_audit_and_comparison": True,
            "ambiguous_origin_does_not_create_distinct_product_context": True
        },
        "thresholds": {
            "held_confirmation_min_sec": HELD_CONFIRMATION_MIN_SEC,
            "held_core_evidence_min_sec": HELD_CORE_EVIDENCE_MIN_SEC,
            "held_max_evidence_gap_sec": HELD_MAX_EVIDENCE_GAP_SEC,
            "presentation_track_min_sec": PRESENTATION_TRACK_MIN_SEC,
            "object_detection_track_min_sec": (
                OBJECT_DETECTION_TRACK_MIN_SEC
            ),
            "multi_product_presentation_min_sec": (
                MULTI_PRODUCT_PRESENTATION_MIN_SEC
            ),
            "multi_product_sequence_max_gap_sec": (
                MULTI_PRODUCT_SEQUENCE_MAX_GAP_SEC
            ),
            "return_max_unobserved_gap_sec": (
                RETURN_MAX_UNOBSERVED_GAP_SEC
            ),
            "comparison_min_overlap_sec": COMPARISON_MIN_OVERLAP_SEC
        }
    }
    for path in (paths["manifest"], LATEST_PRODUCT_MANIFEST):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

    print("Product interaction tracking complete.")
    print("Input behaviour events:", event_csv)
    print("Input features:", feature_csv)
    print("Saved product tracks:", tracks_output)
    print("Saved interaction events:", events_output)
    if rendered_video is not None:
        print("Saved annotated video:", rendered_video)
        print("Video frames annotated:", frames_written)
    print("Tracks generated:", len(tracks))
    print("Held-product candidates:", int(tracks["product_held"].sum()))
    print("Returned candidates:", int(tracks["returned"].sum()))
    print(
        "Comparison candidates:",
        int((interaction_events["interaction"] == "comparison_candidate").sum())
    )
    print(
        "Note: returned=False means no return was observed; it does not by "
        "itself prove that the product was purchased."
    )

if __name__ == "__main__":
    main()
