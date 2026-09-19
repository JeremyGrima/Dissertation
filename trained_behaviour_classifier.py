import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as error:
    raise SystemExit(
        "trained_behaviour_classifier.py requires joblib/scikit-learn in the "
        "dissertation Python environment."
    ) from error

from behaviour_classifier import ( #Two-stage trained MERL classifier 
    BEHAVIOURS,
    PRIMARY_PRIORITY,
    add_smoothed_evidence,
    annotate_behaviour_video,
    contact_origin_decision,
    create_frame_predictions,
    detect_hand_episodes,
    find_latest_feature_csv,
    find_pose_video,
    infer_fps_and_stride,
    make_event,
    manifest_for_feature_csv,
)
from merl_ml import (
    MODEL_VERSION,
    as_boolean,
    build_temporal_features,
    mask_runs,
    smooth_probability,
)
from product_origin_v2 import (
    load_calibration,
    load_product_zones,
    origin_decision,
)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "pose_tests"
LATEST_CLASSIFIER_MANIFEST = OUTPUT_DIR / "latest_behaviour_classifier_run.json"
DEFAULT_MODEL_MANIFEST = PROJECT_DIR / "models" / MODEL_VERSION / "hybrid_v2_model_manifest.json"
DEFAULT_ORIGIN_CALIBRATION = PROJECT_DIR / "models" / MODEL_VERSION / "product_origin_calibration.json"
PUBLIC_EVENT_COLUMNS = [
    "event_id", "behaviour", "start_frame", "end_frame", "start_time_sec",
    "end_time_sec", "duration_sec", "wrist", "shelf_zone", "product_zones",
    "rule_score", "evidence", "source_hand_episode", "product_origin_zone",
    "product_origin_zone_candidate", "product_origin_zone_alternative",
    "product_origin_ambiguous", "product_origin_confidence",
    "product_origin_margin", "product_origin_method", "product_origin_evidence",
    "held_product_origin_candidate", "held_product_detection_confirmed",
    "presentation_pose_share", "attention_to_hands_share",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trained hybrid_v2 MERL behaviour classifier."
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--zones-json", type=Path, default=PROJECT_DIR / "frames" / "zones.json")
    parser.add_argument("--origin-calibration", type=Path, default=DEFAULT_ORIGIN_CALIBRATION)
    parser.add_argument(
        "--emergence-features",
        type=Path,
        default=None,
        help="Augmented features used to strictly confirm inspect_product.",
    )
    parser.add_argument(
        "--stage", choices=("initial", "final"), default="final",
        help="Initial creates shelf-exit events; final applies emergence gating.",
    )
    parser.add_argument("--pose-run-id", default=None)
    parser.add_argument("--frame-output-csv", type=Path, default=None)
    parser.add_argument("--event-output-csv", type=Path, default=None)
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Refuse a model not explicitly frozen after validation tuning.",
    )
    return parser.parse_args()

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)

def model_probability(model, table, positive_class):
    probability = model.predict_proba(table)
    classes = list(model.named_steps["classifier"].classes_)
    if positive_class not in classes:
        return np.zeros(len(table), dtype=float)
    return probability[:, classes.index(positive_class)]

def predict_probabilities(bundle, model_rows):
    columns = bundle["numeric_feature_columns"] + bundle["categorical_feature_columns"]
    missing = sorted(set(columns) - set(model_rows.columns))
    if missing:
        raise ValueError("Model input is missing columns: " + ", ".join(missing))
    table = model_rows[columns]
    stage1 = bundle["stage1_model"]
    stage2 = bundle["stage2_model"]
    event_probability = model_probability(stage1, table, 1)
    conditional = stage2.predict_proba(table)
    classes = [str(value) for value in stage2.named_steps["classifier"].classes_]
    scores = {"event_probability": event_probability}
    for behaviour in BEHAVIOURS:
        scores[behaviour] = (
            event_probability * conditional[:, classes.index(behaviour)]
            if behaviour in classes else np.zeros(len(table), dtype=float)
        )
    return scores

def decoded_runs(features, probabilities, decoder):
    fps, stride = infer_fps_and_stride(features)
    sample_sec = stride / fps
    frames = features["frame"].to_numpy(dtype=int)
    output = {}
    smoothed = {}
    for behaviour in BEHAVIOURS:
        config = decoder[behaviour]
        rows = max(1, int(round(float(config["smoothing_sec"]) / sample_sec)))
        values = smooth_probability(probabilities[behaviour], rows)
        smoothed[behaviour] = values
        output[behaviour] = mask_runs(
            values >= float(config["threshold"]),
            frames,
            fps,
            float(config["min_duration_sec"]),
            float(config.get("max_gap_sec", 0.20)),
        )
    return output, smoothed, fps, stride

def numeric(series):
    return pd.to_numeric(series, errors="coerce")

def mode_text(series):
    values = series.dropna().astype(str)
    values = values[values.str.strip() != ""]
    return None if values.empty else str(values.mode().iloc[0])

def wrist_for_run(subset, behaviour):
    scores = {}
    for wrist in ("left", "right"):
        observed = as_boolean(subset.get(
            f"{wrist}_wrist_observed", pd.Series(False, index=subset.index)
        )).mean()
        in_shelf = as_boolean(subset.get(
            f"{wrist}_in_shelf", pd.Series(False, index=subset.index)
        )).mean()
        extension = numeric(subset.get(
            f"{wrist}_arm_extension_ratio", pd.Series(np.nan, index=subset.index)
        )).mean()
        extension = 0.0 if pd.isna(extension) else float(extension)
        if behaviour == "reach_to_shelf":
            direction = numeric(subset.get(
                f"{wrist}_wrist_toward_shelf_speed_px_sec",
                pd.Series(np.nan, index=subset.index),
            )).clip(lower=0).mean()
        elif behaviour == "retract_from_shelf":
            direction = numeric(subset.get(
                f"{wrist}_wrist_toward_body_speed_px_sec",
                pd.Series(np.nan, index=subset.index),
            )).clip(lower=0).mean()
        else:
            direction = 0.0
        direction = 0.0 if pd.isna(direction) else min(float(direction) / 150.0, 2.0)
        scores[wrist] = 1.5 * in_shelf + 0.5 * observed + 0.4 * extension + direction
    return max(scores, key=scores.get)

def event_score(smoothed, behaviour, run):
    values = smoothed[behaviour][run["start_index"]:run["end_index"] + 1]
    return float(np.nanmean(values)) if len(values) else 0.0

def physical_event(behaviour, run, features, smoothed, fps, wrist=None):
    subset = features.iloc[run["start_index"]:run["end_index"] + 1]
    wrist = wrist or wrist_for_run(subset, behaviour)
    shelf = mode_text(subset.get(
        f"{wrist}_shelf_zone", pd.Series(dtype=object)
    )) or mode_text(subset.get(
        f"{wrist}_wrist_to_shelf_nearest_zone", pd.Series(dtype=object)
    ))
    products = []
    for column in (
        f"{wrist}_single_point_product_zone",
        f"{wrist}_product_zone",
        f"{wrist}_corridor_product_zone",
    ):
        if column in subset.columns:
            products.extend(subset[column].dropna().astype(str).tolist())
    product_text = "|".join(sorted(set(products))) or None
    return make_event(
        behaviour,
        run,
        fps,
        wrist=wrist,
        shelf_zone=shelf,
        product_zones=product_text,
        rule_score=event_score(smoothed, behaviour, run),
        evidence="hybrid_v2 two-stage model with causal smoothing",
    )

def nearest_event(events, frame, direction, max_gap_frames):
    candidates = []
    for event_index, event in enumerate(events):
        if direction == "before":
            gap = frame - event["end_frame"]
        else:
            gap = event["start_frame"] - frame
        if -max_gap_frames // 4 <= gap <= max_gap_frames:
            candidates.append((
                abs(gap), -float(event["rule_score"]), event_index, event
            ))
    return min(candidates, default=(None, None, None, None))[3]

def fallback_motion_run(features, wrist, start_index, end_index, direction, fps):
    if start_index > end_index:
        return None
    subset = features.iloc[start_index:end_index + 1]
    column = (
        f"{wrist}_wrist_toward_shelf_speed_px_sec"
        if direction == "reach" else f"{wrist}_wrist_toward_body_speed_px_sec"
    )
    if column not in subset.columns:
        return None
    motion = numeric(subset[column]).fillna(0).to_numpy() >= 25.0
    runs = mask_runs(
        motion, subset["frame"].to_numpy(), fps, min_duration_sec=0.13, max_gap_sec=0.13
    )
    if not runs:
        return None
    selected = runs[-1] if direction == "reach" else runs[0]
    return {
        **selected,
        "start_index": start_index + selected["start_index"],
        "end_index": start_index + selected["end_index"],
    }

def event_overlap_frames(first, second):
    return max(
        0,
        min(int(first["end_frame"]), int(second["end_frame"]))
        - max(int(first["start_frame"]), int(second["start_frame"])) + 1,
    )

def physical_contact_for_event(event, contacts, relation, max_gap_frames):
    candidates = []
    for contact_index, contact in enumerate(contacts):
        overlap = event_overlap_frames(event, contact)
        if overlap:
            gap = 0
        elif relation == "after":
            gap = int(contact["start_frame"]) - int(event["end_frame"])
        elif relation == "before":
            gap = int(event["start_frame"]) - int(contact["end_frame"])
        else:
            gap = min(
                abs(int(contact["start_frame"]) - int(event["end_frame"])),
                abs(int(event["start_frame"]) - int(contact["end_frame"])),
            )
        if gap < 0 or gap > max_gap_frames:
            continue
        wrist_mismatch = int(contact.get("wrist") != event.get("wrist"))
        candidates.append((
            gap,
            wrist_mismatch,
            -overlap,
            -float(contact.get("rule_score", 0.0)),
            contact_index,
            contact,
        ))
    return min(candidates, default=(None, None, None, None, None, None))[-1]

def apply_physical_contact_origin(
    event, contact, features, zones, calibration, keep_episode=True
):
    if contact is None:
        return event
    subset = features.iloc[
        int(contact["_start_index"]):int(contact["_end_index"]) + 1
    ]
    origin = combined_origin_decision(
        subset, contact["wrist"], zones, calibration
    )
    event.update({
        "wrist": contact["wrist"],
        "shelf_zone": contact.get("shelf_zone") or event.get("shelf_zone"),
        "product_zones": contact.get("product_zones") or event.get("product_zones"),
        "product_origin_zone": origin["zone"],
        "product_origin_zone_candidate": origin["candidate"],
        "product_origin_zone_alternative": origin["alternative"],
        "product_origin_ambiguous": origin["ambiguous"],
        "product_origin_confidence": origin["confidence"],
        "product_origin_margin": origin["margin"],
        "product_origin_method": origin["method"],
        "product_origin_evidence": (
            "nearest_original_physical_contact; " + str(origin["evidence"])
        ),
    })
    if not keep_episode:
        event["source_hand_episode"] = None
    return event

def build_sequence_events(features, runs, smoothed, fps, stride, zones, calibration):
    physical_contacts = detect_hand_episodes(features, fps)
    hand_events = []
    for episode, run in enumerate(runs["hand_in_shelf"], start=1):
        event = physical_event("hand_in_shelf", run, features, smoothed, fps)
        contact = physical_contact_for_event(
            event, physical_contacts, "overlap", int(round(0.40 * fps))
        )
        if contact is not None:
            apply_physical_contact_origin(
                event, contact, features, zones, calibration
            )
        else:
            subset = features.iloc[run["start_index"]:run["end_index"] + 1]
            origin = combined_origin_decision(
                subset, event["wrist"], zones, calibration
            )
            event.update({
                "product_origin_zone": origin["zone"],
                "product_origin_zone_candidate": origin["candidate"],
                "product_origin_zone_alternative": origin["alternative"],
                "product_origin_ambiguous": origin["ambiguous"],
                "product_origin_confidence": origin["confidence"],
                "product_origin_margin": origin["margin"],
                "product_origin_method": origin["method"],
                "product_origin_evidence": origin["evidence"],
            })
        event["source_hand_episode"] = episode
        hand_events.append(event)

    reach_candidates = [
        physical_event("reach_to_shelf", run, features, smoothed, fps)
        for run in runs["reach_to_shelf"]
    ]
    retract_candidates = [
        physical_event("retract_from_shelf", run, features, smoothed, fps)
        for run in runs["retract_from_shelf"]
    ]
    max_gap = int(round(1.20 * fps))
    linked_reaches = []
    linked_retracts = []
    used_reaches = set()
    used_retracts = set()
    for hand in hand_events:
        reach = nearest_event(reach_candidates, hand["start_frame"], "before", max_gap)
        if reach is not None and id(reach) not in used_reaches:
            used_reaches.add(id(reach))
        else:
            start = max(0, hand["_start_index"] - int(np.ceil(1.20 * fps / stride)))
            fallback = fallback_motion_run(
                features, hand["wrist"], start, hand["_start_index"] - 1, "reach", fps
            )
            reach = None if fallback is None else physical_event(
                "reach_to_shelf", fallback, features, smoothed, fps, hand["wrist"]
            )
            if reach is not None:
                reach["rule_score"] = max(float(reach["rule_score"]), 0.25)
                reach["evidence"] = "model-supported forearm approach paired to shelf entry"
        if reach is not None:
            reach.update({
                "source_hand_episode": hand["source_hand_episode"],
                "wrist": hand["wrist"],
                "shelf_zone": hand["shelf_zone"],
                **{key: hand.get(key) for key in (
                    "product_origin_zone", "product_origin_zone_candidate",
                    "product_origin_zone_alternative", "product_origin_ambiguous",
                    "product_origin_confidence", "product_origin_margin",
                    "product_origin_method", "product_origin_evidence",
                )},
            })
            linked_reaches.append(reach)

        retract = nearest_event(retract_candidates, hand["end_frame"], "after", max_gap)
        if retract is not None and id(retract) not in used_retracts:
            used_retracts.add(id(retract))
        else:
            end = min(
                len(features) - 1,
                hand["_end_index"] + int(np.ceil(1.20 * fps / stride)),
            )
            fallback = fallback_motion_run(
                features, hand["wrist"], hand["_end_index"] + 1, end, "retract", fps
            )
            retract = None if fallback is None else physical_event(
                "retract_from_shelf", fallback, features, smoothed, fps, hand["wrist"]
            )
            if retract is not None:
                retract["rule_score"] = max(float(retract["rule_score"]), 0.25)
                retract["evidence"] = "model-supported movement to torso paired to shelf exit"
        if retract is not None:
            retract.update({
                "source_hand_episode": hand["source_hand_episode"],
                "wrist": hand["wrist"],
                "shelf_zone": hand["shelf_zone"],
                "held_product_origin_candidate": hand["product_origin_zone_candidate"],
                **{key: hand.get(key) for key in (
                    "product_origin_zone", "product_origin_zone_candidate",
                    "product_origin_zone_alternative", "product_origin_ambiguous",
                    "product_origin_confidence", "product_origin_margin",
                    "product_origin_method", "product_origin_evidence",
                )},
            })
            linked_retracts.append(retract)

    for candidate in reach_candidates:  # Keep strong events not duplicated by a physical link.
        if id(candidate) not in used_reaches and float(candidate["rule_score"]) >= 0.45:
            contact = physical_contact_for_event(
                candidate, physical_contacts, "after", max_gap
            )
            apply_physical_contact_origin(
                candidate, contact, features, zones, calibration,
                keep_episode=False,
            )
            linked_reaches.append(candidate)
    for candidate in retract_candidates:
        if id(candidate) not in used_retracts and float(candidate["rule_score"]) >= 0.45:
            contact = physical_contact_for_event(
                candidate, physical_contacts, "before", max_gap
            )
            apply_physical_contact_origin(
                candidate, contact, features, zones, calibration,
                keep_episode=False,
            )
            if candidate.get("product_origin_zone_candidate"):
                candidate["held_product_origin_candidate"] = candidate[
                    "product_origin_zone_candidate"
                ]
            linked_retracts.append(candidate)
    return hand_events, linked_reaches, linked_retracts

def combined_origin_decision(subset, wrist, zones, calibration):
    """Keep the proven fixed-corridor vote as the label decision.

    The forearm-relative method is retained as an independent audit. It can
    make a primary decision ambiguous when it provides strong contradictory
    evidence, but it never replaces the primary product-zone label.
    """
    primary = contact_origin_decision(subset, wrist)
    adaptive = origin_decision(subset, wrist, zones, calibration)
    primary_candidate = primary.get("candidate")
    adaptive_candidate = adaptive.get("candidate")
    adaptive_confidence = adaptive.get("confidence")
    adaptive_margin = adaptive.get("margin")
    strong_disagreement = bool(
        primary_candidate
        and adaptive_candidate
        and adaptive_candidate != primary_candidate
        and not adaptive.get("ambiguous", True)
        and adaptive_confidence is not None
        and float(adaptive_confidence) >= 0.45
        and adaptive_margin is not None
        and float(adaptive_margin) >= 0.20
    )
    alternative = primary.get("alternative")
    if alternative is None and strong_disagreement:
        alternative = adaptive_candidate
    primary_evidence = primary.get("evidence") or "none"
    adaptive_evidence = adaptive.get("evidence") or "none"
    return {
        "zone": primary.get("zone"),
        "candidate": primary_candidate,
        "alternative": alternative,
        "ambiguous": bool(primary.get("ambiguous", False) or strong_disagreement),
        "confidence": primary.get("confidence"),
        "margin": primary.get("margin"),
        "method": "hybrid_pre_exit_corridor_vote_with_adaptive_ambiguity_audit",
        "evidence": (
            f"primary=[{primary_evidence}]; "
            f"adaptive_candidate={adaptive_candidate}; "
            f"adaptive_ambiguous={bool(adaptive.get('ambiguous', True))}; "
            f"adaptive_confidence={adaptive_confidence}; "
            f"adaptive_margin={adaptive_margin}; "
            f"adaptive_strong_disagreement={strong_disagreement}; "
            f"adaptive=[{adaptive_evidence}]"
        ),
    }

def emergence_confirmed(emergence, wrist, start_frame, end_frame, retract_frame):
    if emergence is None:
        return False
    window = emergence[
        (emergence["frame"] >= max(0, retract_frame))
        & (emergence["frame"] <= end_frame)
    ]
    if window.empty:
        return False
    wrists = [wrist, "right" if wrist == "left" else "left"]  # Confirmation may transfer hands.
    columns = [
        f"{name}_held_product_detected"
        for name in wrists
        if f"{name}_held_product_detected" in window.columns
    ]
    if "held_product_detected" in window.columns:
        columns.append("held_product_detected")
    return any(bool(as_boolean(window[column]).any()) for column in columns)

def inspect_product_events(features, runs, smoothed, fps, retracts, emergence, final_stage):
    events = []
    max_gap = int(round(7.0 * fps))
    for run in runs["inspect_product"]:
        subset = features.iloc[run["start_index"]:run["end_index"] + 1]
        recent = nearest_event(retracts, run["start_frame"], "before", max_gap)
        if recent is None:
            continue
        presentation = as_boolean(subset.get(
            "product_presentation_pose", pd.Series(False, index=subset.index)
        )).mean()
        near_body = as_boolean(subset.get(
            "hands_near_body", pd.Series(False, index=subset.index)
        )).mean()
        attention = (
            subset.get("attention_target", pd.Series("", index=subset.index))
            .fillna("").astype(str).str.lower().isin({"hands", "held_product"}).mean()
        )
        speed_columns = [
            column for column in ("left_wrist_speed_px_sec", "right_wrist_speed_px_sec")
            if column in subset.columns
        ]
        stationary = 0.0
        if speed_columns:
            speed = pd.concat([numeric(subset[column]) for column in speed_columns], axis=1).min(axis=1)
            stationary = float((speed <= 55.0).mean())
        supporting = sum(value >= 0.35 for value in (presentation, near_body, attention, stationary))
        if supporting < 2:
            continue
        confirmed = emergence_confirmed(
            emergence, recent["wrist"], run["start_frame"], run["end_frame"], recent["end_frame"]
        )
        if final_stage and not confirmed:
            continue
        event = physical_event(
            "inspect_product", run, features, smoothed, fps, recent["wrist"]
        )
        event.update({
            "source_hand_episode": recent.get("source_hand_episode"),
            "shelf_zone": recent.get("shelf_zone"),
            "product_origin_zone": recent.get("product_origin_zone"),
            "product_origin_zone_candidate": recent.get("product_origin_zone_candidate"),
            "product_origin_zone_alternative": recent.get("product_origin_zone_alternative"),
            "product_origin_ambiguous": recent.get("product_origin_ambiguous", False),
            "product_origin_confidence": recent.get("product_origin_confidence"),
            "product_origin_margin": recent.get("product_origin_margin"),
            "product_origin_method": recent.get("product_origin_method"),
            "product_origin_evidence": recent.get("product_origin_evidence"),
            "held_product_origin_candidate": recent.get("product_origin_zone_candidate"),
            "held_product_detection_confirmed": confirmed,
            "presentation_pose_share": round(float(presentation), 3),
            "attention_to_hands_share": round(float(attention), 3),
            "evidence": (
                "two-stage inspect-product model; follows retraction; hands/arms/"
                "attention/stationarity agree; "
                + ("emergence confirmed" if confirmed else "awaiting emergence confirmation")
            ),
        })
        events.append(event)
    return events

def inspect_shelf_events(features, runs, smoothed, fps, minimum_duration_sec):
    events = []
    for run in runs["inspect_shelf"]:
        subset = features.iloc[run["start_index"]:run["end_index"] + 1]
        direct_contact = np.zeros(len(subset), dtype=bool)
        for column in (
            "left_in_shelf", "right_in_shelf",
            "left_in_product", "right_in_product",
        ):
            if column in subset.columns:
                direct_contact |= as_boolean(subset[column]).to_numpy()
        fragments = mask_runs(  # Keeps learned shelf attention around brief contacts.
            ~direct_contact,
            subset["frame"].to_numpy(dtype=int),
            fps,
            min_duration_sec=float(minimum_duration_sec),
            max_gap_sec=0.0,
        )
        for fragment in fragments:
            adjusted = {
                **fragment,
                "start_index": run["start_index"] + fragment["start_index"],
                "end_index": run["start_index"] + fragment["end_index"],
            }
            event = physical_event(
                "inspect_shelf", adjusted, features, smoothed, fps, wrist=None
            )
            event["wrist"] = None
            event["evidence"] = (
                "trained inspect-shelf attention proxy; direct shelf/product "
                "contact frames trimmed rather than rejecting the full event"
            )
            events.append(event)
    return events

def public_events(events):
    order = {name: index for index, name in enumerate(BEHAVIOURS)}
    events.sort(key=lambda event: (
        event["start_frame"], order[event["behaviour"]], event.get("wrist") or ""
    ))
    rows = []
    for event_id, event in enumerate(events, start=1):
        row = {column: event.get(column) for column in PUBLIC_EVENT_COLUMNS}
        row["event_id"] = event_id
        rows.append(row)
    return pd.DataFrame(rows, columns=PUBLIC_EVENT_COLUMNS)

def classify(feature_df, bundle, zones, calibration, emergence=None, final_stage=True):
    features = feature_df.sort_values("frame").reset_index(drop=True).copy()
    model_rows = build_temporal_features(features)
    probabilities = predict_probabilities(bundle, model_rows)
    runs, smoothed, fps, stride = decoded_runs(
        features, probabilities, bundle["decoder"]
    )
    features = add_smoothed_evidence(features)
    hand, reach, retract = build_sequence_events(
        features, runs, smoothed, fps, stride, zones, calibration
    )
    inspect_product = inspect_product_events(
        features, runs, smoothed, fps, retract, emergence, final_stage
    )
    inspect_shelf = inspect_shelf_events(
        features,
        runs,
        smoothed,
        fps,
        bundle["decoder"]["inspect_shelf"]["min_duration_sec"],
    )
    events = public_events([*hand, *reach, *retract, *inspect_product, *inspect_shelf])
    internal_events = events.to_dict("records")
    predictions = create_frame_predictions(features, internal_events)
    predictions["stage1_behaviour_probability"] = probabilities["event_probability"]
    for behaviour in BEHAVIOURS:
        predictions[f"{behaviour}_model_score"] = smoothed[behaviour]
    predictions["primary_model_score"] = predictions[
        [f"{behaviour}_model_score" for behaviour in BEHAVIOURS]
    ].max(axis=1)
    predictions["classifier_source"] = MODEL_VERSION
    return predictions, events, fps, stride

def output_paths(pose_run_id, stage):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{pose_run_id}_{run_id}_{stage}"
    return {
        "run_id": run_id,
        "frame": OUTPUT_DIR / f"behaviour_frame_labels_{stem}.csv",
        "events": OUTPUT_DIR / f"behaviour_events_{stem}.csv",
        "video": OUTPUT_DIR / f"behaviour_video_{stem}.mp4",
        "manifest": OUTPUT_DIR / f"behaviour_classifier_run_{stem}.json",
    }

def main():
    args = parse_args()
    if args.input_csv is None:
        input_csv, feature_manifest = find_latest_feature_csv()
    else:
        input_csv = args.input_csv.resolve()
        feature_manifest = manifest_for_feature_csv(input_csv)
    model_manifest_path = args.model_manifest.resolve()
    zones_path = args.zones_json.resolve()
    for path in (input_csv, model_manifest_path, zones_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    model_manifest = load_json(model_manifest_path)
    if model_manifest.get("pipeline_version") != MODEL_VERSION:
        raise ValueError("Model manifest is not a hybrid_v2 model")
    if model_manifest.get("test_data_used_for_training_or_tuning") is not False:
        raise ValueError("Model provenance does not exclude test-data tuning")
    if args.require_frozen and not model_manifest.get("frozen_for_test"):
        raise ValueError("Test execution requires a validation-tuned frozen model")
    model_path = Path(model_manifest["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if sha256(model_path) != model_manifest.get("model_sha256"):
        raise ValueError("Model file hash differs from the model manifest")
    bundle = joblib.load(model_path)

    pose_run_id = args.pose_run_id or (
        feature_manifest.get("pose_run_id") if feature_manifest else None
    ) or input_csv.stem.replace("behaviour_features_", "").split("_2026")[0]
    paths = output_paths(str(pose_run_id), args.stage)
    frame_output = args.frame_output_csv or paths["frame"]
    event_output = args.event_output_csv or paths["events"]
    video_output = args.video_output or paths["video"]
    origin_path = args.origin_calibration.resolve()
    calibration = load_calibration(origin_path if origin_path.is_file() else None)
    zones = load_product_zones(zones_path)
    features = pd.read_csv(input_csv, low_memory=False)
    emergence = None
    if args.emergence_features is not None:
        emergence_path = args.emergence_features.resolve()
        if not emergence_path.is_file():
            raise FileNotFoundError(emergence_path)
        emergence = pd.read_csv(emergence_path, low_memory=False)
    if args.stage == "final" and emergence is None:
        print(
            "Warning: final stage has no emergence features; inspect_product "
            "will be withheld rather than treated as confirmed."
        )
    predictions, events, fps, stride = classify(
        features, bundle, zones, calibration, emergence, args.stage == "final"
    )
    frame_output.parent.mkdir(parents=True, exist_ok=True)
    event_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(frame_output, index=False, float_format="%.4f")
    events.to_csv(event_output, index=False, float_format="%.4f")

    input_video = None if args.skip_video else args.input_video or find_pose_video(pose_run_id)
    rendered = None
    written = 0
    if not args.skip_video and input_video is not None:
        written = annotate_behaviour_video(input_video, predictions, video_output)
        rendered = video_output
    manifest = {
        "classifier_run_id": paths["run_id"],
        "classifier_type": "trained_two_stage",
        "pipeline_version": MODEL_VERSION,
        "classification_stage": args.stage,
        "pose_run_id": str(pose_run_id),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_feature_csv": str(input_csv.resolve()),
        "emergence_feature_csv": (
            None if args.emergence_features is None
            else str(args.emergence_features.resolve())
        ),
        "frame_output_csv": str(frame_output.resolve()),
        "event_output_csv": str(event_output.resolve()),
        "input_pose_video": None if input_video is None else str(Path(input_video).resolve()),
        "behaviour_video": None if rendered is None else str(rendered.resolve()),
        "video_frames_written": written,
        "fps": fps,
        "stride": stride,
        "input_rows": len(features),
        "events_generated": len(events),
        "behaviour_counts": events["behaviour"].value_counts().to_dict(),
        "model_manifest": str(model_manifest_path),
        "model_path": str(model_path.resolve()),
        "model_sha256": model_manifest["model_sha256"],
        "model_frozen_for_test": bool(model_manifest.get("frozen_for_test")),
        "origin_calibration": str(origin_path) if origin_path.is_file() else None,
        "precautions": {
            "test_data_used_for_training_or_tuning": False,
            "causal_temporal_context": True,
            "two_stage_background_control": True,
            "class_weighted_training": True,
            "validation_tuned_decoder": True,
            "sequence_linked_reach_hand_retract": True,
            "brief_gaps_merged": True,
            "one_reach_and_retract_per_hand_episode": True,
            "adaptive_forearm_origin_projection": True,
            "fixed_corridor_origin_is_primary": True,
            "adaptive_origin_used_for_ambiguity_audit_only": True,
            "unlinked_model_motion_uses_nearest_physical_contact_origin": True,
            "ambiguous_origins_explicit": True,
            "inspect_product_requires_emergence_in_final_stage": True,
            "emergence_confirmation_allows_wrist_transfer": True,
            "inspect_shelf_direct_contact_is_trimmed": True,
            "inspect_shelf_remains_attention_proxy": True,
            "model_scores_are_not_probability_calibrated": True,
        },
        "decoder": bundle["decoder"],
    }
    write_json(paths["manifest"], manifest)
    write_json(LATEST_CLASSIFIER_MANIFEST, manifest)
    print("Trained MERL behaviour classification complete.")
    print("Stage:", args.stage)
    print("Input feature CSV:", input_csv)
    print("Saved frame labels:", frame_output)
    print("Saved behaviour events:", event_output)
    if rendered is not None:
        print("Saved behaviour video:", rendered)
    print("Saved run manifest:", paths["manifest"])
    print("Events generated:", len(events))
    for behaviour in BEHAVIOURS:
        print(f"{behaviour}: {int((events['behaviour'] == behaviour).sum())}")

if __name__ == "__main__":
    main()
