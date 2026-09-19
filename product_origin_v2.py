"""Adaptive forearm-relative product-origin scoring for hybrid_v2."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_CALIBRATION = {
    "schema_version": 1,
    "method": "adaptive_forearm_pre_exit_vote",
    "projection_factors": [0.15, 0.25, 0.35, 0.50],
    "projection_weights": [1.40, 1.15, 0.90, 0.70],
    "shelf_multipliers": {},
    "shelf_adaptive_weights": {},
    "fixed_corridor_weight": 1.0,
    "pre_exit_sec": 0.80,
    "snap_distance_px": 10.0,
    "shelf_match_weight": 1.25,
    "shelf_conflict_weight": 0.75,
    "ambiguity_ratio": 0.82,
    "minimum_top_share": 0.28,
}

def load_calibration(path=None):
    calibration = dict(DEFAULT_CALIBRATION)
    if path is not None:
        with open(Path(path), "r", encoding="utf-8") as file:
            calibration.update(json.load(file))
    return calibration

def load_product_zones(path):
    with open(Path(path), "r", encoding="utf-8") as file:
        payload = json.load(file)
    zones = []
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", ""))
        if "_Product_" not in label:
            continue
        points = np.asarray(shape.get("points", []), dtype=float)
        if len(points) >= 3:
            zones.append((label, points))
    if not zones:
        raise ValueError(f"No product polygons found in {path}")
    return zones

def expected_shelf(product_label):
    match = re.match(r"^([LR])(\d+)_Product_", str(product_label or ""))
    if not match:
        return None
    side, row = match.group(1), int(match.group(2))
    if side == "L":
        if row == 1:
            return "Shelf_Left_Top"
        if row <= 5:
            return "Shelf_Middle_Left"
        return "Shelf_Left_Bottom"
    if row == 1:
        return "Shelf_Right_Top"
    if row <= 4:
        return "Shelf_Right_Middle"
    return "Shelf_Right_Bottom"

def product_zones_are_neighbours(first, second):
    pattern = r"^([LR])(\d+)_Product_(\d+)of(\d+)$"
    first_match = re.match(pattern, str(first or ""))
    second_match = re.match(pattern, str(second or ""))
    if not first_match or not second_match:
        return False
    return (
        first_match.group(1) == second_match.group(1)
        and abs(int(first_match.group(2)) - int(second_match.group(2))) <= 1
        and abs(int(first_match.group(3)) - int(second_match.group(3))) <= 1
    )

def point_in_polygon(point, polygon):
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside

def point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))

def polygon_distance(point, polygon):
    if point_in_polygon(point, polygon):
        return 0.0, True
    distances = [
        point_segment_distance(point, polygon[index - 1], polygon[index])
        for index in range(len(polygon))
    ]
    return min(distances), False

def zone_evidence(point, zones, snap_distance_px):
    evidence = {}
    for label, polygon in zones:
        if cv2 is not None:
            signed = float(cv2.pointPolygonTest(
                polygon.astype(np.float32),
                (float(point[0]), float(point[1])),
                True,
            ))
            inside = signed >= 0
            outside_distance = 0.0 if inside else -signed
        else:
            outside_distance, inside = polygon_distance(point, polygon)
        if not inside and outside_distance > snap_distance_px:
            continue
        centre = polygon.mean(axis=0)
        extent = np.ptp(polygon, axis=0)
        diagonal = max(float(np.hypot(extent[0], extent[1])), 1.0)
        centre_score = float(np.linalg.norm(np.asarray(point) - centre)) / diagonal
        if not inside:
            centre_score += 0.25 * outside_distance / diagonal
        evidence[label] = 1.0 / ((0.20 + centre_score) ** 2)
    return evidence

def reliable_rows(subset, wrist):
    observed = f"{wrist}_wrist_observed"
    held = f"{wrist}_wrist_held"
    if observed not in subset.columns:
        return subset
    trusted = subset[observed].fillna(False).astype(str).str.lower().isin({"true", "1"})
    if held in subset.columns:
        trusted &= ~subset[held].fillna(False).astype(str).str.lower().isin({"true", "1"})
    selected = subset[trusted]
    return selected if len(selected) >= 2 else subset

def contact_shelf_mode(subset, wrist):
    candidates = []
    for column in (
        f"{wrist}_single_point_shelf_zone",
        f"{wrist}_shelf_zone",
        f"{wrist}_corridor_shelf_zone",
    ):
        if column in subset.columns:
            candidates.extend(subset[column].dropna().astype(str).tolist())
    return Counter(candidates).most_common(1)[0][0] if candidates else None

def parse_corridor_candidates(value):
    if value is None or pd.isna(value):
        return {}
    output = {}
    for item in str(value).split("|"):
        if ":" not in item:
            continue
        label, score = item.rsplit(":", 1)
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if label and np.isfinite(score) and score > 0:
            output[label] = score
    return output

def origin_decision(subset, wrist, zones, calibration=None):
    calibration = {**DEFAULT_CALIBRATION, **(calibration or {})}
    if subset.empty:
        return empty_decision("no_contact_rows")
    voting = subset.sort_values("frame").copy()
    if "time_sec" in voting.columns:
        times = pd.to_numeric(voting["time_sec"], errors="coerce")
        if times.notna().any():
            end_time = float(times.max())
            recent = voting[times >= end_time - float(calibration["pre_exit_sec"])]
            if not recent.empty:
                voting = recent
    voting = reliable_rows(voting, wrist)
    shelf_mode = contact_shelf_mode(voting, wrist)
    multiplier = float(calibration.get("shelf_multipliers", {}).get(shelf_mode, 1.0))
    adaptive_weight = float(
        calibration.get("shelf_adaptive_weights", {}).get(shelf_mode, 0.25)
    )
    fixed_weight = float(calibration.get("fixed_corridor_weight", 1.0))
    factors = list(calibration["projection_factors"])
    weights = list(calibration["projection_weights"])
    accumulated = defaultdict(float)
    hits = Counter()
    used_frames = 0

    for position, (_, row) in enumerate(voting.iterrows()):
        wrist_xy = np.asarray([
            pd.to_numeric(row.get(f"{wrist}_wrist_x"), errors="coerce"),
            pd.to_numeric(row.get(f"{wrist}_wrist_y"), errors="coerce"),
        ], dtype=float)
        elbow_xy = np.asarray([
            pd.to_numeric(row.get(f"{wrist}_elbow_x"), errors="coerce"),
            pd.to_numeric(row.get(f"{wrist}_elbow_y"), errors="coerce"),
        ], dtype=float)
        if not np.isfinite(wrist_xy).all() or not np.isfinite(elbow_xy).all():
            continue
        direction = wrist_xy - elbow_xy
        forearm = float(np.linalg.norm(direction))
        if forearm <= 1e-6:
            continue
        unit = direction / forearm
        frame_scores = defaultdict(float)
        fixed_column = f"{wrist}_corridor_product_candidates"
        for label, score in parse_corridor_candidates(row.get(fixed_column)).items():
            frame_scores[label] += fixed_weight * score
        if adaptive_weight > 0:
            for factor, point_weight in zip(factors, weights):
                projection = float(np.clip(forearm * factor * multiplier, 8.0, 75.0))
                point = wrist_xy + unit * projection
                for label, score in zone_evidence(
                    point, zones, float(calibration["snap_distance_px"])
                ).items():
                    expected = expected_shelf(label)
                    if shelf_mode and expected == shelf_mode:
                        score *= float(calibration["shelf_match_weight"])
                    elif shelf_mode and expected and expected != shelf_mode:
                        score *= float(calibration["shelf_conflict_weight"])
                    frame_scores[label] += adaptive_weight * float(point_weight) * score

        single_column = f"{wrist}_single_point_product_zone"
        single = row.get(single_column)
        if pd.notna(single) and str(single).strip():
            frame_scores[str(single)] += 0.50
        if not frame_scores:
            continue
        used_frames += 1
        temporal_weight = 0.60 + 0.80 * (position + 1) / max(len(voting), 1)
        for label, score in frame_scores.items():
            accumulated[label] += temporal_weight * score
            hits[label] += 1

    if not accumulated:
        fallback_column = f"{wrist}_single_point_product_zone"
        if fallback_column in voting.columns and not voting[fallback_column].dropna().empty:
            fallback = str(voting[fallback_column].dropna().mode().iloc[0])
            return {
                **empty_decision("single_point_fallback"),
                "zone": fallback,
                "candidate": fallback,
                "confidence": 1.0,
                "margin": 1.0,
                "evidence": fallback,
            }
        return empty_decision("no_adaptive_corridor_evidence")

    combined = {
        label: score + 0.15 * hits[label]
        for label, score in accumulated.items()
    }
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    alternative = ranked[1][0] if len(ranked) > 1 else None
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(combined.values())
    share = best_score / total if total else 0.0
    ratio = second_score / best_score if best_score else 0.0
    margin = 1.0 - ratio
    ambiguous = (
        share < float(calibration["minimum_top_share"])
        or (
            alternative is not None
            and product_zones_are_neighbours(best_label, alternative)
            and ratio >= float(calibration["ambiguity_ratio"])
        )
    )
    ranked_text = "|".join(f"{label}:{score:.3f}" for label, score in ranked[:5])
    return {
        "zone": best_label,
        "candidate": best_label,
        "alternative": alternative,
        "ambiguous": bool(ambiguous),
        "confidence": round(float(share), 4),
        "margin": round(float(margin), 4),
        "method": "adaptive_forearm_pre_exit_vote",
        "evidence": (
            f"pre_exit_sec={float(calibration['pre_exit_sec']):.2f}; "
            f"shelf={shelf_mode}; multiplier={multiplier:.2f}; "
            f"adaptive_weight={adaptive_weight:.2f}; "
            f"frames={used_frames}; ranked={ranked_text}"
        ),
    }

def empty_decision(method):
    return {
        "zone": None,
        "candidate": None,
        "alternative": None,
        "ambiguous": True,
        "confidence": None,
        "margin": None,
        "method": method,
        "evidence": None,
    }
