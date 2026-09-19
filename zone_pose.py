import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent #Track wrists and map them to shelf and product zones
DEFAULT_VIDEO_PATH = os.path.join(
    PROJECT_DIR,
    "Raw Dataset",
    "Videos_MERL_Shopping_Dataset",
    "raw_vid (1).mp4"
)
DEFAULT_ZONES_JSON = os.path.join(PROJECT_DIR, "frames", "zones.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "pose_tests")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_DIR, "yolov8s-pose.pt")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Track wrists and map them to shelf and product zones."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        type=Path,
        help="Input video path (optional positional form)."
    )
    parser.add_argument(
        "--video",
        dest="video_option",
        type=Path,
        default=None,
        help="Input video path. Overrides the optional positional path."
    )
    parser.add_argument(
        "--zones-json",
        type=Path,
        default=Path(DEFAULT_ZONES_JSON),
        help="LabelMe JSON containing shelf and product polygons."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for pose CSVs, video and run manifests."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(DEFAULT_MODEL_PATH),
        help="Ultralytics YOLO pose model."
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Maximum processed frames; use 0 for the complete video."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run identifier used in output filenames."
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Write CSV/manifests without rendering an annotated video."
    )
    return parser.parse_args()

def safe_run_id(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError("--run-id must contain at least one letter or number")
    return cleaned

ARGS = parse_args()
VIDEO_PATH = str(
    (ARGS.video_option or ARGS.video_path or Path(DEFAULT_VIDEO_PATH)).resolve()
)
ZONES_JSON = str(ARGS.zones_json.resolve())
OUTPUT_DIR = str(ARGS.output_dir.resolve())
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_STARTED_AT = datetime.now()
RUN_ID = safe_run_id(
    ARGS.run_id or RUN_STARTED_AT.strftime("%Y%m%d_%H%M%S")
)
OUT_CSV = os.path.join(
    OUTPUT_DIR,
    f"zone_pose_wrists_{RUN_ID}.csv"
)
OUT_EVENTS_CSV = os.path.join(
    OUTPUT_DIR,
    f"zone_pose_events_{RUN_ID}.csv"
)
OUT_VIDEO = None if ARGS.skip_video else os.path.join(
    OUTPUT_DIR, f"zone_pose_wrists_video_{RUN_ID}.mp4"
)
RUN_MANIFEST = os.path.join(OUTPUT_DIR, f"zone_pose_run_{RUN_ID}.json")
LATEST_RUN_MANIFEST = os.path.join(OUTPUT_DIR, "latest_zone_pose_run.json")
MODEL_PATH = str(ARGS.model.resolve())

CONF = ARGS.confidence
IMG_SIZE = ARGS.image_size
STRIDE = ARGS.stride
MAX_FRAMES = None if ARGS.max_frames <= 0 else ARGS.max_frames

if STRIDE < 1:
    raise ValueError("--stride must be at least 1")
if not (0.0 < CONF <= 1.0):
    raise ValueError("--confidence must be greater than 0 and at most 1")
if IMG_SIZE < 32:
    raise ValueError("--image-size must be at least 32")
for required_path, description in (
    (VIDEO_PATH, "input video"),
    (ZONES_JSON, "zones JSON"),
    (MODEL_PATH, "pose model"),
):
    if not os.path.isfile(required_path):
        raise FileNotFoundError(f"The {description} was not found: {required_path}")

KEYPOINT_CONF_THR = 0.20
ARM_KEYPOINT_CONF_THR = 0.15
FACE_KEYPOINT_CONF_THR = 0.12

HAND_EXTENSION_FOREARM_RATIO = 0.20  # Extends the wrist point towards the fingers.
HAND_EXTENSION_MIN_PX = 20.0
HAND_EXTENSION_MAX_PX = 35.0
PRODUCT_ZONE_SNAP_DISTANCE_PX = 12.0
SHELF_ZONE_SNAP_DISTANCE_PX = 6.0

HAND_CORRIDOR_START_PX = 10.0  # Samples beyond the wrist when one point misses the hand.
HAND_CORRIDOR_END_PX = 40.0
HAND_CORRIDOR_STEP_PX = 5.0
HAND_CORRIDOR_SHELF_WEIGHT = 0.30
HAND_CORRIDOR_MAX_CANDIDATES = 6
HAND_CORRIDOR_FRAME_AMBIGUITY_RATIO = 0.85
HAND_CORRIDOR_NEAR_WEIGHT = 1.35
HAND_CORRIDOR_FAR_WEIGHT = 0.65

DUPLICATE_WRIST_DISTANCE = 45  # Rejects near-identical left and right wrist points.
SWAP_SCORE_MARGIN = 35
BOX_MARGIN_RATIO = 0.15
BOX_MIN_MARGIN_PX = 35
MAX_FOREARM_LENGTH_PX = 190
MAX_FOREARM_TO_UPPER_ARM_RATIO = 2.2
MAX_WRIST_TO_SHOULDER_PX = 330

MAX_TRACKED_JUMP_PX = 120  # Large jumps must settle before the track moves.
SMOOTHING_ALPHA = 0.60
FAST_SMOOTHING_ALPHA = 0.85
FAST_MOVE_THRESHOLD_PX = 45
MISSING_HOLD_ROWS = 2
OUTLIER_HOLD_ROWS = 1
REACQUIRE_ROWS = 3
REACQUIRE_RADIUS_PX = 80
RESET_AFTER_MISSING_ROWS = 6

MIN_EVENT_DURATION_SEC = 0.20
EVENT_MAX_MISSING_ROWS = 1

NOSE = 0  # COCO pose indices
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

def rectangle_to_polygon(points):
    (x1, y1), (x2, y2) = points
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32
    )

def load_zones(labelme_json_path):
    with open(labelme_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zones = []

    for shape in data.get("shapes", []):
        label = shape["label"]
        points = shape["points"]
        shape_type = shape.get("shape_type", "polygon")

        if shape_type == "rectangle" and len(points) == 2:
            polygon = rectangle_to_polygon(points)
        elif shape_type == "polygon":
            polygon = np.array(points, dtype=np.float32)
        else:
            continue

        zones.append((label, polygon))

    return zones

def point_in_zone(point, zones):
    if point is None:
        return None

    x, y = point

    for label, polygon in zones:
        if cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False
        ) >= 0:
            return label

    return None

def projected_hand_point(wrist, elbow):
    """Approximate the hand/finger contact point beyond the wrist."""
    if wrist is None:
        return None
    if elbow is None:
        return wrist

    direction = np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float)
    forearm_length = float(np.linalg.norm(direction))
    if forearm_length <= 1e-6:
        return wrist

    extension = float(np.clip(
        forearm_length * HAND_EXTENSION_FOREARM_RATIO,
        HAND_EXTENSION_MIN_PX,
        HAND_EXTENSION_MAX_PX
    ))
    hand = np.asarray(wrist, dtype=float) + direction / forearm_length * extension
    return float(hand[0]), float(hand[1])

def best_zone_match(point, zones, snap_distance_px=0.0):
    """Return the best nearby zone instead of the first overlapping zone."""
    if point is None:
        return None, None

    point_xy = (float(point[0]), float(point[1]))
    best_label = None
    best_score = float("inf")

    for label, polygon in zones:
        signed_distance = float(cv2.pointPolygonTest(
            polygon,
            point_xy,
            True
        ))
        if signed_distance < -snap_distance_px:
            continue

        centre = np.mean(polygon, axis=0)
        extent = np.ptp(polygon, axis=0)
        diagonal = max(float(np.hypot(extent[0], extent[1])), 1.0)
        centre_distance = float(np.linalg.norm(
            np.asarray(point_xy) - centre
        ))
        score = centre_distance / diagonal
        if signed_distance < 0:
            score += 0.25 * (-signed_distance / diagonal)  # Keeps small boundary offsets eligible.

        if score < best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None, None
    return best_label, best_score

def expected_shelf_for_product(product_label):
    """Map MERL product-row labels to their containing shelf region."""
    match = re.match(r"^([LR])(\d+)_Product_", str(product_label or ""))
    if not match:
        return None

    side = match.group(1)
    row = int(match.group(2))
    if side == "L":
        if row == 1:
            level = "Top"
        elif row <= 5:
            level = "Middle_Left"
        else:
            level = "Left_Bottom"
        return "Shelf_Left_Top" if level == "Top" else (
            "Shelf_Middle_Left" if level == "Middle_Left"
            else "Shelf_Left_Bottom"
        )

    if row == 1:
        return "Shelf_Right_Top"
    if row <= 4:
        return "Shelf_Right_Middle"
    return "Shelf_Right_Bottom"

def nearby_zone_evidence(point, zones, snap_distance_px):
    """Return positive evidence for every zone touched by one sample point."""
    if point is None:
        return {}

    point_xy = (float(point[0]), float(point[1]))
    evidence = {}
    for label, polygon in zones:
        signed_distance = float(cv2.pointPolygonTest(
            polygon, point_xy, True
        ))
        if signed_distance < -snap_distance_px:
            continue

        centre = np.mean(polygon, axis=0)
        extent = np.ptp(polygon, axis=0)
        diagonal = max(float(np.hypot(extent[0], extent[1])), 1.0)
        centre_distance = float(np.linalg.norm(
            np.asarray(point_xy) - centre
        ))
        score = centre_distance / diagonal
        if signed_distance < 0:
            score += 0.25 * (-signed_distance / diagonal)

        evidence[label] = 1.0 / ((0.20 + score) ** 2)  # Central contact counts more than an edge touch.
    return evidence

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
    first_row = int(first_match.group(2))
    second_row = int(second_match.group(2))
    first_slot = int(first_match.group(3))
    second_slot = int(second_match.group(3))
    return (
        abs(first_row - second_row) <= 1
        and abs(first_slot - second_slot) <= 1
    )

def encode_corridor_candidates(ranked_candidates):
    return "|".join(
        f"{label}:{score:.6f}"
        for label, score in ranked_candidates[:HAND_CORRIDOR_MAX_CANDIDATES]
    ) or None

def hand_corridor_evidence(wrist, elbow, product_zones, shelf_zones):
    """Score product zones along a 10-40 px forearm-directed corridor."""
    empty = {
        "product_zone": None,
        "product_score": None,
        "product_margin": None,
        "product_alternative": None,
        "product_ambiguous": False,
        "product_candidates": None,
        "shelf_zone": None,
        "raw_shelf_zone": None,
        "start_point": None,
        "end_point": None,
    }
    if wrist is None or elbow is None:
        return empty

    direction = np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float)
    forearm_length = float(np.linalg.norm(direction))
    if forearm_length <= 1e-6:
        return empty
    unit_direction = direction / forearm_length

    product_evidence = Counter()
    shelf_evidence = Counter()
    distances = np.arange(
        HAND_CORRIDOR_START_PX,
        HAND_CORRIDOR_END_PX + HAND_CORRIDOR_STEP_PX * 0.5,
        HAND_CORRIDOR_STEP_PX,
    )
    for distance_px in distances:
        point = np.asarray(wrist, dtype=float) + unit_direction * distance_px
        depth_fraction = (
            (distance_px - HAND_CORRIDOR_START_PX)
            / max(HAND_CORRIDOR_END_PX - HAND_CORRIDOR_START_PX, 1.0)
        )
        depth_weight = (  # Near-wrist points count more.
            HAND_CORRIDOR_NEAR_WEIGHT * (1.0 - depth_fraction)
            + HAND_CORRIDOR_FAR_WEIGHT * depth_fraction
        )
        for label, score in nearby_zone_evidence(
            point, product_zones, PRODUCT_ZONE_SNAP_DISTANCE_PX
        ).items():
            product_evidence[label] += depth_weight * score
        for label, score in nearby_zone_evidence(
            point, shelf_zones, SHELF_ZONE_SNAP_DISTANCE_PX
        ).items():
            shelf_evidence[label] += depth_weight * score

    start_point = tuple(
        np.asarray(wrist, dtype=float)
        + unit_direction * HAND_CORRIDOR_START_PX
    )
    end_point = tuple(
        np.asarray(wrist, dtype=float)
        + unit_direction * HAND_CORRIDOR_END_PX
    )
    raw_shelf_zone = (
        None
        if not shelf_evidence
        else shelf_evidence.most_common(1)[0][0]
    )
    if not product_evidence:
        return {
            **empty,
            "shelf_zone": raw_shelf_zone,
            "raw_shelf_zone": raw_shelf_zone,
            "start_point": start_point,
            "end_point": end_point,
        }

    maximum_shelf_evidence = max(shelf_evidence.values(), default=0.0)
    joint_evidence = {}
    for product_label, product_score in product_evidence.items():
        expected_shelf = expected_shelf_for_product(product_label)
        shelf_support = (
            shelf_evidence.get(expected_shelf, 0.0) / maximum_shelf_evidence
            if maximum_shelf_evidence > 0 and expected_shelf is not None
            else 0.0
        )
        joint_evidence[product_label] = product_score * (
            (1.0 - HAND_CORRIDOR_SHELF_WEIGHT)
            + HAND_CORRIDOR_SHELF_WEIGHT * shelf_support
        )

    total_evidence = sum(joint_evidence.values())
    ranked = sorted(
        (
            (label, score / total_evidence)
            for label, score in joint_evidence.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    best_label, best_score = ranked[0]
    alternative_label = ranked[1][0] if len(ranked) > 1 else None
    alternative_score = ranked[1][1] if len(ranked) > 1 else 0.0
    relative_margin = (
        (best_score - alternative_score) / best_score
        if best_score > 0 else 0.0
    )
    ambiguous = bool(
        alternative_label
        and product_zones_are_neighbours(best_label, alternative_label)
        and alternative_score / best_score
        >= HAND_CORRIDOR_FRAME_AMBIGUITY_RATIO
    )
    joint_shelf_zone = expected_shelf_for_product(best_label) or raw_shelf_zone
    return {
        "product_zone": best_label,
        "product_score": best_score,
        "product_margin": relative_margin,
        "product_alternative": alternative_label,
        "product_ambiguous": ambiguous,
        "product_candidates": encode_corridor_candidates(ranked),
        "shelf_zone": joint_shelf_zone,
        "raw_shelf_zone": raw_shelf_zone,
        "start_point": start_point,
        "end_point": end_point,
    }

def split_shelf_and_product_zones(zones):
    shelf_zones = [
        (label, polygon)
        for label, polygon in zones
        if label.startswith("Shelf_")
    ]
    product_zones = [
        (label, polygon)
        for label, polygon in zones
        if not label.startswith("Shelf_")
    ]
    return shelf_zones, product_zones

def distance(point_1, point_2):
    if point_1 is None or point_2 is None:
        return None

    return float(np.hypot(
        point_1[0] - point_2[0],
        point_1[1] - point_2[1]
    ))

def midpoint(point_1, point_2):
    if point_1 is None or point_2 is None:
        return None

    return (
        (point_1[0] + point_2[0]) / 2.0,
        (point_1[1] + point_2[1]) / 2.0
    )

def safe_keypoint_xy(kpts_xy, kpts_conf, idx, conf_thr):
    """Return a keypoint only when it is present and confident enough."""
    if kpts_xy is None or idx >= len(kpts_xy):
        return None

    if kpts_conf is not None and kpts_conf[idx] < conf_thr:
        return None

    x, y = kpts_xy[idx]
    return float(x), float(y)

def safe_keypoint_conf(kpts_conf, idx):
    if kpts_conf is None or idx >= len(kpts_conf):
        return None
    return float(kpts_conf[idx])

def point_inside_expanded_box(point, box_xyxy):
    """Reject pose keypoints that are implausibly far outside the person box."""
    if point is None or box_xyxy is None:
        return True

    x1, y1, x2, y2 = box_xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    x_margin = max(BOX_MIN_MARGIN_PX, box_width * BOX_MARGIN_RATIO)
    y_margin = max(BOX_MIN_MARGIN_PX, box_height * BOX_MARGIN_RATIO)
    x, y = point

    return (
        x1 - x_margin <= x <= x2 + x_margin and
        y1 - y_margin <= y <= y2 + y_margin
    )

def arm_assignment_cost(wrist, elbow, shoulder, previous_wrist, confidence):
    """Lower values mean that the wrist fits this arm more plausibly."""
    if wrist is None:
        return float("inf")

    cost = 0.0

    if elbow is not None:
        cost += distance(wrist, elbow) * 1.5
    else:
        cost += 100

    if shoulder is not None:
        cost += distance(wrist, shoulder) * 0.30

    if previous_wrist is not None:
        cost += min(distance(wrist, previous_wrist), 250) * 0.20

    if confidence is not None:
        cost -= min(max(confidence, 0.0), 1.0) * 20

    return cost

def wrist_geometry_note(wrist, elbow, shoulder, person_box):
    """Return None for a plausible wrist, otherwise the rejection reason."""
    if wrist is None:
        return None

    if not point_inside_expanded_box(wrist, person_box):
        return "outside_person_box"

    wrist_to_elbow = distance(wrist, elbow)
    upper_arm = distance(elbow, shoulder)

    if wrist_to_elbow is not None:
        forearm_limit = MAX_FOREARM_LENGTH_PX

        if upper_arm is not None:
            forearm_limit = min(
                MAX_FOREARM_LENGTH_PX,
                max(130, upper_arm * MAX_FOREARM_TO_UPPER_ARM_RATIO)
            )

        if wrist_to_elbow > forearm_limit:
            return "forearm_too_long"

    wrist_to_shoulder = distance(wrist, shoulder)

    if (
        wrist_to_shoulder is not None and
        wrist_to_shoulder > MAX_WRIST_TO_SHOULDER_PX
    ):
        return "too_far_from_shoulder"

    return None

def assign_wrist_candidates(
    left_raw,
    right_raw,
    left_conf,
    right_conf,
    left_elbow,
    right_elbow,
    left_shoulder,
    right_shoulder,
    previous_left,
    previous_right,
    person_box
):

    candidates = []

    if left_raw is not None:
        candidates.append(("left", left_raw, left_conf))

    if right_raw is not None:
        candidates.append(("right", right_raw, right_conf))

    if not candidates:
        return None, None, "no_wrist_candidates"

    arm_data = {
        "left": (left_elbow, left_shoulder, previous_left),
        "right": (right_elbow, right_shoulder, previous_right)
    }

    def cost(candidate, arm):
        _, point, confidence = candidate
        elbow, shoulder, previous = arm_data[arm]
        return arm_assignment_cost(
            point,
            elbow,
            shoulder,
            previous,
            confidence
        )

    left_assigned = None
    right_assigned = None
    assignment_note = "normal"

    if len(candidates) == 1:
        candidate = candidates[0]
        native_arm = candidate[0]
        other_arm = "right" if native_arm == "left" else "left"
        selected_arm = native_arm

        if cost(candidate, other_arm) + SWAP_SCORE_MARGIN < cost(candidate, native_arm):
            selected_arm = other_arm
            assignment_note = f"single_reassigned_to_{other_arm}"
        else:
            assignment_note = f"single_kept_as_{native_arm}"

        if selected_arm == "left":
            left_assigned = candidate[1]
        else:
            right_assigned = candidate[1]

    else:
        left_candidate, right_candidate = candidates
        candidate_separation = distance(left_candidate[1], right_candidate[1])

        if candidate_separation < DUPLICATE_WRIST_DISTANCE:
            choices = [
                (cost(left_candidate, "left"), "left", left_candidate),
                (cost(right_candidate, "left"), "left", right_candidate),
                (cost(left_candidate, "right"), "right", left_candidate),
                (cost(right_candidate, "right"), "right", right_candidate)
            ]
            _, selected_arm, selected_candidate = min(
                choices,
                key=lambda choice: choice[0]
            )

            if selected_arm == "left":
                left_assigned = selected_candidate[1]
                assignment_note = "duplicate_kept_left_only"
            else:
                right_assigned = selected_candidate[1]
                assignment_note = "duplicate_kept_right_only"
        else:
            normal_cost = (
                cost(left_candidate, "left") +
                cost(right_candidate, "right")
            )
            swapped_cost = (
                cost(right_candidate, "left") +
                cost(left_candidate, "right")
            )

            if swapped_cost + SWAP_SCORE_MARGIN < normal_cost:
                left_assigned = right_candidate[1]
                right_assigned = left_candidate[1]
                assignment_note = "swapped_by_arm_geometry"
            else:
                left_assigned = left_candidate[1]
                right_assigned = right_candidate[1]

    rejection_notes = []

    left_rejection = wrist_geometry_note(
        left_assigned,
        left_elbow,
        left_shoulder,
        person_box
    )
    if left_rejection is not None:
        left_assigned = None
        rejection_notes.append(f"left_{left_rejection}")

    right_rejection = wrist_geometry_note(
        right_assigned,
        right_elbow,
        right_shoulder,
        person_box
    )
    if right_rejection is not None:
        right_assigned = None
        rejection_notes.append(f"right_{right_rejection}")

    if rejection_notes:
        assignment_note += ";" + ";".join(rejection_notes)

    return left_assigned, right_assigned, assignment_note

class WristTracker:

    def __init__(self):
        self.point = None
        self.missing_rows = 0
        self.outlier_rows = 0
        self.pending_point = None
        self.pending_rows = 0

    @property
    def reference_point(self):
        if self.point is None:
            return None

        if self.missing_rows > MISSING_HOLD_ROWS + 1:
            return None

        if self.outlier_rows > OUTLIER_HOLD_ROWS + 1:
            return None

        return self.point

    def _clear_pending(self):
        self.pending_point = None
        self.pending_rows = 0
        self.outlier_rows = 0

    def update(self, candidate):
        if candidate is None:
            self.missing_rows += 1
            self._clear_pending()

            if self.point is not None and self.missing_rows <= MISSING_HOLD_ROWS:
                return self.point, "held_missing"

            if self.missing_rows >= RESET_AFTER_MISSING_ROWS:
                self.point = None

            return None, "missing"

        self.missing_rows = 0

        if self.point is None:
            self.point = candidate
            self._clear_pending()
            return self.point, "acquired"

        movement = distance(candidate, self.point)

        if movement <= MAX_TRACKED_JUMP_PX:
            alpha = (
                FAST_SMOOTHING_ALPHA
                if movement >= FAST_MOVE_THRESHOLD_PX
                else SMOOTHING_ALPHA
            )
            self.point = (
                alpha * candidate[0] + (1 - alpha) * self.point[0],
                alpha * candidate[1] + (1 - alpha) * self.point[1]
            )
            self._clear_pending()
            return self.point, "tracked"

        self.outlier_rows += 1

        if (
            self.pending_point is not None and
            distance(candidate, self.pending_point) <= REACQUIRE_RADIUS_PX
        ):
            self.pending_rows += 1
            self.pending_point = candidate
        else:
            self.pending_point = candidate
            self.pending_rows = 1

        if self.pending_rows >= REACQUIRE_ROWS:
            self.point = candidate
            self._clear_pending()
            return self.point, "reacquired"

        if self.outlier_rows <= OUTLIER_HOLD_ROWS:
            return self.point, "held_outlier"

        return None, "waiting_to_reacquire"

def generate_zone_events(
    frame_df,
    fps,
    stride,
    min_duration_sec=0.20,
    max_missing_rows=1
):
    """Combine stable frame-level wrist zones into continuous events."""
    event_columns = [
        "event_id",
        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",
        "wrist",
        "shelf_zone",
        "product_zone",
        "duration_sec",
        "observations"
    ]

    if frame_df.empty:
        return pd.DataFrame(columns=event_columns)

    max_frame_gap = stride * (max_missing_rows + 1)
    events = []

    for wrist in ("left", "right"):
        shelf_col = f"{wrist}_shelf_zone"
        product_col = f"{wrist}_product_zone"
        valid_rows = frame_df.loc[
            frame_df[shelf_col].notna() & frame_df[product_col].notna(),
            ["frame", shelf_col, product_col]
        ].sort_values("frame")
        current_event = None

        for frame, shelf_zone, product_zone in valid_rows.itertuples(
            index=False,
            name=None
        ):
            frame = int(frame)
            continues_event = (
                current_event is not None and
                shelf_zone == current_event["shelf_zone"] and
                product_zone == current_event["product_zone"] and
                frame - current_event["end_frame"] <= max_frame_gap
            )

            if continues_event:
                current_event["end_frame"] = frame
                current_event["observations"] += 1
                continue

            if current_event is not None:
                events.append(current_event)

            current_event = {
                "start_frame": frame,
                "end_frame": frame,
                "wrist": wrist,
                "shelf_zone": shelf_zone,
                "product_zone": product_zone,
                "observations": 1
            }

        if current_event is not None:
            events.append(current_event)

    completed_events = []

    for event in events:
        duration_sec = (event["end_frame"] - event["start_frame"]) / fps

        if duration_sec + 1e-9 < min_duration_sec:
            continue

        event["start_time_sec"] = event["start_frame"] / fps
        event["end_time_sec"] = event["end_frame"] / fps
        event["duration_sec"] = duration_sec
        completed_events.append(event)

    completed_events.sort(
        key=lambda event: (event["start_frame"], event["wrist"])
    )

    for event_id, event in enumerate(completed_events, start=1):
        event["event_id"] = event_id

    return pd.DataFrame(completed_events, columns=event_columns)

def draw_wrist(
    vis,
    wrist,
    shelf_zone,
    product_zone,
    side,
    note,
    corridor=None,
):
    if wrist is None:
        return

    colour = (0, 0, 255) if side == "left" else (255, 0, 0)
    label = "L" if side == "left" else "R"
    x, y = int(wrist[0]), int(wrist[1])

    if note.startswith("held"):
        cv2.circle(vis, (x, y), 7, colour, 2)
    else:
        cv2.circle(vis, (x, y), 6, colour, -1)

    text = (
        f"{label}: {shelf_zone or 'None'} | "
        f"{product_zone or 'None'}"
    )
    if note not in ("tracked", "acquired"):
        text += f" [{note}]"

    cv2.putText(
        vis,
        text,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colour,
        2,
        cv2.LINE_AA
    )
    if corridor and corridor.get("product_zone"):
        corridor_text = f"C: {corridor['product_zone']}"
        if corridor.get("product_ambiguous"):
            corridor_text += (
                f" ? {corridor.get('product_alternative') or 'neighbour'}"
            )
        cv2.putText(
            vis,
            corridor_text,
            (x + 8, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

zones = load_zones(ZONES_JSON)
shelf_zones, product_zones = split_shelf_and_product_zones(zones)
print(
    f"Loaded {len(shelf_zones)} shelf zones and "
    f"{len(product_zones)} product zones"
)

print("Loading pose model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = None
if OUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUT_VIDEO,
        fourcc,
        fps / STRIDE,
        (width, height)
    )

left_tracker = WristTracker()
right_tracker = WristTracker()
rows = []
stats = Counter()
frame_idx = -1
processed_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_idx += 1

    if frame_idx % STRIDE != 0:
        continue

    processed_count += 1
    if MAX_FRAMES is not None and processed_count > MAX_FRAMES:
        break

    results = model.predict(
        frame,
        conf=CONF,
        imgsz=IMG_SIZE,
        verbose=False
    )
    result = results[0]
    vis = frame.copy()

    person_box = None
    nose = None
    left_eye = None
    right_eye = None
    left_ear = None
    right_ear = None
    nose_conf = None
    left_eye_conf = None
    right_eye_conf = None
    left_ear_conf = None
    right_ear_conf = None
    left_shoulder = None
    right_shoulder = None
    left_elbow = None
    right_elbow = None
    left_hip = None
    right_hip = None
    left_knee = None
    right_knee = None
    left_ankle = None
    right_ankle = None
    left_raw = None
    right_raw = None
    left_conf = None
    right_conf = None
    assignment_note = "no_person"

    if result.keypoints is not None and len(result.keypoints) > 0:
        best_i = 0

        if result.boxes is not None and len(result.boxes) > 0:
            box_confs = result.boxes.conf.cpu().numpy()
            best_i = int(np.argmax(box_confs))
            person_box = tuple(
                float(value)
                for value in result.boxes.xyxy[best_i].cpu().numpy()
            )

        kpts_xy = result.keypoints.xy[best_i].cpu().numpy()
        kpts_conf = (
            result.keypoints.conf[best_i].cpu().numpy()
            if result.keypoints.conf is not None
            else None
        )

        nose = safe_keypoint_xy(
            kpts_xy, kpts_conf, NOSE, FACE_KEYPOINT_CONF_THR
        )
        left_eye = safe_keypoint_xy(
            kpts_xy, kpts_conf, L_EYE, FACE_KEYPOINT_CONF_THR
        )
        right_eye = safe_keypoint_xy(
            kpts_xy, kpts_conf, R_EYE, FACE_KEYPOINT_CONF_THR
        )
        left_ear = safe_keypoint_xy(
            kpts_xy, kpts_conf, L_EAR, FACE_KEYPOINT_CONF_THR
        )
        right_ear = safe_keypoint_xy(
            kpts_xy, kpts_conf, R_EAR, FACE_KEYPOINT_CONF_THR
        )
        nose_conf = safe_keypoint_conf(kpts_conf, NOSE)
        left_eye_conf = safe_keypoint_conf(kpts_conf, L_EYE)
        right_eye_conf = safe_keypoint_conf(kpts_conf, R_EYE)
        left_ear_conf = safe_keypoint_conf(kpts_conf, L_EAR)
        right_ear_conf = safe_keypoint_conf(kpts_conf, R_EAR)
        left_shoulder = safe_keypoint_xy(
            kpts_xy, kpts_conf, L_SHOULDER, ARM_KEYPOINT_CONF_THR
        )
        right_shoulder = safe_keypoint_xy(
            kpts_xy, kpts_conf, R_SHOULDER, ARM_KEYPOINT_CONF_THR
        )
        left_elbow = safe_keypoint_xy(
            kpts_xy, kpts_conf, L_ELBOW, ARM_KEYPOINT_CONF_THR
        )
        right_elbow = safe_keypoint_xy(
            kpts_xy, kpts_conf, R_ELBOW, ARM_KEYPOINT_CONF_THR
        )
        left_hip = safe_keypoint_xy(
            kpts_xy, kpts_conf, L_HIP, ARM_KEYPOINT_CONF_THR
        )
        right_hip = safe_keypoint_xy(
            kpts_xy, kpts_conf, R_HIP, ARM_KEYPOINT_CONF_THR
        )
        left_knee = safe_keypoint_xy(
            kpts_xy, kpts_conf, L_KNEE, ARM_KEYPOINT_CONF_THR
        )
        right_knee = safe_keypoint_xy(
            kpts_xy, kpts_conf, R_KNEE, ARM_KEYPOINT_CONF_THR
        )
        left_ankle = safe_keypoint_xy(
            kpts_xy, kpts_conf, L_ANKLE, ARM_KEYPOINT_CONF_THR
        )
        right_ankle = safe_keypoint_xy(
            kpts_xy, kpts_conf, R_ANKLE, ARM_KEYPOINT_CONF_THR
        )
        left_raw = safe_keypoint_xy(
            kpts_xy, kpts_conf, L_WRIST, KEYPOINT_CONF_THR
        )
        right_raw = safe_keypoint_xy(
            kpts_xy, kpts_conf, R_WRIST, KEYPOINT_CONF_THR
        )
        left_conf = safe_keypoint_conf(kpts_conf, L_WRIST)
        right_conf = safe_keypoint_conf(kpts_conf, R_WRIST)

        left_candidate, right_candidate, assignment_note = assign_wrist_candidates(
            left_raw,
            right_raw,
            left_conf,
            right_conf,
            left_elbow,
            right_elbow,
            left_shoulder,
            right_shoulder,
            left_tracker.reference_point,
            right_tracker.reference_point,
            person_box
        )
    else:
        left_candidate = None
        right_candidate = None

    left_wrist, left_track_note = left_tracker.update(left_candidate)
    right_wrist, right_track_note = right_tracker.update(right_candidate)
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    body_center = hip_center

    if body_center is None and person_box is not None:
        body_center = (
            (person_box[0] + person_box[2]) / 2.0,
            (person_box[1] + person_box[3]) / 2.0
        )

    left_hand = projected_hand_point(left_wrist, left_elbow)
    right_hand = projected_hand_point(right_wrist, right_elbow)
    left_corridor = hand_corridor_evidence(
        left_wrist, left_elbow, product_zones, shelf_zones
    )
    right_corridor = hand_corridor_evidence(
        right_wrist, right_elbow, product_zones, shelf_zones
    )
    left_product_zone_wrist = point_in_zone(left_wrist, product_zones)
    right_product_zone_wrist = point_in_zone(right_wrist, product_zones)
    left_product_zone, left_product_zone_score = best_zone_match(
        left_hand,
        product_zones,
        snap_distance_px=PRODUCT_ZONE_SNAP_DISTANCE_PX
    )
    right_product_zone, right_product_zone_score = best_zone_match(
        right_hand,
        product_zones,
        snap_distance_px=PRODUCT_ZONE_SNAP_DISTANCE_PX
    )
    left_shelf_zone, _ = best_zone_match(
        left_hand,
        shelf_zones,
        snap_distance_px=SHELF_ZONE_SNAP_DISTANCE_PX
    )
    right_shelf_zone, _ = best_zone_match(
        right_hand,
        shelf_zones,
        snap_distance_px=SHELF_ZONE_SNAP_DISTANCE_PX
    )

    left_zone = left_product_zone or left_shelf_zone  # Keeps the earlier combined-zone column.
    right_zone = right_product_zone or right_shelf_zone

    stats[f"assignment:{assignment_note}"] += 1
    stats[f"left_track:{left_track_note}"] += 1
    stats[f"right_track:{right_track_note}"] += 1

    rows.append({
        "frame": frame_idx,
        "time_sec": frame_idx / fps,
        "person_box_x1": None if person_box is None else person_box[0],
        "person_box_y1": None if person_box is None else person_box[1],
        "person_box_x2": None if person_box is None else person_box[2],
        "person_box_y2": None if person_box is None else person_box[3],
        "body_center_x": None if body_center is None else body_center[0],
        "body_center_y": None if body_center is None else body_center[1],
        "nose_x": None if nose is None else nose[0],
        "nose_y": None if nose is None else nose[1],
        "nose_conf": nose_conf,
        "left_eye_x": None if left_eye is None else left_eye[0],
        "left_eye_y": None if left_eye is None else left_eye[1],
        "left_eye_conf": left_eye_conf,
        "right_eye_x": None if right_eye is None else right_eye[0],
        "right_eye_y": None if right_eye is None else right_eye[1],
        "right_eye_conf": right_eye_conf,
        "left_ear_x": None if left_ear is None else left_ear[0],
        "left_ear_y": None if left_ear is None else left_ear[1],
        "left_ear_conf": left_ear_conf,
        "right_ear_x": None if right_ear is None else right_ear[0],
        "right_ear_y": None if right_ear is None else right_ear[1],
        "right_ear_conf": right_ear_conf,
        "shoulder_center_x": (
            None if shoulder_center is None else shoulder_center[0]
        ),
        "shoulder_center_y": (
            None if shoulder_center is None else shoulder_center[1]
        ),
        "left_shoulder_x": None if left_shoulder is None else left_shoulder[0],
        "left_shoulder_y": None if left_shoulder is None else left_shoulder[1],
        "right_shoulder_x": None if right_shoulder is None else right_shoulder[0],
        "right_shoulder_y": None if right_shoulder is None else right_shoulder[1],
        "left_elbow_x": None if left_elbow is None else left_elbow[0],
        "left_elbow_y": None if left_elbow is None else left_elbow[1],
        "right_elbow_x": None if right_elbow is None else right_elbow[0],
        "right_elbow_y": None if right_elbow is None else right_elbow[1],
        "left_hip_x": None if left_hip is None else left_hip[0],
        "left_hip_y": None if left_hip is None else left_hip[1],
        "right_hip_x": None if right_hip is None else right_hip[0],
        "right_hip_y": None if right_hip is None else right_hip[1],
        "left_knee_x": None if left_knee is None else left_knee[0],
        "left_knee_y": None if left_knee is None else left_knee[1],
        "right_knee_x": None if right_knee is None else right_knee[0],
        "right_knee_y": None if right_knee is None else right_knee[1],
        "left_ankle_x": None if left_ankle is None else left_ankle[0],
        "left_ankle_y": None if left_ankle is None else left_ankle[1],
        "right_ankle_x": None if right_ankle is None else right_ankle[0],
        "right_ankle_y": None if right_ankle is None else right_ankle[1],
        "left_wrist_x_raw": None if left_raw is None else left_raw[0],
        "left_wrist_y_raw": None if left_raw is None else left_raw[1],
        "left_wrist_conf": left_conf,
        "left_wrist_x": None if left_wrist is None else left_wrist[0],
        "left_wrist_y": None if left_wrist is None else left_wrist[1],
        "left_hand_x": None if left_hand is None else left_hand[0],
        "left_hand_y": None if left_hand is None else left_hand[1],
        "left_wrist_zone": left_zone,
        "left_shelf_zone": left_shelf_zone,
        "left_product_zone": left_product_zone,
        "left_product_zone_wrist_only": left_product_zone_wrist,
        "left_product_zone_score": left_product_zone_score,
        "left_corridor_product_zone": left_corridor["product_zone"],
        "left_corridor_product_score": left_corridor["product_score"],
        "left_corridor_product_margin": left_corridor["product_margin"],
        "left_corridor_product_alternative": left_corridor[
            "product_alternative"
        ],
        "left_corridor_product_ambiguous": left_corridor[
            "product_ambiguous"
        ],
        "left_corridor_product_candidates": left_corridor[
            "product_candidates"
        ],
        "left_corridor_shelf_zone": left_corridor["shelf_zone"],
        "left_corridor_raw_shelf_zone": left_corridor["raw_shelf_zone"],
        "left_track_note": left_track_note,
        "right_wrist_x_raw": None if right_raw is None else right_raw[0],
        "right_wrist_y_raw": None if right_raw is None else right_raw[1],
        "right_wrist_conf": right_conf,
        "right_wrist_x": None if right_wrist is None else right_wrist[0],
        "right_wrist_y": None if right_wrist is None else right_wrist[1],
        "right_hand_x": None if right_hand is None else right_hand[0],
        "right_hand_y": None if right_hand is None else right_hand[1],
        "right_wrist_zone": right_zone,
        "right_shelf_zone": right_shelf_zone,
        "right_product_zone": right_product_zone,
        "right_product_zone_wrist_only": right_product_zone_wrist,
        "right_product_zone_score": right_product_zone_score,
        "right_corridor_product_zone": right_corridor["product_zone"],
        "right_corridor_product_score": right_corridor["product_score"],
        "right_corridor_product_margin": right_corridor["product_margin"],
        "right_corridor_product_alternative": right_corridor[
            "product_alternative"
        ],
        "right_corridor_product_ambiguous": right_corridor[
            "product_ambiguous"
        ],
        "right_corridor_product_candidates": right_corridor[
            "product_candidates"
        ],
        "right_corridor_shelf_zone": right_corridor["shelf_zone"],
        "right_corridor_raw_shelf_zone": right_corridor["raw_shelf_zone"],
        "right_track_note": right_track_note,
        "assignment_note": assignment_note
    })

    if writer:
        for joint, colour in (
            (left_shoulder, (0, 100, 255)),
            (left_elbow, (0, 160, 255)),
            (right_shoulder, (255, 100, 0)),
            (right_elbow, (255, 160, 0))
        ):
            if joint is not None:
                cv2.circle(
                    vis,
                    (int(joint[0]), int(joint[1])),
                    4,
                    colour,
                    -1
                )

        for wrist_point, hand_point, colour in (
            (left_wrist, left_hand, (0, 255, 255)),
            (right_wrist, right_hand, (255, 255, 0))
        ):
            if wrist_point is not None and hand_point is not None:
                cv2.line(
                    vis,
                    (int(wrist_point[0]), int(wrist_point[1])),
                    (int(hand_point[0]), int(hand_point[1])),
                    colour,
                    1,
                    cv2.LINE_AA
                )
                cv2.circle(
                    vis,
                    (int(hand_point[0]), int(hand_point[1])),
                    3,
                    colour,
                    -1
                )

        for corridor, colour in (
            (left_corridor, (0, 255, 255)),
            (right_corridor, (255, 255, 0)),
        ):
            start_point = corridor.get("start_point")
            end_point = corridor.get("end_point")
            if start_point is not None and end_point is not None:
                cv2.line(
                    vis,
                    (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])),
                    colour,
                    3,
                    cv2.LINE_AA,
                )

        draw_wrist(
            vis,
            left_wrist,
            left_shelf_zone,
            left_product_zone,
            "left",
            left_track_note,
            corridor=left_corridor,
        )
        draw_wrist(
            vis,
            right_wrist,
            right_shelf_zone,
            right_product_zone,
            "right",
            right_track_note,
            corridor=right_corridor,
        )
        writer.write(vis)

    if processed_count % 50 == 0:
        print(
            f"Processed {processed_count} frames "
            f"(source frame {frame_idx}/{total_frames})"
        )

cap.release()
if writer:
    writer.release()

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

events_df = generate_zone_events(
    df,
    fps=fps,
    stride=STRIDE,
    min_duration_sec=MIN_EVENT_DURATION_SEC,
    max_missing_rows=EVENT_MAX_MISSING_ROWS
)
events_df.to_csv(OUT_EVENTS_CSV, index=False, float_format="%.3f")

run_manifest = {
    "run_id": RUN_ID,
    "video_id": re.sub(r"_crop$", "", Path(VIDEO_PATH).stem),
    "video_filename": Path(VIDEO_PATH).name,
    "started_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "video_path": os.path.abspath(VIDEO_PATH),
    "zones_json": os.path.abspath(ZONES_JSON),
    "model_path": os.path.abspath(MODEL_PATH),
    "frame_csv": os.path.abspath(OUT_CSV),
    "event_csv": os.path.abspath(OUT_EVENTS_CSV),
    "annotated_video": (
        None if OUT_VIDEO is None else os.path.abspath(OUT_VIDEO)
    ),
    "fps": fps,
    "stride": STRIDE,
    "processed_frames": len(df),
    "source_last_frame": None if df.empty else int(df["frame"].iloc[-1]),
    "events_generated": len(events_df),
    "settings": {
        "confidence": CONF,
        "image_size": IMG_SIZE,
        "stride": STRIDE,
        "max_frames": MAX_FRAMES,
        "keypoint_confidence": KEYPOINT_CONF_THR,
        "arm_keypoint_confidence": ARM_KEYPOINT_CONF_THR,
        "face_keypoint_confidence": FACE_KEYPOINT_CONF_THR,
        "hand_extension_forearm_ratio": HAND_EXTENSION_FOREARM_RATIO,
        "hand_extension_min_px": HAND_EXTENSION_MIN_PX,
        "hand_extension_max_px": HAND_EXTENSION_MAX_PX,
        "product_zone_snap_distance_px": PRODUCT_ZONE_SNAP_DISTANCE_PX,
        "shelf_zone_snap_distance_px": SHELF_ZONE_SNAP_DISTANCE_PX,
        "hand_corridor_start_px": HAND_CORRIDOR_START_PX,
        "hand_corridor_end_px": HAND_CORRIDOR_END_PX,
        "hand_corridor_step_px": HAND_CORRIDOR_STEP_PX,
        "hand_corridor_shelf_weight": HAND_CORRIDOR_SHELF_WEIGHT,
        "hand_corridor_near_weight": HAND_CORRIDOR_NEAR_WEIGHT,
        "hand_corridor_far_weight": HAND_CORRIDOR_FAR_WEIGHT,
        "hand_corridor_frame_ambiguity_ratio": (
            HAND_CORRIDOR_FRAME_AMBIGUITY_RATIO
        ),
        "smoothing_alpha": SMOOTHING_ALPHA,
        "fast_smoothing_alpha": FAST_SMOOTHING_ALPHA,
        "max_tracked_jump_px": MAX_TRACKED_JUMP_PX,
        "minimum_event_duration_sec": MIN_EVENT_DURATION_SEC,
        "event_max_missing_rows": EVENT_MAX_MISSING_ROWS
    },
    "wrist_tracking_summary": dict(sorted(stats.items()))
}

for manifest_path in (RUN_MANIFEST, LATEST_RUN_MANIFEST):
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(run_manifest, manifest_file, indent=2)

print("Done.")
print("Run ID:", RUN_ID)
print("Saved frame CSV:", OUT_CSV)
print("Saved event CSV:", OUT_EVENTS_CSV)
print("Events generated:", len(events_df))
if OUT_VIDEO:
    print("Saved annotated video:", OUT_VIDEO)
print("Saved run manifest:", RUN_MANIFEST)

print("\nWrist-tracking summary:")
for key, value in sorted(stats.items()):
    print(f"{key}: {value}")
