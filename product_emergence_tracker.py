import argparse
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "pose_tests" #Detects possible products emerging from the shelf near the wrist.
LATEST_POSE_MANIFEST = OUTPUT_DIR / "latest_zone_pose_run.json" #The detected product is tracked across frames and linked to its estimated shelf and product zone.
LATEST_CLASSIFIER_MANIFEST = (
    OUTPUT_DIR / "latest_behaviour_classifier_run.json"
)
LATEST_EMERGENCE_MANIFEST = (
    OUTPUT_DIR / "latest_product_emergence_run.json"
)

PATCH_SIZE = 96
SEARCH_AFTER_RETRACT_SEC = 1.50
PRE_ENTRY_BASELINE_SEC = 0.60
MAX_BACKGROUND_FRAMES = 25
MIN_BACKGROUND_FRAMES = 8
FOREGROUND_DIFFERENCE_THRESHOLD = 24
CANDIDATE_SCORE_THRESHOLD = 0.58
STRONG_SCORE_THRESHOLD = 0.70
MIN_CONFIRMATION_SEC = 0.20
MAX_DETECTION_GAP_SEC = 0.20
MIN_WRIST_COUPLING_STABILITY = 0.35

EMERGENCE_EVENT_COLUMNS = [
    "emergence_id",
    "source_hand_episode",
    "wrist",
    "origin_product_zone",
    "origin_shelf_zone",
    "search_start_frame",
    "search_end_frame",
    "confirmation_start_frame",
    "confirmation_end_frame",
    "confirmation_duration_sec",
    "confirmed",
    "confidence",
    "detected_rows",
    "strong_rows",
    "post_detection_share",
    "pre_detection_share",
    "wrist_coupling_stability",
    "evidence",
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
        "false": False,
    })
    return mapped.fillna(False).astype(bool)

def resolved_path(value):
    if value is None or str(value).strip() == "":
        return None
    return Path(value).resolve()

def same_path(first, second):
    first_path = resolved_path(first)
    second_path = resolved_path(second)
    return (
        first_path is not None
        and second_path is not None
        and first_path == second_path
    )

def episode_key(value):
    if pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric

def text_or_none(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value)

def infer_fps_and_stride(features):
    frame_delta = pd.to_numeric(features["frame"], errors="coerce").diff()
    time_delta = pd.to_numeric(
        features["time_sec"], errors="coerce"
    ).diff()
    valid = (frame_delta > 0) & (time_delta > 0)
    if not valid.any():
        return 30.0, 1
    fps = float((frame_delta[valid] / time_delta[valid]).median())
    rounded = round(fps)
    if abs(fps - rounded) < 0.10:
        fps = float(rounded)
    stride = max(1, int(round(frame_delta[frame_delta > 0].median())))
    return fps, stride

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def resolve_inputs(args):
    if not LATEST_CLASSIFIER_MANIFEST.is_file():
        raise FileNotFoundError(
            "No classifier manifest was found. Run behaviour_classifier.py "
            "before product_emergence_tracker.py."
        )
    classifier_manifest = load_json(LATEST_CLASSIFIER_MANIFEST)

    pose_manifest_path = LATEST_POSE_MANIFEST
    if not pose_manifest_path.is_file():
        raise FileNotFoundError(
            "No pose manifest was found. Run zone_pose.py first."
        )
    pose_manifest = load_json(pose_manifest_path)

    feature_csv = args.features or Path(
        classifier_manifest.get("input_feature_csv", "")
    )
    event_csv = args.behaviour_events or Path(
        classifier_manifest.get("event_output_csv", "")
    )
    raw_video = args.raw_video or Path(
        pose_manifest.get("video_path", "")
    )
    annotated_input_video = args.input_video
    if annotated_input_video is None:
        value = classifier_manifest.get("behaviour_video")
        annotated_input_video = None if not value else Path(value)

    for path, description in (
        (feature_csv, "behaviour feature CSV"),
        (event_csv, "behaviour event CSV"),
        (raw_video, "raw input video"),
    ):
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"The matching {description} was not found: {path}"
            )

    classifier_pose_run = str(
        classifier_manifest.get("pose_run_id", "")
    )
    pose_run = str(pose_manifest.get("run_id", ""))
    if (
        classifier_pose_run
        and pose_run
        and classifier_pose_run != pose_run
        and args.features is None
        and args.behaviour_events is None
    ):
        raise ValueError(
            "The latest pose and classifier manifests belong to different "
            "runs. Re-run behaviour_features.py and behaviour_classifier.py."
        )

    return {
        "features": feature_csv,
        "events": event_csv,
        "raw_video": raw_video,
        "annotated_input_video": annotated_input_video,
        "classifier_manifest": classifier_manifest,
        "pose_manifest": pose_manifest,
    }

def make_output_paths(classifier_manifest):
    pose_run_id = str(classifier_manifest.get("pose_run_id", "unknown"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{pose_run_id}_{run_id}"
    return {
        "run_id": run_id,
        "pose_run_id": pose_run_id,
        "features": (
            OUTPUT_DIR / f"product_emergence_features_{stem}.csv"
        ),
        "events": OUTPUT_DIR / f"product_emergence_events_{stem}.csv",
        "video": OUTPUT_DIR / f"product_emergence_video_{stem}.mp4",
        "manifest": OUTPUT_DIR / f"product_emergence_run_{stem}.json",
    }

def required_columns(features, events):
    feature_columns = {
        "frame",
        "time_sec",
        "person_box_height_px",
        "left_elbow_x",
        "left_elbow_y",
        "right_elbow_x",
        "right_elbow_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
    }
    event_columns = {
        "behaviour",
        "start_frame",
        "end_frame",
        "wrist",
        "shelf_zone",
        "source_hand_episode",
        "product_origin_zone",
    }
    missing_features = feature_columns.difference(features.columns)
    missing_events = event_columns.difference(events.columns)
    if missing_features:
        raise ValueError(
            "Feature CSV is missing columns: "
            + ", ".join(sorted(missing_features))
        )
    if missing_events:
        raise ValueError(
            "Behaviour-event CSV is missing columns: "
            + ", ".join(sorted(missing_events))
        )

def choose_evenly(values, maximum):
    values = list(dict.fromkeys(int(value) for value in values))
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).round().astype(int)
    return [values[index] for index in indices]

def background_frame_numbers(features):
    missing_person = pd.to_numeric(
        features["person_box_height_px"], errors="coerce"
    ).isna()
    frames = features.loc[missing_person, "frame"].astype(int).tolist()
    if not frames:
        return []

    prefix = []
    for row in features.itertuples(index=False):
        height = pd.to_numeric(
            pd.Series([getattr(row, "person_box_height_px")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(height):
            break
        prefix.append(int(getattr(row, "frame")))
    source = prefix if len(prefix) >= MIN_BACKGROUND_FRAMES else frames
    return choose_evenly(source, MAX_BACKGROUND_FRAMES)

def read_selected_frames(video_path, frame_numbers):
    requested = sorted(set(int(value) for value in frame_numbers))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    selected = {}
    for frame_number in requested:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if ok:
            selected[frame_number] = frame
    capture.release()
    return selected

def build_static_background(video_path, features):
    frame_numbers = background_frame_numbers(features)
    frames = read_selected_frames(video_path, frame_numbers)
    if len(frames) < MIN_BACKGROUND_FRAMES:
        raise RuntimeError(
            "Fewer than eight reliable no-person frames were available, so "
            "a safe static background could not be built. The emergence "
            "stage was stopped instead of forcing product confirmations."
        )
    stack = np.stack([frames[number] for number in sorted(frames)])
    background = np.median(stack, axis=0).astype(np.uint8)
    return background, sorted(frames)

def canonical_patch(image, wrist, elbow, person_scale):
    wrist = np.asarray(wrist, dtype=np.float32)
    elbow = np.asarray(elbow, dtype=np.float32)
    if not np.isfinite([*wrist, *elbow, person_scale]).all():
        return None, None
    direction = wrist - elbow
    forearm_length = float(np.linalg.norm(direction))
    if forearm_length < 8:
        return None, None
    direction /= forearm_length
    perpendicular = np.array(
        [-direction[1], direction[0]], dtype=np.float32
    )
    radius = float(np.clip(
        forearm_length,
        0.22 * float(person_scale),
        0.42 * float(person_scale),
    ))
    source = np.float32([
        wrist,
        wrist + direction * radius,
        wrist + perpendicular * radius,
    ])
    target = np.float32([
        [PATCH_SIZE / 2, 72],
        [PATCH_SIZE / 2, 20],
        [90, 72],
    ])
    matrix = cv2.getAffineTransform(source, target)
    patch = cv2.warpAffine(
        image,
        matrix,
        (PATCH_SIZE, PATCH_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    inverse = cv2.invertAffineTransform(matrix)
    return patch, inverse

def object_candidate_mask(patch, background_patch):
    blurred = cv2.GaussianBlur(patch, (5, 5), 0)
    background_blurred = cv2.GaussianBlur(
        background_patch, (5, 5), 0
    )
    difference = cv2.cvtColor(
        cv2.absdiff(blurred, background_blurred),
        cv2.COLOR_BGR2GRAY,
    )
    foreground = difference >= FOREGROUND_DIFFERENCE_THRESHOLD

    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    likely_skin = (
        (cr >= 132)
        & (cr <= 180)
        & (cb >= 72)
        & (cb <= 140)
    )

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 130) > 0
    edges = cv2.dilate(
        edges.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(bool)

    focus = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)  # Ignores sleeve, torso and hair behind the wrist.
    focus[2:76, 8:88] = True
    mask = (
        foreground
        & ~likely_skin
        & ((saturation >= 28) | edges)
        & focus
    )
    mask = cv2.morphologyEx(
        mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    return mask

def component_candidates(mask, inverse):
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        cx, cy = centroids[component]
        if area < 70 or area > 2700:
            continue
        if cy > 75 or abs(cx - PATCH_SIZE / 2) > 42:
            continue

        touches_side = x <= 9 or x + width >= 87
        tall_boundary_fragment = (
            touches_side and height >= 60 and height > 1.50 * width
        )
        oversized_boundary_fragment = touches_side and area >= 1500
        if tall_boundary_fragment or oversized_boundary_fragment:
            continue

        centrality = max(
            0.0, 1.0 - abs(cx - PATCH_SIZE / 2) / 42
        )
        distal_position = max(0.0, min(1.0, (76 - cy) / 64))
        area_score = min(area / 700.0, 1.0)
        score = (
            0.35 * area_score
            + 0.35 * distal_position
            + 0.30 * centrality
        )
        global_point = cv2.transform(
            np.float32([[[cx, cy]]]), inverse
        )[0, 0]
        corners = np.float32([[
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ]])
        global_corners = cv2.transform(corners, inverse)[0]
        candidates.append({
            "score": float(score),
            "area": int(area),
            "relative_x": float(cx),
            "relative_y": float(cy),
            "x": float(global_point[0]),
            "y": float(global_point[1]),
            "polygon": [
                [float(point[0]), float(point[1])]
                for point in global_corners
            ],
        })
    return sorted(
        candidates, key=lambda item: item["score"], reverse=True
    )

def detect_frame_candidate(frame, background, row, wrist):
    values = [
        row.get(f"{wrist}_wrist_x"),
        row.get(f"{wrist}_wrist_y"),
        row.get(f"{wrist}_elbow_x"),
        row.get(f"{wrist}_elbow_y"),
        row.get("person_box_height_px"),
    ]
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any():
        return None
    wrist_point = (float(numeric.iloc[0]), float(numeric.iloc[1]))
    elbow_point = (float(numeric.iloc[2]), float(numeric.iloc[3]))
    scale = float(numeric.iloc[4])
    patch, inverse = canonical_patch(
        frame, wrist_point, elbow_point, scale
    )
    background_patch, _ = canonical_patch(
        background, wrist_point, elbow_point, scale
    )
    if patch is None or background_patch is None:
        return None
    mask = object_candidate_mask(patch, background_patch)
    candidates = component_candidates(mask, inverse)
    return None if not candidates else candidates[0]

def matching_hand_event(hand_events, retract):
    wrist = str(retract["wrist"])
    episode = episode_key(retract["source_hand_episode"])
    keys = hand_events["source_hand_episode"].map(episode_key)
    matches = hand_events[
        (hand_events["wrist"] == wrist) & (keys == episode)
    ]
    if matches.empty:
        return None
    return matches.sort_values("start_frame").iloc[0]

def next_same_wrist_hand_start(hand_events, retract):
    matches = hand_events[
        (hand_events["wrist"] == str(retract["wrist"]))
        & (
            pd.to_numeric(
                hand_events["start_frame"], errors="coerce"
            )
            > int(retract["end_frame"])
        )
    ]
    if matches.empty:
        return None
    return int(matches["start_frame"].min())

def make_searches(features, events, fps, stride):
    minimum_frame = int(features["frame"].min())
    maximum_frame = int(features["frame"].max())
    hand_events = events[events["behaviour"] == "hand_in_shelf"].copy()
    retracts = events[
        events["behaviour"] == "retract_from_shelf"
    ].copy()
    searches = []
    for _, retract in retracts.sort_values("start_frame").iterrows():
        hand = matching_hand_event(hand_events, retract)
        if hand is None:
            continue
        search_start = int(retract["end_frame"]) + stride
        search_end = min(
            maximum_frame,
            int(retract["end_frame"])
            + int(round(SEARCH_AFTER_RETRACT_SEC * fps)),
        )
        next_hand_start = next_same_wrist_hand_start(
            hand_events, retract
        )
        if next_hand_start is not None:
            search_end = min(search_end, next_hand_start - stride)
        pre_start = max(
            minimum_frame,
            int(hand["start_frame"])
            - int(round(PRE_ENTRY_BASELINE_SEC * fps)),
        )
        searches.append({
            "source_hand_episode": episode_key(
                retract["source_hand_episode"]
            ),
            "wrist": str(retract["wrist"]),
            "origin_product_zone": (
                text_or_none(retract.get("product_origin_zone"))
                or text_or_none(hand.get("product_origin_zone"))
            ),
            "origin_shelf_zone": (
                text_or_none(retract.get("shelf_zone"))
                or text_or_none(hand.get("shelf_zone"))
            ),
            "pre_start_frame": pre_start,
            "pre_end_frame": int(hand["start_frame"]) - stride,
            "search_start_frame": search_start,
            "search_end_frame": search_end,
            "measurements": {},
        })
    return searches

def frame_tasks_for_searches(features, searches):
    available_frames = set(features["frame"].astype(int).tolist())
    tasks = {}
    for search_index, search in enumerate(searches):
        ranges = (
            (
                search["pre_start_frame"],
                search["pre_end_frame"],
                "pre",
            ),
            (
                search["search_start_frame"],
                search["search_end_frame"],
                "post",
            ),
        )
        for start, end, phase in ranges:
            for frame_number in available_frames:
                if start <= frame_number <= end:
                    tasks.setdefault(frame_number, []).append(
                        (search_index, phase)
                    )
    return tasks

def measure_searches(
    raw_video,
    background,
    features,
    searches,
):
    row_by_frame = {
        int(row["frame"]): row
        for _, row in features.iterrows()
    }
    tasks = frame_tasks_for_searches(features, searches)
    if not tasks:
        return
    maximum_frame = max(tasks)
    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {raw_video}")
    source_frame = 0
    while source_frame <= maximum_frame:
        ok, frame = capture.read()
        if not ok:
            break
        for search_index, phase in tasks.get(source_frame, []):
            search = searches[search_index]
            candidate = detect_frame_candidate(
                frame,
                background,
                row_by_frame[source_frame],
                search["wrist"],
            )
            search["measurements"][source_frame] = {
                "phase": phase,
                "candidate": candidate,
            }
        source_frame += 1
    capture.release()

def detection_clusters(records, stride, fps):
    positive = [
        record
        for record in records
        if record["score"] >= CANDIDATE_SCORE_THRESHOLD
    ]
    if not positive:
        return []
    maximum_gap = max(
        stride,
        int(round(MAX_DETECTION_GAP_SEC * fps)),
    )
    clusters = []
    current = [positive[0]]
    for record in positive[1:]:
        if record["frame"] - current[-1]["frame"] <= maximum_gap:
            current.append(record)
        else:
            clusters.append(current)
            current = [record]
    clusters.append(current)
    return clusters

def coupling_stability(records):
    if len(records) < 2:
        return 0.0
    points = np.asarray([
        [record["relative_x"], record["relative_y"]]
        for record in records
    ])
    median = np.median(points, axis=0)
    spread = float(np.mean(np.linalg.norm(points - median, axis=1)))
    return float(np.clip(1.0 - spread / 30.0, 0.0, 1.0))

def candidate_records(search, phase):
    records = []
    for frame_number, measurement in sorted(
        search["measurements"].items()
    ):
        if measurement["phase"] != phase:
            continue
        candidate = measurement["candidate"]
        if candidate is None:
            records.append({
                "frame": frame_number,
                "score": 0.0,
            })
        else:
            records.append({
                "frame": frame_number,
                **candidate,
            })
    return records

def evaluate_search(search, fps, stride):
    pre_records = candidate_records(search, "pre")
    post_records = candidate_records(search, "post")
    pre_positive = [
        row for row in pre_records
        if row["score"] >= CANDIDATE_SCORE_THRESHOLD
    ]
    post_positive = [
        row for row in post_records
        if row["score"] >= CANDIDATE_SCORE_THRESHOLD
    ]
    pre_share = len(pre_positive) / max(1, len(pre_records))
    post_share = len(post_positive) / max(1, len(post_records))

    minimum_rows = max(
        3,
        int(np.ceil(MIN_CONFIRMATION_SEC * fps / stride)) + 1,
    )
    valid = []
    for cluster in detection_clusters(post_records, stride, fps):
        duration = (
            cluster[-1]["frame"] - cluster[0]["frame"]
        ) / fps
        stability = coupling_stability(cluster)
        strong_rows = sum(
            row["score"] >= STRONG_SCORE_THRESHOLD
            for row in cluster
        )
        confirmed = (
            len(cluster) >= minimum_rows
            and duration + 1e-9 >= MIN_CONFIRMATION_SEC
            and stability >= MIN_WRIST_COUPLING_STABILITY
        )
        if confirmed:
            valid.append({
                "records": cluster,
                "duration": duration,
                "stability": stability,
                "strong_rows": strong_rows,
            })

    best = None
    if valid:
        best = max(
            valid,
            key=lambda item: (
                len(item["records"]),
                item["strong_rows"],
                np.mean([
                    row["score"] for row in item["records"]
                ]),
            ),
        )

    if best is None:
        stability = coupling_stability(post_positive)
        maximum_score = max(
            [row["score"] for row in post_positive],
            default=0.0,
        )
        confidence = float(np.clip(
            0.35 * maximum_score
            + 0.30 * min(post_share / 0.50, 1.0)
            + 0.20 * stability
            + 0.15 * max(0.0, post_share - pre_share),
            0.0,
            0.59,
        ))
        evidence = (
            "object-like pixels were absent, too brief, or insufficiently "
            "coupled to the wrist; the shelf interaction remains a pickup "
            "candidate"
        )
        return {
            "confirmed": False,
            "confidence": confidence,
            "records": [],
            "confirmation_start_frame": None,
            "confirmation_end_frame": None,
            "duration": None,
            "detected_rows": len(post_positive),
            "strong_rows": sum(
                row["score"] >= STRONG_SCORE_THRESHOLD
                for row in post_positive
            ),
            "post_share": post_share,
            "pre_share": pre_share,
            "stability": stability,
            "evidence": evidence,
        }

    records = best["records"]
    mean_score = float(np.mean([row["score"] for row in records]))
    strong_share = best["strong_rows"] / len(records)
    novelty = max(0.0, post_share - pre_share)
    confidence = float(np.clip(
        0.30 * mean_score
        + 0.25 * min(post_share / 0.50, 1.0)
        + 0.25 * best["stability"]
        + 0.15 * strong_share
        + 0.05 * min(novelty / 0.40, 1.0),
        0.60,
        0.98,
    ))
    evidence = (
        f"{len(records)} persistent distal-wrist detections over "
        f"{best['duration']:.2f}s; wrist-coupling stability="
        f"{best['stability']:.2f}; the product origin is inherited from "
        "the triggering shelf/product zone"
    )
    return {
        "confirmed": True,
        "confidence": confidence,
        "records": records,
        "confirmation_start_frame": records[0]["frame"],
        "confirmation_end_frame": records[-1]["frame"],
        "duration": best["duration"],
        "detected_rows": len(post_positive),
        "strong_rows": sum(
            row["score"] >= STRONG_SCORE_THRESHOLD
            for row in post_positive
        ),
        "post_share": post_share,
        "pre_share": pre_share,
        "stability": best["stability"],
        "evidence": evidence,
    }

def initialise_detection_columns(features):
    for wrist in ("left", "right"):
        defaults = {
            f"{wrist}_held_product_available": False,
            f"{wrist}_held_product_detected": False,
            f"{wrist}_held_product_conf": np.nan,
            f"{wrist}_held_product_x": np.nan,
            f"{wrist}_held_product_y": np.nan,
            f"{wrist}_held_product_source": None,
            f"{wrist}_held_product_origin_zone": None,
            f"{wrist}_product_emergence_episode": np.nan,
            f"{wrist}_product_emergence_score": 0.0,
        }
        for column, default in defaults.items():
            if column not in features.columns:
                features[column] = default
        for column in (
            f"{wrist}_held_product_source",
            f"{wrist}_held_product_origin_zone",
        ):
            features[column] = features[column].astype("object")

    global_defaults = {
        "held_product_detected": False,
        "held_product_available": False,
        "held_product_conf": np.nan,
        "held_product_x": np.nan,
        "held_product_y": np.nan,
        "held_product_origin_zone": None,
        "held_product_detection_source": None,
    }
    for column, default in global_defaults.items():
        if column not in features.columns:
            features[column] = default
    for column in (
        "held_product_origin_zone",
        "held_product_detection_source",
    ):
        features[column] = features[column].astype("object")

def interpolate_confirmed_records(features, search, evaluation):
    wrist = search["wrist"]
    records = evaluation["records"]
    if not records:
        return
    record_frame = np.asarray(
        [record["frame"] for record in records], dtype=float
    )
    record_x = np.asarray([record["x"] for record in records], dtype=float)
    record_y = np.asarray([record["y"] for record in records], dtype=float)
    record_score = np.asarray(
        [record["score"] for record in records], dtype=float
    )
    start = int(evaluation["confirmation_start_frame"])
    end = int(evaluation["confirmation_end_frame"])
    mask = (features["frame"] >= start) & (features["frame"] <= end)
    target_frames = features.loc[mask, "frame"].to_numpy(dtype=float)
    if len(target_frames) == 0:
        return

    x_values = np.interp(target_frames, record_frame, record_x)
    y_values = np.interp(target_frames, record_frame, record_y)
    score_values = np.interp(
        target_frames, record_frame, record_score
    )
    confidence_values = np.clip(
        0.55 * evaluation["confidence"] + 0.45 * score_values,
        0.0,
        1.0,
    )
    features.loc[mask, f"{wrist}_held_product_available"] = True
    features.loc[mask, f"{wrist}_held_product_detected"] = True
    features.loc[mask, f"{wrist}_held_product_conf"] = confidence_values
    features.loc[mask, f"{wrist}_held_product_x"] = x_values
    features.loc[mask, f"{wrist}_held_product_y"] = y_values
    features.loc[
        mask, f"{wrist}_held_product_source"
    ] = "shelf_exit_emergence"
    features.loc[
        mask, f"{wrist}_held_product_origin_zone"
    ] = search["origin_product_zone"]
    features.loc[
        mask, f"{wrist}_product_emergence_episode"
    ] = search["source_hand_episode"]

def add_raw_scores(features, search):
    wrist = search["wrist"]
    for frame_number, measurement in search["measurements"].items():
        if measurement["phase"] != "post":
            continue
        candidate = measurement["candidate"]
        score = 0.0 if candidate is None else candidate["score"]
        mask = features["frame"] == frame_number
        current = pd.to_numeric(
            features.loc[
                mask, f"{wrist}_product_emergence_score"
            ],
            errors="coerce",
        ).fillna(0.0)
        features.loc[
            mask, f"{wrist}_product_emergence_score"
        ] = np.maximum(current.to_numpy(), score)

def combine_global_detection_fields(features):
    left_available = as_boolean(
        features["left_held_product_available"]
    )
    right_available = as_boolean(
        features["right_held_product_available"]
    )
    left_conf = pd.to_numeric(
        features["left_held_product_conf"], errors="coerce"
    ).fillna(-1.0)
    right_conf = pd.to_numeric(
        features["right_held_product_conf"], errors="coerce"
    ).fillna(-1.0)
    choose_left = left_available & (
        ~right_available | (left_conf >= right_conf)
    )
    choose_right = right_available & ~choose_left
    either = left_available | right_available

    features.loc[either, "held_product_detected"] = True
    features.loc[either, "held_product_available"] = True
    for wrist, selection in (
        ("left", choose_left),
        ("right", choose_right),
    ):
        features.loc[selection, "held_product_conf"] = features.loc[
            selection, f"{wrist}_held_product_conf"
        ]
        features.loc[selection, "held_product_x"] = features.loc[
            selection, f"{wrist}_held_product_x"
        ]
        features.loc[selection, "held_product_y"] = features.loc[
            selection, f"{wrist}_held_product_y"
        ]
        features.loc[
            selection, "held_product_origin_zone"
        ] = features.loc[
            selection, f"{wrist}_held_product_origin_zone"
        ]
        features.loc[
            selection, "held_product_detection_source"
        ] = features.loc[
            selection, f"{wrist}_held_product_source"
        ]

def build_emergence_outputs(features, events, raw_video):
    required_columns(features, events)
    features = features.sort_values("frame").reset_index(drop=True).copy()
    events = events.sort_values("start_frame").reset_index(drop=True).copy()
    fps, stride = infer_fps_and_stride(features)
    background, background_frames = build_static_background(
        raw_video, features
    )
    searches = make_searches(features, events, fps, stride)
    measure_searches(raw_video, background, features, searches)
    initialise_detection_columns(features)

    event_rows = []
    confirmed_number = 0
    for search_number, search in enumerate(searches, start=1):
        add_raw_scores(features, search)
        evaluation = evaluate_search(search, fps, stride)
        if evaluation["confirmed"]:
            confirmed_number += 1
            interpolate_confirmed_records(features, search, evaluation)
        event_rows.append({
            "emergence_id": f"Emergence_{search_number:02d}",
            "source_hand_episode": search["source_hand_episode"],
            "wrist": search["wrist"],
            "origin_product_zone": search["origin_product_zone"],
            "origin_shelf_zone": search["origin_shelf_zone"],
            "search_start_frame": search["search_start_frame"],
            "search_end_frame": search["search_end_frame"],
            "confirmation_start_frame": (
                evaluation["confirmation_start_frame"]
            ),
            "confirmation_end_frame": (
                evaluation["confirmation_end_frame"]
            ),
            "confirmation_duration_sec": evaluation["duration"],
            "confirmed": evaluation["confirmed"],
            "confidence": round(evaluation["confidence"], 3),
            "detected_rows": evaluation["detected_rows"],
            "strong_rows": evaluation["strong_rows"],
            "post_detection_share": round(
                evaluation["post_share"], 3
            ),
            "pre_detection_share": round(
                evaluation["pre_share"], 3
            ),
            "wrist_coupling_stability": round(
                evaluation["stability"], 3
            ),
            "evidence": evaluation["evidence"],
        })

    combine_global_detection_fields(features)
    emergence_events = pd.DataFrame(
        event_rows, columns=EMERGENCE_EVENT_COLUMNS
    )
    return (
        features,
        emergence_events,
        fps,
        stride,
        background_frames,
        confirmed_number,
    )

def draw_text(frame, text, x, y, color, scale=0.48, thickness=1):
    cv2.putText(
        frame,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def annotate_video(input_video, features, emergence_events, output_path):
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise FileNotFoundError(
            f"Could not open annotated input video: {input_video}"
        )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {output_path}")

    confirmed_events = emergence_events[
        as_boolean(emergence_events["confirmed"])
    ]
    written = 0
    for _, row in features.iterrows():
        ok, frame = capture.read()
        if not ok:
            break
        source_frame = int(row["frame"])
        labels = []
        for wrist, color in (
            ("left", (255, 80, 255)),
            ("right", (80, 255, 80)),
        ):
            available = as_boolean(pd.Series([
                row.get(f"{wrist}_held_product_available", False)
            ])).iloc[0]
            score = float(pd.to_numeric(pd.Series([
                row.get(f"{wrist}_product_emergence_score", 0.0)
            ]), errors="coerce").fillna(0.0).iloc[0])
            if available:
                x = float(row[f"{wrist}_held_product_x"])
                y = float(row[f"{wrist}_held_product_y"])
                wrist_x = float(row[f"{wrist}_wrist_x"])
                wrist_y = float(row[f"{wrist}_wrist_y"])
                confidence = float(row[f"{wrist}_held_product_conf"])
                if np.isfinite([wrist_x, wrist_y, x, y]).all():
                    cv2.line(
                        frame,
                        (int(wrist_x), int(wrist_y)),
                        (int(x), int(y)),
                        color,
                        2,
                    )
                if np.isfinite([x, y]).all():
                    cv2.circle(frame, (int(x), int(y)), 12, color, 3)
                origin = row.get(
                    f"{wrist}_held_product_origin_zone"
                )
                labels.append(
                    f"{wrist}: CONFIRMED {confidence:.2f} | {origin}"
                )
            elif score >= CANDIDATE_SCORE_THRESHOLD:
                labels.append(
                    f"{wrist}: emergence evidence {score:.2f} (uncertain)"
                )

        active = confirmed_events[
            (confirmed_events["confirmation_start_frame"] <= source_frame)
            & (
                confirmed_events["confirmation_end_frame"]
                >= source_frame
            )
        ]
        if labels or not active.empty:
            panel_height = 34 + 24 * max(1, len(labels))
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (16, height - panel_height - 16),
                (min(width - 16, 760), height - 16),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)
            draw_text(
                frame,
                "SHELF-EXIT PRODUCT CONFIRMATION",
                28,
                height - panel_height + 8,
                (230, 230, 230),
                scale=0.50,
                thickness=1,
            )
            y = height - panel_height + 32
            for label in labels:
                color = (
                    (80, 255, 80)
                    if "CONFIRMED" in label
                    else (0, 210, 255)
                )
                draw_text(frame, label, 28, y, color, scale=0.46)
                y += 22

        writer.write(frame)
        written += 1

    capture.release()
    writer.release()
    return written

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Confirm products emerging from shelf zones without product "
            "labels or a trained hand-crop classifier."
        )
    )
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--behaviour-events", type=Path, default=None)
    parser.add_argument("--raw-video", type=Path, default=None)
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--features-output", type=Path, default=None)
    parser.add_argument("--events-output", type=Path, default=None)
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Generate CSV outputs without rendering an annotated video.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    inputs = resolve_inputs(args)
    paths = make_output_paths(inputs["classifier_manifest"])
    feature_output = args.features_output or paths["features"]
    event_output = args.events_output or paths["events"]
    video_output = args.video_output or paths["video"]

    features = pd.read_csv(inputs["features"])
    behaviour_events = pd.read_csv(inputs["events"])
    (
        augmented_features,
        emergence_events,
        fps,
        stride,
        background_frames,
        confirmed_number,
    ) = build_emergence_outputs(
        features,
        behaviour_events,
        inputs["raw_video"],
    )

    feature_output.parent.mkdir(parents=True, exist_ok=True)
    event_output.parent.mkdir(parents=True, exist_ok=True)
    augmented_features.to_csv(
        feature_output, index=False, float_format="%.4f"
    )
    emergence_events.to_csv(
        event_output, index=False, float_format="%.4f"
    )

    rendered_video = None
    frames_written = 0
    annotated_input = inputs["annotated_input_video"]
    if not args.skip_video:
        if annotated_input is None or not annotated_input.is_file():
            print(
                "Warning: no matching behaviour video was found; CSV "
                "outputs were still generated."
            )
        else:
            video_output.parent.mkdir(parents=True, exist_ok=True)
            frames_written = annotate_video(
                annotated_input,
                augmented_features,
                emergence_events,
                video_output,
            )
            rendered_video = video_output

    manifest = {
        "product_emergence_run_id": paths["run_id"],
        "pose_run_id": paths["pose_run_id"],
        "classifier_run_id": inputs["classifier_manifest"].get(
            "classifier_run_id"
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_features": str(inputs["features"].resolve()),
        "input_behaviour_events": str(inputs["events"].resolve()),
        "input_raw_video": str(inputs["raw_video"].resolve()),
        "input_annotated_video": (
            None
            if annotated_input is None
            else str(annotated_input.resolve())
        ),
        "augmented_feature_csv": str(feature_output.resolve()),
        "emergence_event_csv": str(event_output.resolve()),
        "annotated_video": (
            None
            if rendered_video is None
            else str(rendered_video.resolve())
        ),
        "video_frames_written": frames_written,
        "fps": fps,
        "stride": stride,
        "shelf_exit_searches": len(emergence_events),
        "confirmed_emergences": confirmed_number,
        "background_source_frames": background_frames,
        "method": {
            "requires_visible_product_on_shelf": False,
            "requires_manual_labels": False,
            "identifies_product_sku": False,
            "origin_zone_is_inherited_from_retraction": True,
            "uses_static_background": True,
            "uses_skin_rejection": True,
            "uses_forearm_aligned_distal_wrist_region": True,
            "requires_temporal_wrist_coupling": True,
            "weak_evidence_remains_candidate": True,
        },
        "thresholds": {
            "search_after_retract_sec": SEARCH_AFTER_RETRACT_SEC,
            "pre_entry_baseline_sec": PRE_ENTRY_BASELINE_SEC,
            "foreground_difference_threshold": (
                FOREGROUND_DIFFERENCE_THRESHOLD
            ),
            "candidate_score_threshold": CANDIDATE_SCORE_THRESHOLD,
            "strong_score_threshold": STRONG_SCORE_THRESHOLD,
            "minimum_confirmation_sec": MIN_CONFIRMATION_SEC,
            "maximum_detection_gap_sec": MAX_DETECTION_GAP_SEC,
            "minimum_wrist_coupling_stability": (
                MIN_WRIST_COUPLING_STABILITY
            ),
        },
    }
    for target in (paths["manifest"], LATEST_EMERGENCE_MANIFEST):
        with open(target, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

    print("Product emergence confirmation complete.")
    print(f"Input behaviour events: {inputs['events']}")
    print(f"Input features: {inputs['features']}")
    print(f"Saved augmented features: {feature_output}")
    print(f"Saved emergence events: {event_output}")
    if rendered_video is not None:
        print(f"Saved annotated video: {rendered_video}")
        print(f"Video frames annotated: {frames_written}")
    print(f"Shelf-exit searches: {len(emergence_events)}")
    print(f"Confirmed product emergences: {confirmed_number}")
    print(
        "Unconfirmed shelf exits remain pickup candidates; they are not "
        "forced to product-held."
    )

if __name__ == "__main__":
    main()
