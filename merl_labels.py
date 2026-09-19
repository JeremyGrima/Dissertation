"""Convert the official MERL Shopping MATLAB labels into CSV files.

The original dataset uses matching filenames:

    1_1_crop.mp4  <->  1_1_label.mat

MATLAB frame indices are one-based. The pipeline and OpenCV use zero-based
source frames, so both representations are retained in the event CSV.
"""

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_DIR = PROJECT_DIR / "Videos_MERL_Shopping_Dataset"
DEFAULT_LABEL_DIR = PROJECT_DIR / "Labels MERL"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "evaluation"

ACTION_NAMES = [
    "reach_to_shelf",
    "retract_from_shelf",
    "hand_in_shelf",
    "inspect_product",
    "inspect_shelf",
]

VIDEO_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<session>\d+)_crop\.mp4$",
    re.IGNORECASE,
)
LABEL_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<session>\d+)_label\.mat$",
    re.IGNORECASE,
)

MANIFEST_COLUMNS = [
    "video_id",
    "subject_id",
    "session_id",
    "split",
    "video_path",
    "label_path",
    "total_frames",
    "fps",
    "width",
    "height",
    "duration_sec",
    "event_count",
]

EVENT_COLUMNS = [
    "event_id",
    "video_id",
    "subject_id",
    "session_id",
    "split",
    "class_index_matlab",
    "class_index_python",
    "behaviour",
    "merl_start_frame",
    "merl_end_frame",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "duration_frames",
    "duration_sec",
]

FRAME_COLUMNS = [
    "video_id",
    "subject_id",
    "session_id",
    "split",
    "source_frame",
    "merl_frame",
    "time_sec",
    *ACTION_NAMES,
    "background",
    "overlap_count",
    "behaviour_labels",
]

def dataset_split(subject_id):
    """Return the split from the official released-dataset README."""
    if 1 <= subject_id <= 20:
        return "development"
    if 21 <= subject_id <= 26:
        return "validation"
    if 27 <= subject_id <= 41:
        return "test"
    raise ValueError(f"Unexpected MERL subject ID: {subject_id}")

def parse_dataset_file(path, pattern):
    match = pattern.match(path.name)
    if not match:
        return None
    subject = int(match.group("subject"))
    session = int(match.group("session"))
    return {
        "video_id": f"{subject}_{session}",
        "subject_id": subject,
        "session_id": session,
    }

def discover_files(directory, suffix_pattern, parser_pattern):
    discovered = {}
    for path in directory.glob(suffix_pattern):
        metadata = parse_dataset_file(path, parser_pattern)
        if metadata is None:
            continue
        video_id = metadata["video_id"]
        if video_id in discovered:
            raise ValueError(
                f"Duplicate dataset ID {video_id} in {directory}"
            )
        discovered[video_id] = {
            **metadata,
            "path": path.resolve(),
        }
    return discovered

def video_id_sort_key(video_id):
    subject, session = video_id.split("_")
    return int(subject), int(session)

def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    metadata = {
        "total_frames": int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if metadata["total_frames"] <= 0:
        raise ValueError(f"Video contains no readable frames: {path}")
    if metadata["fps"] <= 0:
        raise ValueError(f"Video has an invalid frame rate: {path}")
    return metadata

def normalise_intervals(value, label_path, action_name):
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.size % 2 != 0:
        raise ValueError(
            f"{label_path.name}: {action_name} contains an odd number "
            "of frame values."
        )
    intervals = np.asarray(array, dtype=np.int64).reshape(-1, 2)
    return intervals

def load_label_intervals(label_path):
    contents = loadmat(
        label_path,
        squeeze_me=True,
        struct_as_record=False,
    )
    if "tlabs" not in contents:
        raise ValueError(f"{label_path.name} does not contain tlabs.")
    cells = np.asarray(contents["tlabs"], dtype=object).reshape(-1)
    if len(cells) != len(ACTION_NAMES):
        raise ValueError(
            f"{label_path.name} contains {len(cells)} tlabs cells; "
            f"expected {len(ACTION_NAMES)}."
        )
    return {
        action_name: normalise_intervals(
            cells[index], label_path, action_name
        )
        for index, action_name in enumerate(ACTION_NAMES)
    }

def validate_intervals(
    intervals_by_action,
    total_frames,
    label_path,
):
    for action_name, intervals in intervals_by_action.items():
        for start, end in intervals:
            if start < 1:
                raise ValueError(
                    f"{label_path.name}: {action_name} starts at "
                    f"MERL frame {start}; labels must be one-based."
                )
            if end < start:
                raise ValueError(
                    f"{label_path.name}: {action_name} interval "
                    f"{start}-{end} is reversed."
                )
            if end > total_frames:
                raise ValueError(
                    f"{label_path.name}: {action_name} ends at frame "
                    f"{end}, beyond its {total_frames}-frame video."
                )

def write_csv_atomic(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(
        temporary, "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)

def open_frame_writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    file = open(
        temporary, "w", encoding="utf-8-sig", newline=""
    )
    writer = csv.DictWriter(file, fieldnames=FRAME_COLUMNS)
    writer.writeheader()
    return temporary, file, writer

def write_video_frame_rows(
    writer,
    video_id,
    subject_id,
    session_id,
    split,
    total_frames,
    fps,
    intervals_by_action,
):
    masks = {
        action_name: np.zeros(total_frames, dtype=bool)
        for action_name in ACTION_NAMES
    }
    for action_name, intervals in intervals_by_action.items():
        for merl_start, merl_end in intervals:
            masks[action_name][
                int(merl_start) - 1:int(merl_end)  # Python excludes the end; MERL includes it.
            ] = True

    for source_frame in range(total_frames):
        active = [
            action_name for action_name in ACTION_NAMES
            if masks[action_name][source_frame]
        ]
        row = {
            "video_id": video_id,
            "subject_id": subject_id,
            "session_id": session_id,
            "split": split,
            "source_frame": source_frame,
            "merl_frame": source_frame + 1,
            "time_sec": f"{source_frame / fps:.4f}",
            "background": str(not active).lower(),
            "overlap_count": len(active),
            "behaviour_labels": "|".join(active)
            if active else "background",
        }
        for action_name in ACTION_NAMES:
            row[action_name] = str(
                bool(masks[action_name][source_frame])
            ).lower()
        writer.writerow(row)

def expected_video_ids():
    expected = set()
    for subject in range(1, 33):
        for session in range(1, 4):
            expected.add(f"{subject}_{session}")
    for subject in range(33, 41):
        expected.add(f"{subject}_1")
    expected.update({"41_1", "41_2"})
    return expected

def convert_dataset(
    video_dir,
    label_dir,
    output_dir,
    include_frame_csv=True,
):
    videos = discover_files(video_dir, "*_crop.mp4", VIDEO_PATTERN)
    labels = discover_files(label_dir, "*_label.mat", LABEL_PATTERN)
    if not videos:
        raise FileNotFoundError(
            f"No xx_yy_crop.mp4 files found in {video_dir}"
        )
    if not labels:
        raise FileNotFoundError(
            f"No xx_yy_label.mat files found in {label_dir}"
        )

    video_ids = set(videos)
    label_ids = set(labels)
    if video_ids != label_ids:
        missing_labels = sorted(
            video_ids - label_ids, key=video_id_sort_key
        )
        missing_videos = sorted(
            label_ids - video_ids, key=video_id_sort_key
        )
        raise ValueError(
            "Video/label IDs do not match. "
            f"Missing labels: {missing_labels}; "
            f"missing videos: {missing_videos}"
        )

    expected = expected_video_ids()
    if video_ids != expected:
        missing = sorted(expected - video_ids, key=video_id_sort_key)
        unexpected = sorted(
            video_ids - expected, key=video_id_sort_key
        )
        raise ValueError(
            "The files do not match the released 106-video dataset. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    manifest_path = output_dir / "merl_video_manifest.csv"
    event_path = output_dir / "merl_ground_truth_events.csv"
    frame_path = output_dir / "merl_ground_truth_frames.csv"

    frame_temporary = None
    frame_file = None
    frame_writer = None
    if include_frame_csv:
        frame_temporary, frame_file, frame_writer = open_frame_writer(
            frame_path
        )

    manifest_rows = []
    event_rows = []
    class_counts = Counter()
    split_video_counts = Counter()
    split_event_counts = Counter()

    try:
        for video_id in sorted(video_ids, key=video_id_sort_key):
            video_record = videos[video_id]
            label_record = labels[video_id]
            subject_id = video_record["subject_id"]
            session_id = video_record["session_id"]
            split = dataset_split(subject_id)
            video_info = inspect_video(video_record["path"])
            intervals_by_action = load_label_intervals(
                label_record["path"]
            )
            validate_intervals(
                intervals_by_action,
                video_info["total_frames"],
                label_record["path"],
            )

            video_event_count = 0
            for class_index, action_name in enumerate(ACTION_NAMES):
                intervals = intervals_by_action[action_name]
                for event_number, (merl_start, merl_end) in enumerate(
                    intervals, start=1
                ):
                    start_frame = int(merl_start) - 1
                    end_frame = int(merl_end) - 1
                    duration_frames = end_frame - start_frame + 1
                    event_rows.append({
                        "event_id": (
                            f"{video_id}_{action_name}_{event_number:03d}"
                        ),
                        "video_id": video_id,
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "split": split,
                        "class_index_matlab": class_index + 1,
                        "class_index_python": class_index,
                        "behaviour": action_name,
                        "merl_start_frame": int(merl_start),
                        "merl_end_frame": int(merl_end),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "start_time_sec": (
                            f"{start_frame / video_info['fps']:.4f}"
                        ),
                        "end_time_sec": (
                            f"{end_frame / video_info['fps']:.4f}"
                        ),
                        "duration_frames": duration_frames,
                        "duration_sec": (
                            f"{duration_frames / video_info['fps']:.4f}"
                        ),
                    })
                    video_event_count += 1
                    class_counts[action_name] += 1
                    split_event_counts[split] += 1

            manifest_rows.append({
                "video_id": video_id,
                "subject_id": subject_id,
                "session_id": session_id,
                "split": split,
                "video_path": str(video_record["path"]),
                "label_path": str(label_record["path"]),
                "total_frames": video_info["total_frames"],
                "fps": f"{video_info['fps']:.4f}",
                "width": video_info["width"],
                "height": video_info["height"],
                "duration_sec": (
                    f"{video_info['total_frames'] / video_info['fps']:.4f}"
                ),
                "event_count": video_event_count,
            })
            split_video_counts[split] += 1

            if frame_writer is not None:
                write_video_frame_rows(
                    frame_writer,
                    video_id,
                    subject_id,
                    session_id,
                    split,
                    video_info["total_frames"],
                    video_info["fps"],
                    intervals_by_action,
                )

        expected_split_counts = {
            "development": 60,
            "validation": 18,
            "test": 28,
        }
        if dict(split_video_counts) != expected_split_counts:
            raise ValueError(
                "Official split counts are incorrect: "
                f"{dict(split_video_counts)}"
            )

        write_csv_atomic(
            manifest_path, MANIFEST_COLUMNS, manifest_rows
        )
        write_csv_atomic(event_path, EVENT_COLUMNS, event_rows)

        if frame_file is not None:
            frame_file.flush()
            os.fsync(frame_file.fileno())
            frame_file.close()
            frame_file = None
            frame_temporary.replace(frame_path)
    finally:
        if frame_file is not None:
            frame_file.close()
        if (
            frame_temporary is not None
            and frame_temporary.exists()
        ):
            frame_temporary.unlink()

    return {
        "manifest_path": manifest_path,
        "event_path": event_path,
        "frame_path": frame_path if include_frame_csv else None,
        "videos": len(manifest_rows),
        "events": len(event_rows),
        "class_counts": dict(class_counts),
        "split_video_counts": dict(split_video_counts),
        "split_event_counts": dict(split_event_counts),
        "total_frame_rows": sum(
            int(row["total_frames"]) for row in manifest_rows
        ),
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert MERL xx_yy_label.mat files into auditable event and "
            "frame-level CSV ground truth."
        )
    )
    parser.add_argument(
        "--video-dir", type=Path, default=DEFAULT_VIDEO_DIR
    )
    parser.add_argument(
        "--label-dir", type=Path, default=DEFAULT_LABEL_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--skip-frame-csv",
        action="store_true",
        help=(
            "Create only the manifest and event CSV. The frame CSV can be "
            "generated later by rerunning without this option."
        ),
    )
    return parser.parse_args()

def main():
    args = parse_args()
    result = convert_dataset(
        args.video_dir,
        args.label_dir,
        args.output_dir,
        include_frame_csv=not args.skip_frame_csv,
    )
    print("MERL label conversion complete.")
    print(f"Videos mapped: {result['videos']}")
    print(f"Events converted: {result['events']}")
    print(f"Official split: {result['split_video_counts']}")
    print(f"Events by class: {result['class_counts']}")
    print(f"Events by split: {result['split_event_counts']}")
    print(f"Saved video manifest: {result['manifest_path']}")
    print(f"Saved ground-truth events: {result['event_path']}")
    if result["frame_path"] is not None:
        print(
            "Saved ground-truth frames: "
            f"{result['frame_path']} "
            f"({result['total_frame_rows']} rows)"
        )

if __name__ == "__main__":
    main()
