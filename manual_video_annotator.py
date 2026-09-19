import argparse
import csv
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = PROJECT_DIR / "Videos_MERL_Shopping_Dataset"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "annotations" / "manual_event_annotations.csv" # GUI used to manually annotate product interactions and attention in MERL videos.
)

ANNOTATION_TYPES = { #Annotations are saved to CSV using the original video frame numbers.
    "pickup": {
        "category": "product_interaction",
        "description": (
            "Object leaves the shelf and begins moving with the hand."
        ),
    },
    "product_held": {
        "category": "product_interaction",
        "description": (
            "Object remains associated with the hand after retraction."
        ),
    },
    "return": {
        "category": "product_interaction",
        "description": (
            "Held object re-enters the shelf and stops following the hand."
        ),
    },
    "comparison": {
        "category": "product_interaction",
        "description": (
            "Two distinct product tracks are held or inspected together."
        ),
    },
    "attention_shelf": {
        "category": "attention",
        "description": "Head direction is visibly directed towards the shelf.",
    },
    "attention_product_or_hands": {
        "category": "attention",
        "description": (
            "Head direction is visibly directed towards a held product "
            "or the hands."
        ),
    },
    "attention_other": {
        "category": "attention",
        "description": (
            "Attention is visibly directed away from shelf and hands."
        ),
    },
    "attention_uncertain": {
        "category": "attention",
        "description": "The attention target cannot be judged reliably.",
    },
}

ANNOTATION_COLUMNS = [
    "annotation_id",
    "video_id",
    "video_filename",
    "subject_id",
    "session_id",
    "split",
    "category",
    "label",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "wrist",
    "product_track_id",
    "related_track_ids",
    "returned",
    "uncertain",
    "annotator_id",
    "notes",
    "created_at",
    "updated_at",
]

VIDEO_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<session>\d+)_crop\.mp4$",
    re.IGNORECASE,
)

def video_sort_key(path):
    match = VIDEO_PATTERN.match(path.name)
    if match:
        return (
            int(match.group("subject")),
            int(match.group("session")),
        )
    return (10**9, path.name.lower())

def video_metadata_from_path(path):
    match = VIDEO_PATTERN.match(path.name)
    if not match:
        return {
            "video_id": path.stem,
            "subject_id": "",
            "session_id": "",
            "split": "unassigned",
        }
    subject = int(match.group("subject"))
    session = int(match.group("session"))
    if subject <= 20:
        split = "development"
    elif subject <= 26:
        split = "validation"
    else:
        split = "test"
    return {
        "video_id": f"{subject}_{session}",
        "subject_id": str(subject),
        "session_id": str(session),
        "split": split,
    }

def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}

class AnnotationStore:
    """CSV-backed store that saves atomically after every change."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        self.load()

    def load(self):
        self.rows = []
        if not self.path.is_file():
            return
        with open(self.path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = set(ANNOTATION_COLUMNS).difference(
                reader.fieldnames or []
            )
            if missing:
                raise ValueError(
                    "Existing annotation CSV is missing columns: "
                    + ", ".join(sorted(missing))
                )
            for row in reader:
                self.rows.append({
                    column: row.get(column, "")
                    for column in ANNOTATION_COLUMNS
                })

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(
            temporary, "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=ANNOTATION_COLUMNS
            )
            writer.writeheader()
            writer.writerows(self.rows)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(self.path)

    def next_id(self):
        maximum = 0
        for row in self.rows:
            match = re.search(r"(\d+)$", row["annotation_id"])
            if match:
                maximum = max(maximum, int(match.group(1)))
        return f"ANN_{maximum + 1:06d}"

    def add(self, row):
        row = {
            column: str(row.get(column, ""))
            for column in ANNOTATION_COLUMNS
        }
        if not row["annotation_id"]:
            row["annotation_id"] = self.next_id()
        self.rows.append(row)
        self.save()
        return row["annotation_id"]

    def update(self, annotation_id, replacement):
        for index, row in enumerate(self.rows):
            if row["annotation_id"] != annotation_id:
                continue
            created_at = row["created_at"]
            replacement = {
                column: str(replacement.get(column, ""))
                for column in ANNOTATION_COLUMNS
            }
            replacement["annotation_id"] = annotation_id
            replacement["created_at"] = created_at
            self.rows[index] = replacement
            self.save()
            return
        raise KeyError(f"Unknown annotation ID: {annotation_id}")

    def delete(self, annotation_id):
        original_length = len(self.rows)
        self.rows = [
            row for row in self.rows
            if row["annotation_id"] != annotation_id
        ]
        if len(self.rows) == original_length:
            raise KeyError(f"Unknown annotation ID: {annotation_id}")
        self.save()

    def for_video(self, video_id):
        return sorted(
            [
                row for row in self.rows
                if row["video_id"] == video_id
            ],
            key=lambda row: (
                int(row["start_frame"]),
                int(row["end_frame"]),
                row["label"],
            ),
        )

    def find(self, annotation_id):
        for row in self.rows:
            if row["annotation_id"] == annotation_id:
                return row
        return None

class VideoAnnotator:
    def __init__(
        self,
        root,
        video_dir,
        output_path,
        initial_video=None,
        annotator_id="",
    ):
        self.root = root
        self.video_dir = Path(video_dir)
        self.store = AnnotationStore(output_path)
        self.videos = sorted(
            self.video_dir.glob("*_crop.mp4"),
            key=video_sort_key,
        )
        if not self.videos and initial_video is None:
            raise FileNotFoundError(
                f"No xx_yy_crop.mp4 videos found in {self.video_dir}"
            )

        self.capture = None
        self.video_path = None
        self.video_meta = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0
        self.playing = False
        self.playback_after_id = None
        self.photo = None
        self.slider_is_updating = False
        self.editing_annotation_id = None

        self.video_choice = tk.StringVar()
        self.position_text = tk.StringVar(value="No video loaded")
        self.start_frame_var = tk.StringVar()
        self.end_frame_var = tk.StringVar()
        self.label_var = tk.StringVar(value="pickup")
        self.wrist_var = tk.StringVar(value="right")
        self.track_id_var = tk.StringVar()
        self.related_ids_var = tk.StringVar()
        self.returned_var = tk.StringVar(value="")
        self.uncertain_var = tk.BooleanVar(value=False)
        self.annotator_var = tk.StringVar(value=annotator_id)
        self.notes_var = tk.StringVar()
        self.definition_var = tk.StringVar()
        self.edit_status_var = tk.StringVar(
            value="Adding a new annotation"
        )
        self.slider_var = tk.DoubleVar(value=0)

        self.build_ui()
        self.bind_shortcuts()
        self.on_label_changed()

        selected = None
        if initial_video:
            initial_path = Path(initial_video)
            if not initial_path.is_absolute():
                initial_path = self.video_dir / initial_path
            selected = initial_path
            if selected not in self.videos and selected.is_file():
                self.videos.append(selected)
                self.videos.sort(key=video_sort_key)
                self.video_combo["values"] = [
                    path.name for path in self.videos
                ]
        elif self.videos:
            selected = self.videos[0]

        if selected is not None:
            self.video_choice.set(selected.name)
            self.load_video(selected)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def build_ui(self):
        self.root.title("Dissertation Manual Video Annotator")
        self.root.geometry("1460x900")
        self.root.minsize(1180, 760)

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Video:").pack(side=tk.LEFT)
        self.video_combo = ttk.Combobox(
            top,
            textvariable=self.video_choice,
            values=[path.name for path in self.videos],
            state="readonly",
            width=28,
        )
        self.video_combo.pack(side=tk.LEFT, padx=(5, 5))
        self.video_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.load_selected_video(),
        )
        ttk.Button(
            top, text="Load", command=self.load_selected_video
        ).pack(side=tk.LEFT)
        ttk.Label(top, text="Annotator ID:").pack(
            side=tk.LEFT, padx=(24, 5)
        )
        ttk.Entry(
            top, textvariable=self.annotator_var, width=18
        ).pack(side=tk.LEFT)
        ttk.Button(
            top, text="Annotation guide", command=self.show_guide
        ).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body)
        right = ttk.Frame(body, width=500)
        body.add(left, weight=3)
        body.add(right, weight=2)

        video_frame = ttk.Frame(left)
        video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(
            video_frame, anchor=tk.CENTER, background="black"
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        timeline = ttk.Frame(left, padding=(0, 8, 0, 0))
        timeline.pack(fill=tk.X)
        ttk.Label(
            timeline, textvariable=self.position_text
        ).pack(anchor=tk.W)
        self.slider = ttk.Scale(
            timeline,
            from_=0,
            to=1,
            variable=self.slider_var,
            command=self.on_slider,
        )
        self.slider.pack(fill=tk.X)

        playback = ttk.Frame(left, padding=(0, 5, 0, 0))
        playback.pack(fill=tk.X)
        for text, command in (
            ("-30", lambda: self.step(-30)),
            ("-10", lambda: self.step(-10)),
            ("-1", lambda: self.step(-1)),
            ("Play / pause", self.toggle_play),
            ("+1", lambda: self.step(1)),
            ("+10", lambda: self.step(10)),
            ("+30", lambda: self.step(30)),
        ):
            ttk.Button(
                playback, text=text, command=command
            ).pack(side=tk.LEFT, padx=2)

        form = ttk.LabelFrame(
            right, text="Event annotation", padding=10
        )
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Label").grid(
            row=0, column=0, sticky=tk.W, pady=3
        )
        label_combo = ttk.Combobox(
            form,
            textvariable=self.label_var,
            values=list(ANNOTATION_TYPES),
            state="readonly",
        )
        label_combo.grid(
            row=0, column=1, columnspan=3, sticky=tk.EW, pady=3
        )
        label_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.on_label_changed(),
        )

        ttk.Label(
            form,
            textvariable=self.definition_var,
            wraplength=420,
            foreground="#555555",
        ).grid(
            row=1,
            column=0,
            columnspan=4,
            sticky=tk.W,
            pady=(0, 7),
        )

        ttk.Label(form, text="Start frame").grid(
            row=2, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(
            form, textvariable=self.start_frame_var, width=12
        ).grid(row=2, column=1, sticky=tk.EW, pady=3)
        ttk.Button(
            form, text="Mark start (S)", command=self.mark_start
        ).grid(row=2, column=2, columnspan=2, sticky=tk.EW, padx=(5, 0))

        ttk.Label(form, text="End frame").grid(
            row=3, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(
            form, textvariable=self.end_frame_var, width=12
        ).grid(row=3, column=1, sticky=tk.EW, pady=3)
        ttk.Button(
            form, text="Mark end (E)", command=self.mark_end
        ).grid(row=3, column=2, columnspan=2, sticky=tk.EW, padx=(5, 0))

        ttk.Label(form, text="Wrist").grid(
            row=4, column=0, sticky=tk.W, pady=3
        )
        ttk.Combobox(
            form,
            textvariable=self.wrist_var,
            values=["", "left", "right", "both"],
            state="readonly",
        ).grid(row=4, column=1, sticky=tk.EW, pady=3)

        ttk.Label(form, text="Product track ID").grid(
            row=5, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(
            form, textvariable=self.track_id_var
        ).grid(row=5, column=1, columnspan=3, sticky=tk.EW, pady=3)

        ttk.Label(form, text="Related track IDs").grid(
            row=6, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(
            form, textvariable=self.related_ids_var
        ).grid(row=6, column=1, columnspan=3, sticky=tk.EW, pady=3)

        ttk.Label(form, text="Returned").grid(
            row=7, column=0, sticky=tk.W, pady=3
        )
        ttk.Combobox(
            form,
            textvariable=self.returned_var,
            values=["", "true", "false", "uncertain"],
            state="readonly",
        ).grid(row=7, column=1, sticky=tk.EW, pady=3)
        ttk.Checkbutton(
            form, text="Uncertain", variable=self.uncertain_var
        ).grid(row=7, column=2, columnspan=2, sticky=tk.W, padx=(8, 0))

        ttk.Label(form, text="Notes").grid(
            row=8, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(
            form, textvariable=self.notes_var
        ).grid(row=8, column=1, columnspan=3, sticky=tk.EW, pady=3)

        ttk.Label(
            form,
            textvariable=self.edit_status_var,
            foreground="#555555",
        ).grid(
            row=9,
            column=0,
            columnspan=4,
            sticky=tk.W,
            pady=(8, 3),
        )

        buttons = ttk.Frame(form)
        buttons.grid(
            row=10, column=0, columnspan=4, sticky=tk.EW, pady=(3, 0)
        )
        ttk.Button(
            buttons,
            text="Add annotation",
            command=self.add_annotation,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            buttons,
            text="Update selected",
            command=self.update_annotation,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            buttons,
            text="Clear form",
            command=self.clear_form,
        ).pack(side=tk.LEFT, padx=4)

        existing = ttk.LabelFrame(
            right, text="Saved annotations for this video", padding=8
        )
        existing.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        tree_columns = (
            "id",
            "label",
            "start",
            "end",
            "wrist",
            "track",
        )
        self.tree = ttk.Treeview(
            existing,
            columns=tree_columns,
            show="headings",
            height=15,
            selectmode="browse",
        )
        headings = {
            "id": "ID",
            "label": "Label",
            "start": "Start",
            "end": "End",
            "wrist": "Wrist",
            "track": "Track",
        }
        widths = {
            "id": 88,
            "label": 155,
            "start": 55,
            "end": 55,
            "wrist": 55,
            "track": 90,
        }
        for column in tree_columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(
                column, width=widths[column], anchor=tk.W
            )
        self.tree.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True
        )
        scrollbar = ttk.Scrollbar(
            existing, orient=tk.VERTICAL, command=self.tree.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind(
            "<Double-1>", lambda _event: self.load_selected_annotation()
        )

        tree_buttons = ttk.Frame(right, padding=(0, 6, 0, 0))
        tree_buttons.pack(fill=tk.X)
        ttk.Button(
            tree_buttons,
            text="Load selected for editing",
            command=self.load_selected_annotation,
        ).pack(side=tk.LEFT)
        ttk.Button(
            tree_buttons,
            text="Jump to selected",
            command=self.jump_to_selected,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            tree_buttons,
            text="Delete selected",
            command=self.delete_selected,
        ).pack(side=tk.RIGHT)

    def bind_shortcuts(self):
        self.root.bind("<space>", self.shortcut_play)
        self.root.bind("<Left>", lambda event: self.shortcut_step(event, -1))
        self.root.bind("<Right>", lambda event: self.shortcut_step(event, 1))
        self.root.bind("a", lambda event: self.shortcut_step(event, -10))
        self.root.bind("d", lambda event: self.shortcut_step(event, 10))
        self.root.bind("j", lambda event: self.shortcut_step(event, -30))
        self.root.bind("l", lambda event: self.shortcut_step(event, 30))
        self.root.bind("s", self.shortcut_mark_start)
        self.root.bind("e", self.shortcut_mark_end)

    @staticmethod
    def is_text_input(widget):
        return isinstance(
            widget,
            (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox),
        )

    def shortcut_play(self, event):
        if self.is_text_input(event.widget):
            return
        self.toggle_play()
        return "break"

    def shortcut_step(self, event, amount):
        if self.is_text_input(event.widget):
            return
        self.step(amount)
        return "break"

    def shortcut_mark_start(self, event):
        if self.is_text_input(event.widget):
            return
        self.mark_start()
        return "break"

    def shortcut_mark_end(self, event):
        if self.is_text_input(event.widget):
            return
        self.mark_end()
        return "break"

    def show_guide(self):
        guide = (
            "Use raw, unannotated videos so model predictions do not "
            "influence the ground truth.\n\n"
            "Controls:\n"
            "Space: play/pause\n"
            "Left/Right: one frame\n"
            "A/D: ten frames\n"
            "J/L: thirty frames\n"
            "S: mark start\n"
            "E: mark end\n\n"
            "For a product, reuse the same Product Track ID across pickup, "
            "product_held, and return rows. For a comparison, place both IDs "
            "in Related Track IDs, separated by |.\n\n"
            "Mark Uncertain rather than guessing when the action or attention "
            "target is not reliably visible. End frames are inclusive."
        )
        messagebox.showinfo("Annotation guide", guide)

    def load_selected_video(self):
        filename = self.video_choice.get()
        matches = [
            path for path in self.videos if path.name == filename
        ]
        if not matches:
            messagebox.showerror(
                "Video not found", f"Could not find {filename}"
            )
            return
        self.load_video(matches[0])

    def load_video(self, path):
        self.pause()
        if self.capture is not None:
            self.capture.release()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            messagebox.showerror(
                "Video error", f"Could not open:\n{path}"
            )
            return
        self.capture = capture
        self.video_path = Path(path)
        self.video_meta = video_metadata_from_path(self.video_path)
        self.total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.fps = float(capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.fps = 30.0
        self.slider.configure(to=max(0, self.total_frames - 1))
        self.current_frame = 0
        self.video_choice.set(self.video_path.name)
        self.clear_form()
        self.refresh_tree()
        self.show_frame(0)

    def show_frame(self, frame_number):
        if self.capture is None or self.total_frames <= 0:
            return
        frame_number = max(
            0, min(int(frame_number), self.total_frames - 1)
        )
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok:
            return
        self.current_frame = frame_number
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        available_width = max(
            640, self.video_label.winfo_width() - 8
        )
        available_height = max(
            480, self.video_label.winfo_height() - 8
        )
        height, width = rgb.shape[:2]
        scale = min(
            available_width / width,
            available_height / height,
            1.15,
        )
        display_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )
        image = Image.fromarray(rgb).resize(
            display_size, Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo)

        seconds = frame_number / self.fps
        duration = self.total_frames / self.fps
        self.position_text.set(
            f"{self.video_meta['video_id']} | "
            f"{self.video_meta['split']} | "
            f"frame {frame_number}/{self.total_frames - 1} | "
            f"{seconds:.2f}s/{duration:.2f}s | {self.fps:.2f} fps"
        )
        self.slider_is_updating = True
        self.slider_var.set(frame_number)
        self.slider_is_updating = False

    def on_slider(self, value):
        if self.slider_is_updating:
            return
        self.pause()
        self.show_frame(int(float(value)))

    def step(self, amount):
        self.pause()
        self.show_frame(self.current_frame + amount)

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.playing = True
            self.playback_tick()

    def pause(self):
        self.playing = False
        if self.playback_after_id is not None:
            try:
                self.root.after_cancel(self.playback_after_id)
            except tk.TclError:
                pass
            self.playback_after_id = None

    def playback_tick(self):
        if not self.playing:
            return
        if self.current_frame >= self.total_frames - 1:
            self.pause()
            return
        self.show_frame(self.current_frame + 1)
        delay = max(1, int(round(1000.0 / self.fps)))
        self.playback_after_id = self.root.after(
            delay, self.playback_tick
        )

    def mark_start(self):
        self.start_frame_var.set(str(self.current_frame))

    def mark_end(self):
        self.end_frame_var.set(str(self.current_frame))

    def on_label_changed(self):
        information = ANNOTATION_TYPES[self.label_var.get()]
        self.definition_var.set(information["description"])
        label = self.label_var.get()
        if label == "return":
            self.returned_var.set("true")
        if label == "comparison":
            self.wrist_var.set("both")
        if label.startswith("attention_"):
            self.wrist_var.set("")

    def validate_form(self):
        if self.video_meta is None:
            raise ValueError("Load a video first.")
        if not self.annotator_var.get().strip():
            raise ValueError("Enter an Annotator ID.")
        try:
            start = int(self.start_frame_var.get())
            end = int(self.end_frame_var.get())
        except ValueError as error:
            raise ValueError(
                "Start and end frames must be whole numbers."
            ) from error
        if start < 0 or end < 0:
            raise ValueError("Frame numbers cannot be negative.")
        if start > end:
            raise ValueError(
                "Start frame cannot be after end frame."
            )
        if end >= self.total_frames:
            raise ValueError(
                f"End frame must be below {self.total_frames}."
            )
        label = self.label_var.get()
        if label not in ANNOTATION_TYPES:
            raise ValueError("Choose a valid annotation label.")
        if (
            label in {"pickup", "product_held", "return"}
            and not self.track_id_var.get().strip()
        ):
            raise ValueError(
                "Product Track ID is required for pickup, product_held, "
                "and return annotations."
            )
        if (
            label == "comparison"
            and not self.related_ids_var.get().strip()
        ):
            raise ValueError(
                "Related Track IDs are required for a comparison."
            )
        return start, end

    def form_row(self, annotation_id="", created_at=""):
        start, end = self.validate_form()
        now = datetime.now().isoformat(timespec="seconds")
        return {
            "annotation_id": annotation_id,
            "video_id": self.video_meta["video_id"],
            "video_filename": self.video_path.name,
            "subject_id": self.video_meta["subject_id"],
            "session_id": self.video_meta["session_id"],
            "split": self.video_meta["split"],
            "category": ANNOTATION_TYPES[
                self.label_var.get()
            ]["category"],
            "label": self.label_var.get(),
            "start_frame": start,
            "end_frame": end,
            "start_time_sec": f"{start / self.fps:.3f}",
            "end_time_sec": f"{end / self.fps:.3f}",
            "duration_sec": f"{(end - start + 1) / self.fps:.3f}",
            "wrist": self.wrist_var.get().strip(),
            "product_track_id": self.track_id_var.get().strip(),
            "related_track_ids": self.related_ids_var.get().strip(),
            "returned": self.returned_var.get().strip(),
            "uncertain": str(bool(self.uncertain_var.get())).lower(),
            "annotator_id": self.annotator_var.get().strip(),
            "notes": self.notes_var.get().strip(),
            "created_at": created_at or now,
            "updated_at": now,
        }

    def overlapping_attention(self, candidate, ignore_id=None):
        if candidate["category"] != "attention":
            return []
        start = int(candidate["start_frame"])
        end = int(candidate["end_frame"])
        overlaps = []
        for row in self.store.for_video(candidate["video_id"]):
            if row["annotation_id"] == ignore_id:
                continue
            if row["category"] != "attention":
                continue
            row_start = int(row["start_frame"])
            row_end = int(row["end_frame"])
            if max(start, row_start) <= min(end, row_end):
                overlaps.append(row["annotation_id"])
        return overlaps

    def confirm_attention_overlap(self, row, ignore_id=None):
        overlaps = self.overlapping_attention(row, ignore_id)
        if not overlaps:
            return True
        return messagebox.askyesno(
            "Overlapping attention annotation",
            "This attention interval overlaps: "
            + ", ".join(overlaps)
            + "\n\nSave it anyway?",
        )

    def add_annotation(self):
        try:
            row = self.form_row()
            if not self.confirm_attention_overlap(row):
                return
            annotation_id = self.store.add(row)
        except Exception as error:
            messagebox.showerror("Cannot save annotation", str(error))
            return
        self.refresh_tree(select_id=annotation_id)
        self.clear_form(keep_label=True)

    def update_annotation(self):
        if not self.editing_annotation_id:
            messagebox.showinfo(
                "No annotation selected",
                "Load a saved annotation for editing first.",
            )
            return
        existing = self.store.find(self.editing_annotation_id)
        try:
            row = self.form_row(
                annotation_id=self.editing_annotation_id,
                created_at=existing["created_at"],
            )
            if not self.confirm_attention_overlap(
                row, ignore_id=self.editing_annotation_id
            ):
                return
            self.store.update(self.editing_annotation_id, row)
        except Exception as error:
            messagebox.showerror("Cannot update annotation", str(error))
            return
        annotation_id = self.editing_annotation_id
        self.refresh_tree(select_id=annotation_id)
        self.clear_form(keep_label=True)

    def clear_form(self, keep_label=False):
        if not keep_label:
            self.label_var.set("pickup")
        self.start_frame_var.set("")
        self.end_frame_var.set("")
        self.wrist_var.set("right")
        self.track_id_var.set("")
        self.related_ids_var.set("")
        self.returned_var.set("")
        self.uncertain_var.set(False)
        self.notes_var.set("")
        self.editing_annotation_id = None
        self.edit_status_var.set("Adding a new annotation")
        self.on_label_changed()

    def refresh_tree(self, select_id=None):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.video_meta is None:
            return
        for row in self.store.for_video(
            self.video_meta["video_id"]
        ):
            item = self.tree.insert(
                "",
                tk.END,
                iid=row["annotation_id"],
                values=(
                    row["annotation_id"],
                    row["label"],
                    row["start_frame"],
                    row["end_frame"],
                    row["wrist"],
                    row["product_track_id"]
                    or row["related_track_ids"],
                ),
            )
            if select_id == row["annotation_id"]:
                self.tree.selection_set(item)
                self.tree.see(item)

    def selected_annotation_id(self):
        selected = self.tree.selection()
        return None if not selected else selected[0]

    def load_selected_annotation(self):
        annotation_id = self.selected_annotation_id()
        if not annotation_id:
            return
        row = self.store.find(annotation_id)
        if row is None:
            return
        self.pause()
        self.label_var.set(row["label"])
        self.start_frame_var.set(row["start_frame"])
        self.end_frame_var.set(row["end_frame"])
        self.wrist_var.set(row["wrist"])
        self.track_id_var.set(row["product_track_id"])
        self.related_ids_var.set(row["related_track_ids"])
        self.returned_var.set(row["returned"])
        self.uncertain_var.set(parse_bool(row["uncertain"]))
        self.notes_var.set(row["notes"])
        self.editing_annotation_id = annotation_id
        self.edit_status_var.set(f"Editing {annotation_id}")
        self.on_label_changed()
        self.show_frame(int(row["start_frame"]))

    def jump_to_selected(self):
        annotation_id = self.selected_annotation_id()
        if not annotation_id:
            return
        row = self.store.find(annotation_id)
        if row is not None:
            self.pause()
            self.show_frame(int(row["start_frame"]))

    def delete_selected(self):
        annotation_id = self.selected_annotation_id()
        if not annotation_id:
            return
        if not messagebox.askyesno(
            "Delete annotation",
            f"Delete {annotation_id}? This cannot be undone.",
        ):
            return
        try:
            self.store.delete(annotation_id)
        except Exception as error:
            messagebox.showerror("Cannot delete annotation", str(error))
            return
        self.refresh_tree()
        self.clear_form(keep_label=True)

    def close(self):
        self.pause()
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()

def run_self_test():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "annotations.csv"
        store = AnnotationStore(path)
        now = datetime.now().isoformat(timespec="seconds")
        row = {
            column: "" for column in ANNOTATION_COLUMNS
        }
        row.update({
            "video_id": "1_1",
            "video_filename": "1_1_crop.mp4",
            "subject_id": "1",
            "session_id": "1",
            "split": "development",
            "category": "product_interaction",
            "label": "pickup",
            "start_frame": "100",
            "end_frame": "110",
            "product_track_id": "GT_Product_01",
            "uncertain": "false",
            "annotator_id": "self-test",
            "created_at": now,
            "updated_at": now,
        })
        annotation_id = store.add(row)
        assert path.is_file()
        reloaded = AnnotationStore(path)
        assert len(reloaded.rows) == 1
        replacement = dict(reloaded.rows[0])
        replacement["end_frame"] = "112"
        reloaded.update(annotation_id, replacement)
        assert AnnotationStore(path).rows[0]["end_frame"] == "112"
        reloaded.delete(annotation_id)
        assert len(AnnotationStore(path).rows) == 0
    print("Manual annotator self-test passed.")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Open a local GUI for manually annotating product and "
            "attention events."
        )
    )
    parser.add_argument(
        "--video-dir", type=Path, default=DEFAULT_VIDEO_DIR
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional initial xx_yy_crop.mp4 video.",
    )
    parser.add_argument("--annotator", default="")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test CSV persistence without opening the GUI.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = tk.Tk()
    try:
        VideoAnnotator(
            root,
            args.video_dir,
            args.output,
            initial_video=args.video,
            annotator_id=args.annotator,
        )
    except Exception as error:
        root.destroy()
        raise SystemExit(str(error)) from error
    root.mainloop()

if __name__ == "__main__":
    main()
