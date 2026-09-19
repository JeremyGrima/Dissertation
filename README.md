# Hybrid V2 Shelf-Interaction Pipeline

This repository contains the source code and configuration for the Hybrid V2 system developed as part of my dissertation.

The dissertation investigates how pose, wrist movement, shelf context, and product-related evidence can be integrated to recognise customer behaviour and model product interactions from fixed overhead video. 

## Repository Contents

The repository contains the source code, configuration files, annotations, validation settings, and supporting files required to run the pipeline.

The MERL videos are not included. Generated files such as feature CSVs, logs, rendered videos, trained model files, and evaluation outputs are also excluded.

See `HOW TO RUN.md` for the commands used to train and evaluate Hybrid V2.

## Repository Structure

```text

|-- *.py                              Pipeline, training and evaluation scripts
|-- annotations/
|   |-- README.md                     Annotation information
|   |-- manual_event_annotations.csv  Cleaned product annotations
|   `-- manual_event_annotations.template.csv
|-- frames/
|     -- zones.json                    Shelf and product-zone definitions
|-- models/hybrid_v2/
|   |-- README.md
|   |-- hybrid_v2_validation_tuning.csv
|     -- product_origin_calibration.json
|-- HOW TO RUN.md
|-- requirements.txt
 -- run_configuration.json
```

## Setup

The final system was developed and evaluated using Python 3.9.13.

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt')"
```

The final command downloads the YOLOv8s pose model required by the pipeline. The model weights are not included in this repository.

## Final Results

Hybrid V2 was evaluated on the 28-video MERL test split.

The system achieved a mean F1 score of 0.61 and mAP of 0.52 across the five MERL behaviour classes. The definition-aligned mean F1 for product interactions was 0.26.

Product-origin accuracy was 0.23 and returned-status accuracy was 0.46. The attention proxy covered 36.6% of eligible frames and achieved 56.6% agreement when available. The attention results represent agreement with expected inspection targets rather than direct gaze accuracy.

Full evaluation settings and results are provided in `run_configuration.json`.

## Dataset

The MERL Shopping Dataset is not included in this repository and must be
downloaded separately from the official MERL source.

The separate manual product annotations are included as a cleaned CSV. They
contain event labels and timing, but no MERL media or official label files.

Dataset page:
https://www.merl.com/research/downloads/MERL_Shopping_Dataset

The dataset files can then be converted into the format required by the
pipeline using: merl_labels.py

This creates the local video manifest and frame-level and event-level
ground-truth files used by the pipeline.


## Licence

The source code in this repository is provided for the dissertation project. Third-party datasets, libraries, models, and model weights remain subject to their respective licences and terms of use.
