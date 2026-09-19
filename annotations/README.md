# Manual product annotations

`manual_event_annotations.csv` contains the 1,147 original product-interaction annotations used for the dissertation evaluation.

The public copy omits `video_filename`, `subject_id`, `session_id`, `annotator_id`, `notes`, `created_at`, `updated_at`, and `uncertain`. 

No MERL videos, images or official annotation files are redistributed here. Obtain the original dataset from [MERL](https://www.merl.com/research/downloads/MERL_Shopping_Dataset/) and comply with its terms. 

The included header-only template is provided for creating annotations for another authorised dataset. The cleaned CSV is in the path expected by `evaluate_pipeline.py` for reproducing the product-interaction, origin and returned-status evaluation.
