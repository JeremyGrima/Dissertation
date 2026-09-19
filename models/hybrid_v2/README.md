# Hybrid V2 model files

This directory contains the frozen validation thresholds and product-origin calibration used by the final pipeline. The fitted `hybrid_v2_merl_model.joblib` and generated `hybrid_v2_model_manifest.json` are not distributed publicly.

Run `train_merl_classifier.py` as described in `WORKFLOW.md` to regenerate both files. The SHA-256 checksum of the privately archived fitted model is recorded in `run_configuration.json` for audit comparison.

Never load an untrusted `joblib`, pickle or similar serialized model because deserialization can execute arbitrary code.
