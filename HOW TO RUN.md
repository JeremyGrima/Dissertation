# Hybrid V2 reproduction workflow

Run all commands from the repository root. Download the MERL data and pose weights first, as described in `README.md` and `DATASET.md`.

## 1. Convert the MERL labels

```powershell
python .\merl_labels.py `
  --video-dir '.\Videos_MERL_Shopping_Dataset' `
  --label-dir '.\Labels MERL' `
  --output-dir '.\evaluation'
```

## 2. Extract development and validation features

```powershell
python .\run_pipeline_batch.py `
  --split train `
  --stop-after features `
  --output-root '.\evaluation\hybrid_v2_training' `
  --batch-run-id hybrid_v2_training_features

python .\run_pipeline_batch.py `
  --split validation `
  --stop-after features `
  --output-root '.\evaluation\hybrid_v2_validation_features' `
  --batch-run-id hybrid_v2_validation_features
```

## 3. Prepare labelled model rows

```powershell
python .\prepare_merl_training_data.py `
  --batch-manifest '.\evaluation\hybrid_v2_training\hybrid_v2_training_features\batch_manifest.json' `
  --split development

python .\prepare_merl_training_data.py `
  --batch-manifest '.\evaluation\hybrid_v2_validation_features\hybrid_v2_validation_features\batch_manifest.json' `
  --split validation
```

The resulting manifests are placed below `model_data/hybrid_v2/`.

## 4. Train, tune and freeze the classifier

```powershell
python .\train_merl_classifier.py `
  --training-data '.\model_data\hybrid_v2\development_hybrid_v2_training_features\prepared_dataset_manifest.json' `
  --validation-data '.\model_data\hybrid_v2\validation_hybrid_v2_validation_features\prepared_dataset_manifest.json' `
  --output-dir '.\models\hybrid_v2' `
  --run-id hybrid_v2 `
  --freeze
```

The supplied product-origin calibration is the frozen validation configuration used in the dissertation. The cleaned manual annotation file is included if recalibration is required.

## 5. Run Hybrid V2 on validation

```powershell
python .\run_pipeline_batch.py `
  --split validation `
  --classifier-mode hybrid_v2 `
  --model-manifest '.\models\hybrid_v2\hybrid_v2_model_manifest.json' `
  --origin-calibration '.\models\hybrid_v2\product_origin_calibration.json' `
  --reuse-upstream-batch '.\evaluation\hybrid_v2_validation_features\hybrid_v2_validation_features\batch_manifest.json' `
  --output-root '.\evaluation\hybrid_v2_validation' `
  --batch-run-id hybrid_v2_validation
```

## 6. Run the frozen test

Do not change the model, thresholds, zones or calibration after inspecting test results.

```powershell
python .\run_pipeline_batch.py `
  --split test `
  --classifier-mode hybrid_v2 `
  --model-manifest '.\models\hybrid_v2\hybrid_v2_model_manifest.json' `
  --origin-calibration '.\models\hybrid_v2\product_origin_calibration.json' `
  --confirm-frozen `
  --output-root '.\evaluation\hybrid_v2_test' `
  --batch-run-id hybrid_v2_test
```

Evaluate a completed batch using the included `annotations/manual_event_annotations.csv`:

```powershell
python .\evaluate_pipeline.py `
  --batch-manifest '.\evaluation\hybrid_v2_test\hybrid_v2_test\batch_manifest.json' `
  --split test
```

The original frozen identifiers and reported metrics are preserved in `run_configuration.json`; the portable reproduction commands intentionally use new run identifiers so the archived private runs are not overwritten.
