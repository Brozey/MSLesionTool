# Reproducing the study

This repository ships the **tool** (GUI, CLI, Docker, inference). The training, evaluation, and
explainability code used to produce the published results is not included in the public tree to keep it
lean, but the pipeline is summarised here and is available on request.

## Data

Two public datasets were combined into nnU-Net-format datasets (see `config/dataset_config.yaml` for the
layout):

- **MSLesSeg-2024** — 93 cases, ICPR 2024 MS Lesion Segmentation Challenge.
- **WMH-2017** — 60 cases, MICCAI 2017 White Matter Hyperintensity Challenge.

Inputs are co-registered FLAIR + T1; nnU-Net handles resampling to 1 mm isotropic and z-score normalisation.

## Training

- Three architectures (CNN 3D, ResEncL 3D, 2.5D K=7), 5-fold cross-validation each.
- Custom nnU-Net trainers (`trainers/`) add Weights & Biases logging, early stopping, and the 2.5D axial-context
  variant. Focal- and TopK-loss variants were also explored.

## Evaluation

- Per-subject metrics (DSC, HD95, NSD, sensitivity, PPV) via MetricsReloaded.
- Exhaustive search over fold/architecture combinations, with the final ensemble selected on validation EMA
  `fg_dice` (no test-set leakage).
- Lesion-level analysis by lesion size, post-processing experiments, and an LST-AI baseline comparison.

## Explainability

Grad-CAM, RISE, and occlusion-sensitivity analyses were used to inspect model behaviour.

## ONNX export

`scripts/export_onnx.py` converts the PyTorch checkpoints to ONNX for the accelerated inference backend.
