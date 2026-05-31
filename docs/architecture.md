# Architecture

MSLesionTool segments MS lesions with an ensemble of three complementary nnU-Net configurations.
Combining architectures with different receptive fields and inductive biases is more robust than any
single model, and the per-fold diversity further stabilises predictions.

## The three architectures

| Model | Type | Patch size | Input channels | Notes |
|-------|------|-----------|----------------|-------|
| CNN 3D | PlainConvUNet 3D | 128×128×128 | 2 (FLAIR + T1) | Volumetric baseline, balanced cost/accuracy |
| ResEncL 3D | ResidualEncoderUNet 3D | 160×192×160 | 2 (FLAIR + T1) | Large residual-encoder model, strongest single architecture |
| 2.5D (K=7) | PlainConvUNet 2D | 192×160 | 14 (7 axial slices × 2) | Axial context stack, complements the 3D models |

All models consume co-registered FLAIR + T1. nnU-Net resamples to 1 mm isotropic and applies z-score
normalisation internally.

## Ensembling

Predictions are averaged in softmax probability space, then thresholded.

- **Best-2/arch (default):** the top 2 folds per architecture → 6 models.
- **All 15 folds:** the full 5-fold ensemble for all three architectures.
- **Custom:** any subset of architectures/folds (via the GUI).

Folds are selected on validation EMA `fg_dice`, never on the held-out test set, so reported numbers
are free of test-set leakage.

## Backends

- **PyTorch** (`msseg/inference.py`) — default; uses the nnU-Net predictor with optional test-time augmentation.
- **ONNX Runtime** (`msseg/inference_ort.py`) — optional, GPU-accelerated via TensorRT/CUDA; export models with
  `scripts/export_onnx.py`.
