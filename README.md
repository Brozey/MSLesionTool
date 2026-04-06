# MSLesionTool

Multi-architecture nnUNet ensemble for automated MS lesion segmentation from brain MRI. It is a part of my master thesis. I hope that you will like it. Please don't judge me harshly based on the esthetic when you see the app. It is a passion project for me. And I really wanted to make it look cool. It's my start with programming (to be honest mostly vibe coding because I'm still learning it). I have tested around 10 architecture designs from start, till I found the best combo possible. The combo wasn't based on the holdout test, so that shouldn't be a problem. If you have any tips, tricks, recommendation, I'm fully open to them. I tried to optimize it so that it works on any machine, but I have no way to test it now. So if there is any problem, contact me and I'll fix it

## Features

- **3-Architecture Ensemble**: CNN 3D + ResEncL 3D + 2.5D (K=7) for robust segmentation
- **Best-in-class accuracy**: 0.7179 Dice on MSLesSeg-2024 (exceeds MadSeg challenge winner at 0.714)
- **Multi-planar viewer**: Axial, sagittal, coronal views with aspect-ratio-correct display
- **3D rendering**: Real-time lesion mesh with classification coloring
- **Manual editing**: Brush/eraser tools, probability-based lesion growth, lesion classification
- **ONNX Runtime support**: Optional GPU-accelerated inference via TensorRT/CUDA
- **CLI + Docker**: Headless batch processing for cloud/HPC environments

## Quick Start

### Installation

```bash
# 1. Create and activate a virtual environment (Python 3.9–3.12)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2. Run the installer (auto-detects your GPU and CUDA version)
python install.py

# 3. Download model weights from HuggingFace
python download_models.py
```

The installer will:
- Detect your NVIDIA GPU and install PyTorch with the correct CUDA support
- Install all dependencies
- Set up custom nnUNet trainers needed for inference

**Download options:**
```bash
python download_models.py                   # Default: PyTorch checkpoints (folds 1,3)
python download_models.py --onnx            # Also download ONNX models
python download_models.py --onnx-only       # Only ONNX (smaller, ~1.2 GB)
python download_models.py --all-folds       # All 5 folds per architecture
```

**Options:**
```bash
python install.py --cpu            # Force CPU-only (no GPU)
python install.py --cuda 12.8      # Override CUDA version detection
python install.py --cli-only       # Minimal CLI deps (no GUI)
python install.py --verify-only    # Check existing installation
```

**Manual install** (if you already have PyTorch set up):
```bash
pip install -r requirements.txt
```

### GUI Application

```bash
python msseg_app.py
```

1. Click **Open Patient** to select FLAIR and T1 NIfTI files
2. Click **Run Segmentation** to run the ensemble
3. Review results in the multi-planar viewer
4. Optionally refine with manual tools, then **Save Results**

### Command Line

```bash
# Single patient
python -m msseg.cli --flair patient_flair.nii.gz --t1 patient_t1.nii.gz -o seg.nii.gz

# Batch processing
python -m msseg.cli --batch /data/subjects/ --output-dir /data/results/

# ONNX backend (faster, requires exported models)
python -m msseg.cli --flair f.nii.gz --t1 t.nii.gz -o seg.nii.gz --backend onnx
```

### Docker

```bash
docker build -t msseg .
docker run --gpus all \
  -v /path/to/data:/data \
  -v /path/to/models:/app/msseg \
  msseg --flair /data/flair.nii.gz --t1 /data/t1.nii.gz -o /data/seg.nii.gz
```

## ONNX Export

Convert PyTorch checkpoints to ONNX for faster inference:

```bash
python scripts/export_onnx.py                          # Default Best-2/arch (6 models)
python scripts/export_onnx.py --all                    # All 15 models
python scripts/export_onnx.py --arch cnn3d --folds 1 3 # Specific arch+folds
```

## Architecture

| Model | Type | Patch Size | Input | Checkpoint |
|-------|------|-----------|-------|------------|
| CNN 3D | PlainConvUNet 3D | 128x128x128 | 2ch (FLAIR+T1) | ~239 MB/fold |
| ResEncL 3D | ResidualEncoderUNet 3D | 160x192x160 | 2ch (FLAIR+T1) | ~782 MB/fold |
| 2.5D (K=7) | PlainConvUNet 2D | 192x160 | 14ch (7 slices x 2ch) | ~158 MB/fold |

Default ensemble: folds {1, 3} for all 3 architectures = **6 models**.
Selection based on validation EMA fg_dice (no test-set leakage).

## Input Requirements

- **FLAIR** and **T1** volumes in NIfTI format (.nii or .nii.gz)
- Volumes must be co-registered (same space/resolution)
- nnUNet handles resampling to 1mm isotropic and z-score normalization internally
- T2 is optional (display only, not used for segmentation)

## Project Structure

```
MSLesionTool_portable/
  msseg_app.py                  # GUI application
  install.py                    # Cross-platform installer
  download_models.py            # Download weights from HuggingFace
  msseg/
    __init__.py                 # Package init
    constants.py                # Architecture registry, labels
    io.py                       # NIfTI loading/saving
    viewer.py                   # 2D slice viewer widgets
    inference.py                # PyTorch inference backend
    inference_ort.py            # ONNX Runtime inference backend
    mesh.py                     # 3D mesh builder
    cli.py                      # CLI entry point
    cnn3d/                      # CNN 3D model files
    resencl3d/                  # ResEncL 3D model files
    conv25d/                    # 2.5D model files
  trainers/                     # Custom nnUNet trainers (copied by installer)
  Dockerfile                    # GPU inference container
  requirements.txt              # Full dependencies (GUI)
  requirements-base.txt         # Dependencies without torch/nnunet (used by installer)
  requirements-inference.txt    # Minimal dependencies (CLI/Docker)
```

## Ensemble Modes

- **Best-2/arch (Recommended)**: 6 models -- top 2 folds per architecture (default)
- **All 15 folds**: Full 5-fold ensemble for all 3 architectures
- **Custom**: Select individual architectures and folds via GUI

## Notes

- Model weights are hosted on [HuggingFace](https://huggingface.co/Broozey/MSLesionTool) — run `python download_models.py` to fetch them
- ~2.4 GB for Best-2 PyTorch checkpoints, ~1.2 GB for ONNX models
- Requires NVIDIA GPU with CUDA for fast inference; CPU fallback available
- ONNX models are approximately half the checkpoint size

## Citations

If you use this tool, please cite our work and the underlying frameworks and datasets:

### This work
```bibtex
@thesis{broz2025ms,
  title={Automated MS Lesion Segmentation using Multi-Architecture nnUNet Ensemble},
  author={Broz, Jindrich},
  year={2025},
  school={VSB - Technical University of Ostrava}
}
```

### nnU-Net (segmentation framework)
```bibtex
@article{isensee2021,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

### Training datasets

**MSLesSeg-2024** (93 cases — ICPR 2024 MS Lesion Segmentation Challenge):
```bibtex
@article{guarnera2025,
  title={A multi-center MRI dataset of multiple sclerosis with cross-sectional and longitudinal data},
  author={Guarnera, Francesco and Rondinella, Alessia and Mammone, Nadia and others},
  journal={Scientific Data},
  volume={12},
  year={2025},
  publisher={Nature Publishing Group}
}
```

**WMH Challenge 2017** (60 cases — MICCAI 2017 White Matter Hyperintensity Challenge):
```bibtex
@article{kuijf2019,
  title={Standardized assessment of automatic segmentation of white matter hyperintensities and results of the WMH segmentation challenge},
  author={Kuijf, Hugo J and Biesbroek, J Matthijs and De Bresser, Jeroen and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={38},
  number={11},
  pages={2556--2568},
  year={2019},
  publisher={IEEE}
}
```

### Foundational architectures

```bibtex
@inproceedings{ronneberger2015,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  pages={234--241},
  year={2015},
  publisher={Springer}
}

@inproceedings{he2016,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  pages={770--778},
  year={2016}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
