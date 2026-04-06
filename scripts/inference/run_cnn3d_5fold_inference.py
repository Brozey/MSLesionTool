#!/usr/bin/env python3
"""Run CNN 3D 5-fold inference on MSLesSeg test set.

Produces:
  - Per-fold predictions (folds 0-4 individually, with TTA)
  - 5-fold ensemble prediction (softmax averaging across folds, with TTA)
"""
import os
import sys
import time
from pathlib import Path

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ── Environment check ─────────────────────────────────────────────────────────
for _var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    if _var not in os.environ:
        sys.exit(f"ERROR: Set {_var} environment variable (standard nnUNet setup)")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths
MODEL_DIR = Path(os.environ["nnUNet_results"]) / "Dataset003_Combined" / "nnUNetTrainer_WandB__nnUNetPlans__3d_fullres"
INPUT_DIR = Path(os.environ["nnUNet_raw"]) / "Dataset001_MSLesSeg" / "imagesTs"
PRED_BASE = REPO_ROOT / "results" / "predictions"

def run_inference(folds, output_name, save_probs=True):
    out_dir = PRED_BASE / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_n = len(list(INPUT_DIR.glob("*_0000.nii.gz")))
    have_nii = len(list(out_dir.glob("*.nii.gz")))
    if have_nii >= exp_n:
        print(f"[SKIP] {output_name} already has {have_nii}/{exp_n} predictions")
        return

    print(f"\n{'='*60}")
    print(f"Running: {output_name} (folds={folds})")
    print(f"{'='*60}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,       # TTA
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(MODEL_DIR),
        use_folds=folds,
        checkpoint_name="checkpoint_best.pth",
    )

    t0 = time.time()
    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(INPUT_DIR),
        output_folder_or_list_of_truncated_output_files=str(out_dir),
        save_probabilities=save_probs,
        overwrite=False,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
    )
    dt = time.time() - t0
    have_nii = len(list(out_dir.glob("*.nii.gz")))
    print(f"[DONE] {output_name}: {have_nii}/{exp_n} in {dt/60:.1f} min")


if __name__ == "__main__":
    curr_input = INPUT_DIR
    run_inference((0, 1, 2, 3, 4), "CNN_3D_5fold_TTA")
    for fold in range(5):
        run_inference((fold,), f"CNN_3D_fold{fold}_TTA")

    # DS002
    INPUT_DIR = Path(os.environ["nnUNet_raw"]) / "Dataset002_WMH" / "imagesTs"
    run_inference((0, 1, 2, 3, 4), "CNN_3D_5fold_TTA_DS002")
    for fold in range(5):
        run_inference((fold,), f"CNN_3D_fold{fold}_TTA_DS002")

    print("\nAll CNN 3D inference complete!")
