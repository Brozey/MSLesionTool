#!/usr/bin/env python3
"""Run ResEncL 3D 5-fold inference on DS001 (MSLesSeg) and DS002 (WMH).

Produces:
  - Per-fold predictions (folds 0-4 individually, with TTA)
  - 5-fold ensemble prediction (softmax averaging across folds, with TTA)
  - Evaluation metrics for each configuration
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ── Environment check ─────────────────────────────────────────────────────────
for _var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    if _var not in os.environ:
        sys.exit(f"ERROR: Set {_var} environment variable (standard nnUNet setup)")

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ["nnUNet_results"]) / "Dataset003_Combined" \
            / "nnUNetTrainer_WandB__nnUNetResEncUNetLPlans__3d_fullres"
DS001_IMAGES = Path(os.environ["nnUNet_raw"]) / "Dataset001_MSLesSeg" / "imagesTs"
DS001_LABELS = Path(os.environ["nnUNet_raw"]) / "Dataset001_MSLesSeg" / "labelsTs"
DS002_IMAGES = Path(os.environ["nnUNet_raw"]) / "Dataset002_WMH" / "imagesTs"
DS002_LABELS = Path(os.environ["nnUNet_raw"]) / "Dataset002_WMH" / "labelsTs"
PRED_BASE = REPO_ROOT / "results" / "predictions"

FOLDS = (0, 1, 2, 3, 4)


def run_inference(folds, output_name, input_dir, save_probs=True):
    """Run nnUNet inference for given folds on input_dir."""
    out_dir = PRED_BASE / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_n = len(list(input_dir.glob("*_0000.nii.gz")))
    have_nii = len(list(out_dir.glob("*.nii.gz")))
    if have_nii >= exp_n:
        print(f"[SKIP] {output_name} already has {have_nii}/{exp_n} predictions")
        return

    print(f"\n{'='*60}")
    print(f"Running: {output_name} (folds={folds}, {exp_n} cases)")
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
        list_of_lists_or_source_folder=str(input_dir),
        output_folder_or_list_of_truncated_output_files=str(out_dir),
        save_probabilities=save_probs,
        overwrite=False,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
    )
    dt = time.time() - t0
    have_nii = len(list(out_dir.glob("*.nii.gz")))
    print(f"[DONE] {output_name}: {have_nii}/{exp_n} in {dt/60:.1f} min")

    # Free VRAM
    del predictor
    torch.cuda.empty_cache()


def build_ensemble(fold_dir_names, out_dir_name, ref_fold_dir, thresholds):
    """Average softmax across folds, threshold, save NIfTIs."""
    ref_dir = PRED_BASE / ref_fold_dir
    case_ids = sorted(
        p.name.replace(".nii.gz.npz", "")
        for p in ref_dir.glob("*.nii.gz.npz")
    ) if ref_dir.exists() else []

    if not case_ids:
        print(f"[WARN] No cases found for ensemble {out_dir_name}")
        return

    print(f"\n{'='*60}")
    print(f"  Building 5-fold ensemble: {out_dir_name} ({len(case_ids)} cases)")
    print(f"{'='*60}")

    for case_id in case_ids:
        probs_list = []
        for fd in fold_dir_names:
            npz_path = PRED_BASE / fd / f"{case_id}.nii.gz.npz"
            if not npz_path.exists():
                print(f"  [WARN] Missing {fd}/{case_id}.nii.gz.npz — skip case")
                break
            probs = np.load(str(npz_path))["probabilities"]
            probs_list.append(probs)

        if len(probs_list) != len(fold_dir_names):
            continue

        avg_probs = np.mean(probs_list, axis=0)

        ref_nii = nib.load(str(PRED_BASE / ref_fold_dir / f"{case_id}.nii.gz"))

        for thr in thresholds:
            if avg_probs.shape[0] == 2:
                seg = (avg_probs[1] >= thr).astype(np.uint8)
            else:
                seg = np.argmax(avg_probs, axis=0).astype(np.uint8)

            seg = seg.transpose(2, 1, 0)

            suffix = f"_thr{thr:.2f}" if thr != 0.5 else ""
            thr_dir = PRED_BASE / f"{out_dir_name}{suffix}"
            thr_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(seg, ref_nii.affine, ref_nii.header),
                     str(thr_dir / f"{case_id}.nii.gz"))

        print(f"  {case_id} done")

    print(f"[DONE] Ensemble {out_dir_name} built")


def evaluate_dir(pred_dir, labels_dir, label=""):
    """Evaluate all predictions against ground truth."""
    try:
        from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures
    except ImportError:
        print("[ERROR] MetricsReloaded not installed")
        return None

    pred_dir = Path(pred_dir)
    labels_dir = Path(labels_dir)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"  No predictions in {pred_dir}")
        return None

    dices, sens_list, ppv_list = [], [], []
    for pf in pred_files:
        gt_file = labels_dir / pf.name
        if not gt_file.exists():
            continue
        pred = (nib.load(str(pf)).get_fdata() > 0).astype(np.uint8)
        gt_img = nib.load(str(gt_file))
        gt = (gt_img.get_fdata() > 0).astype(np.uint8)
        pixdim = list(gt_img.header.get_zooms()[:3])
        bpm = BinaryPairwiseMeasures(pred, gt, connectivity_type=1, pixdim=pixdim)
        dices.append(bpm.dsc())
        sens_list.append(bpm.sensitivity())
        ppv_list.append(bpm.positive_predictive_value())

    if not dices:
        return None

    mean_dice = float(np.mean(dices))
    print(f"  {label:50s}  Dice={mean_dice:.4f}±{np.std(dices):.4f}  "
          f"Sens={np.mean(sens_list):.4f}  PPV={np.mean(ppv_list):.4f}  "
          f"n={len(dices)}")
    return mean_dice


def main():
    # ===============================================================
    #  DS001 — MSLesSeg (22 test cases)
    # ===============================================================
    print("\n" + "=" * 60)
    print("  DS001 — MSLesSeg per-fold inference (ResEncL 3D)")
    print("=" * 60)

    # 5-fold ensemble (all weights loaded together — nnUNet native)
    run_inference((0, 1, 2, 3, 4), "ResEncL_3D_5fold_TTA", DS001_IMAGES)

    # Individual folds
    for fold in FOLDS:
        run_inference((fold,), f"ResEncL_3D_fold{fold}_TTA", DS001_IMAGES)

    # Build softmax-averaged ensemble at multiple thresholds
    ds001_fold_dirs = [f"ResEncL_3D_fold{f}_TTA" for f in FOLDS]
    build_ensemble(ds001_fold_dirs, "ResEncL_3D_5fold_ensemble_TTA",
                   ref_fold_dir="ResEncL_3D_fold0_TTA",
                   thresholds=[0.30, 0.40, 0.50])

    # ===============================================================
    #  DS002 — WMH (110 test cases)
    # ===============================================================
    print("\n" + "=" * 60)
    print("  DS002 — WMH per-fold inference (ResEncL 3D)")
    print("=" * 60)

    run_inference((0, 1, 2, 3, 4), "ResEncL_3D_5fold_TTA_DS002", DS002_IMAGES)

    for fold in FOLDS:
        run_inference((fold,), f"ResEncL_3D_fold{fold}_TTA_DS002", DS002_IMAGES)

    ds002_fold_dirs = [f"ResEncL_3D_fold{f}_TTA_DS002" for f in FOLDS]
    build_ensemble(ds002_fold_dirs, "ResEncL_3D_5fold_ensemble_TTA_DS002",
                   ref_fold_dir="ResEncL_3D_fold0_TTA_DS002",
                   thresholds=[0.30, 0.40, 0.50])

    # ===============================================================
    #  EVALUATION
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  EVALUATION — DS001 MSLesSeg (22 cases)")
    print(f"{'='*60}")
    for fold in FOLDS:
        evaluate_dir(PRED_BASE / f"ResEncL_3D_fold{fold}_TTA", DS001_LABELS,
                     f"ResEncL-3D fold {fold} (DS001)")
    evaluate_dir(PRED_BASE / "ResEncL_3D_5fold_TTA", DS001_LABELS,
                 "ResEncL-3D 5-fold native ens (DS001)")
    for thr in [0.50, 0.40, 0.30]:
        suffix = f"_thr{thr:.2f}" if thr != 0.5 else ""
        evaluate_dir(PRED_BASE / f"ResEncL_3D_5fold_ensemble_TTA{suffix}", DS001_LABELS,
                     f"ResEncL-3D 5-fold softmax ens thr={thr:.2f} (DS001)")

    print(f"\n{'='*60}")
    print(f"  EVALUATION — DS002 WMH (110 cases)")
    print(f"{'='*60}")
    for fold in FOLDS:
        evaluate_dir(PRED_BASE / f"ResEncL_3D_fold{fold}_TTA_DS002", DS002_LABELS,
                     f"ResEncL-3D fold {fold} (DS002)")
    evaluate_dir(PRED_BASE / "ResEncL_3D_5fold_TTA_DS002", DS002_LABELS,
                 "ResEncL-3D 5-fold native ens (DS002)")
    for thr in [0.50, 0.40, 0.30]:
        suffix = f"_thr{thr:.2f}" if thr != 0.5 else ""
        evaluate_dir(PRED_BASE / f"ResEncL_3D_5fold_ensemble_TTA_DS002{suffix}", DS002_LABELS,
                     f"ResEncL-3D 5-fold softmax ens thr={thr:.2f} (DS002)")

    print(f"\n{'='*60}")
    print(f"  All ResEncL 3D inference complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
