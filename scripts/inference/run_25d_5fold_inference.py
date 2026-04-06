#!/usr/bin/env python3
"""
2.5D inference (5-fold + fold_all) on DS001 (MSLesSeg, 22 cases) and
DS002 (WMH, 110 cases).

Produces:
  - Per-fold predictions with TTA (folds 0-4) for both datasets
  - 5-fold softmax-averaged ensemble for both datasets
  - fold_all predictions for both datasets
  - Per-case and summary evaluation metrics
"""
import os, sys, time, shutil
from pathlib import Path
import numpy as np
import torch
import nibabel as nib

# ── Environment check ─────────────────────────────────────────────────────────
for _var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    if _var not in os.environ:
        sys.exit(f"ERROR: Set {_var} environment variable (standard nnUNet setup)")

REPO_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.trainers.nnUNetTrainer_25D import nnUNetPredictor25D

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ["nnUNet_results"]) / "Dataset003_Combined" / "nnUNetTrainer_25D__nnUNetPlans__3d_fullres"
DS001_IMAGES = Path(os.environ["nnUNet_raw"]) / "Dataset001_MSLesSeg" / "imagesTs"
DS001_LABELS = Path(os.environ["nnUNet_raw"]) / "Dataset001_MSLesSeg" / "labelsTs"
DS002_IMAGES = Path(os.environ["nnUNet_raw"]) / "Dataset002_WMH" / "imagesTs"
DS002_LABELS = Path(os.environ["nnUNet_raw"]) / "Dataset002_WMH" / "labelsTs"
PRED_BASE = REPO_ROOT / "results" / "predictions"

FOLDS = (0, 1, 2, 3, 4)

# Folds 1,2,4 were retrained but have now been re-inferred — no force needed
DS001_FORCE_FOLDS = set()


def _build_predictor(use_folds, checkpoint_name, device):
    """Instantiate and initialize nnUNetPredictor25D."""
    predictor = nnUNetPredictor25D(
        num_adjacent_slices=7,
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(MODEL_DIR),
        use_folds=use_folds,
        checkpoint_name=checkpoint_name,
    )
    return predictor


def run_fold_inference(fold, device, images_dir, out_dir_name, force=False):
    """Run single-fold 2.5D TTA inference, saving softmax NPZ + hard NIfTI."""
    out_dir = PRED_BASE / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    case_files = sorted(images_dir.glob("*_0000.nii.gz"))
    total = len(case_files)

    # Optionally wipe stale predictions
    if force:
        for f in out_dir.glob("*.nii.gz"):
            f.unlink()
        for f in out_dir.glob("*.npz"):
            f.unlink()
        print(f"[FORCE] Cleared stale predictions in {out_dir_name}")

    have_nii = len(list(out_dir.glob("*.nii.gz")))
    have_npz = len(list(out_dir.glob("*.nii.gz.npz")))
    if have_nii >= total and have_npz >= total:
        print(f"[SKIP] {out_dir_name} already complete ({have_nii} nii, {have_npz} npz)")
        return

    print(f"\n{'='*60}")
    print(f"  2.5D Fold {fold} — TTA on {total} cases  [{out_dir_name}]")
    print(f"{'='*60}")

    predictor = _build_predictor((fold,), "checkpoint_best.pth", device)
    config_3d = predictor.plans_manager.get_configuration("3d_fullres")
    n_channels = len(predictor.dataset_json["channel_names"])

    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from nnunetv2.inference.export_prediction import export_prediction_from_logits

    reader = SimpleITKIO()
    preprocessor = DefaultPreprocessor()

    t0_total = time.time()
    for ci, ch0_file in enumerate(case_files):
        case_id = ch0_file.name.replace("_0000.nii.gz", "")
        nii_out = out_dir / f"{case_id}.nii.gz"
        npz_out = out_dir / f"{case_id}.nii.gz.npz"

        if nii_out.exists() and npz_out.exists():
            continue

        channel_files = [images_dir / f"{case_id}_{ch:04d}.nii.gz" for ch in range(n_channels)]
        # DS001 has 1 channel; DS002 has 2 channels — only _0000 is always present
        channel_files = [f for f in channel_files if f.exists()]
        print(f"  [{ci+1}/{total}] {case_id}", end=" ", flush=True)
        t0 = time.time()

        images, properties = reader.read_images([str(f) for f in channel_files])
        data, seg, properties = preprocessor.run_case_npy(
            images, None, properties,
            predictor.plans_manager, config_3d, predictor.dataset_json,
        )
        data_tensor = torch.from_numpy(data).float()

        predictor.network.load_state_dict(predictor.list_of_parameters[0])
        logits = predictor.predict_sliding_window_return_logits(data_tensor.to(device))
        logits_np = logits.float().cpu().numpy()

        export_prediction_from_logits(
            logits_np, properties, config_3d,
            predictor.plans_manager, predictor.dataset_json,
            str(nii_out), save_probabilities=True,
        )

        dt = time.time() - t0
        print(f"-> {dt:.1f}s")
        del logits, logits_np, data_tensor, data
        torch.cuda.empty_cache()

    # Clean up predictor to free VRAM before next fold
    del predictor
    torch.cuda.empty_cache()

    dt_total = time.time() - t0_total
    have_nii = len(list(out_dir.glob("*.nii.gz")))
    print(f"[DONE] {out_dir_name}: {have_nii}/{total} in {dt_total/60:.1f} min")


def run_foldall_inference(device, images_dir, out_dir_name, ckpt="checkpoint_final.pth"):
    """Run fold_all 2.5D inference (checkpoint_final.pth)."""
    out_dir = PRED_BASE / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    case_files = sorted(images_dir.glob("*_0000.nii.gz"))
    total = len(case_files)

    have_nii = len(list(out_dir.glob("*.nii.gz")))
    if have_nii >= total:
        print(f"[SKIP] {out_dir_name} already complete ({have_nii} nii)")
        return

    print(f"\n{'='*60}")
    print(f"  2.5D fold_all — TTA on {total} cases  [{out_dir_name}]")
    print(f"{'='*60}")

    predictor = _build_predictor(("all",), ckpt, device)
    config_3d = predictor.plans_manager.get_configuration("3d_fullres")
    n_channels = len(predictor.dataset_json["channel_names"])

    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from nnunetv2.inference.export_prediction import export_prediction_from_logits

    reader = SimpleITKIO()
    preprocessor = DefaultPreprocessor()

    t0_total = time.time()
    for ci, ch0_file in enumerate(case_files):
        case_id = ch0_file.name.replace("_0000.nii.gz", "")
        nii_out = out_dir / f"{case_id}.nii.gz"
        npz_out = out_dir / f"{case_id}.nii.gz.npz"

        if nii_out.exists() and npz_out.exists():
            continue

        channel_files = [images_dir / f"{case_id}_{ch:04d}.nii.gz" for ch in range(n_channels)]
        channel_files = [f for f in channel_files if f.exists()]
        print(f"  [{ci+1}/{total}] {case_id}", end=" ", flush=True)
        t0 = time.time()

        images, properties = reader.read_images([str(f) for f in channel_files])
        data, seg, properties = preprocessor.run_case_npy(
            images, None, properties,
            predictor.plans_manager, config_3d, predictor.dataset_json,
        )
        data_tensor = torch.from_numpy(data).float()

        predictor.network.load_state_dict(predictor.list_of_parameters[0])
        logits = predictor.predict_sliding_window_return_logits(data_tensor.to(device))
        logits_np = logits.float().cpu().numpy()

        export_prediction_from_logits(
            logits_np, properties, config_3d,
            predictor.plans_manager, predictor.dataset_json,
            str(nii_out), save_probabilities=True,
        )

        dt = time.time() - t0
        print(f"-> {dt:.1f}s")
        del logits, logits_np, data_tensor, data
        torch.cuda.empty_cache()

    del predictor
    torch.cuda.empty_cache()

    dt_total = time.time() - t0_total
    have_nii = len(list(out_dir.glob("*.nii.gz")))
    print(f"[DONE] {out_dir_name}: {have_nii}/{total} in {dt_total/60:.1f} min")


def build_ensemble(fold_dir_names, out_dir_name, ref_fold_dir, thresholds):
    """Average softmax across folds, threshold, save NIfTIs."""
    case_ids = [f.name[:-len(".nii.gz")]
                for f in sorted((PRED_BASE / ref_fold_dir).glob("*.nii.gz")
                                if (PRED_BASE / ref_fold_dir).exists() else [])]

    if not case_ids:
        # Fall back: derive case IDs from NPZ files in first fold dir
        first_dir = PRED_BASE / fold_dir_names[0]
        case_ids = sorted(p.name.replace(".nii.gz.npz", "")
                          for p in first_dir.glob("*.nii.gz.npz"))

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
            probs = np.load(str(npz_path))["probabilities"]  # [C, z, y, x]
            probs_list.append(probs)

        if len(probs_list) != len(fold_dir_names):
            continue

        avg_probs = np.mean(probs_list, axis=0)

        # Reference NIfTI for header/affine
        ref_nii = nib.load(str(PRED_BASE / ref_fold_dir / f"{case_id}.nii.gz"))

        for thr in thresholds:
            if avg_probs.shape[0] == 2:
                seg = (avg_probs[1] >= thr).astype(np.uint8)
            else:
                seg = np.argmax(avg_probs, axis=0).astype(np.uint8)

            # nnU-Net NPZ is (z, y, x); NIfTI expects (x, y, z)
            seg = seg.transpose(2, 1, 0)

            suffix = f"_thr{thr:.2f}" if thr != 0.5 else ""
            thr_dir = PRED_BASE / f"{out_dir_name}{suffix}"
            thr_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(seg, ref_nii.affine, ref_nii.header),
                     str(thr_dir / f"{case_id}.nii.gz"))

        print(f"  {case_id} done")

    print(f"[DONE] Ensemble {out_dir_name} built")


def evaluate_dir(pred_dir, labels_dir, label=""):
    """Evaluate all predictions in pred_dir against ground truth in labels_dir."""
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
    device = torch.device("cuda")

    # ===============================================================
    #  DS001 — MSLesSeg (22 test cases)
    # ===============================================================
    print("\n" + "="*60)
    print("  DS001 — MSLesSeg per-fold inference")
    print("="*60)
    for fold in FOLDS:
        run_fold_inference(
            fold, device,
            images_dir=DS001_IMAGES,
            out_dir_name=f"25D_fold{fold}_TTA",
            force=(fold in DS001_FORCE_FOLDS),
        )

    ds001_fold_dirs = [f"25D_fold{f}_TTA" for f in FOLDS]
    build_ensemble(ds001_fold_dirs, "25D_5fold_ensemble_TTA",
                   ref_fold_dir="25D_fold0_TTA",
                   thresholds=[0.30, 0.40, 0.50])

    # ===============================================================
    #  DS002 — WMH (110 test cases)
    # ===============================================================
    print("\n" + "="*60)
    print("  DS002 — WMH per-fold inference")
    print("="*60)
    for fold in FOLDS:
        run_fold_inference(
            fold, device,
            images_dir=DS002_IMAGES,
            out_dir_name=f"25D_fold{fold}_TTA_DS002",
        )

    ds002_fold_dirs = [f"25D_fold{f}_TTA_DS002" for f in FOLDS]
    build_ensemble(ds002_fold_dirs, "25D_5fold_ensemble_TTA_DS002",
                   ref_fold_dir="25D_fold0_TTA_DS002",
                   thresholds=[0.30, 0.40, 0.50])

    # ===============================================================
    #  fold_all inference — DS001 + DS002
    # ===============================================================
    print("\n" + "="*60)
    print("  fold_all inference")
    print("="*60)
    run_foldall_inference(device, DS001_IMAGES, "25D_foldall_best_TTA", ckpt="checkpoint_best.pth")
    run_foldall_inference(device, DS002_IMAGES, "25D_foldall_best_TTA_DS002", ckpt="checkpoint_best.pth")

    # ===============================================================
    #  EVALUATION
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"  EVALUATION — DS001 MSLesSeg (22 cases)")
    print(f"{'='*60}")
    for fold in FOLDS:
        evaluate_dir(PRED_BASE / f"25D_fold{fold}_TTA", DS001_LABELS,
                     f"2.5D fold {fold} (DS001)")
    for thr in [0.50, 0.40, 0.30]:
        suffix = f"_thr{thr:.2f}" if thr != 0.5 else ""
        evaluate_dir(PRED_BASE / f"25D_5fold_ensemble_TTA{suffix}", DS001_LABELS,
                     f"2.5D 5-fold ens thr={thr:.2f} (DS001)")
    evaluate_dir(PRED_BASE / "25D_foldall_TTA", DS001_LABELS,
                 "2.5D fold_all final ckpt (DS001)")
    evaluate_dir(PRED_BASE / "25D_foldall_best_TTA", DS001_LABELS,
                 "2.5D fold_all best ckpt (DS001)")

    print(f"\n{'='*60}")
    print(f"  EVALUATION — DS002 WMH (110 cases)")
    print(f"{'='*60}")
    for fold in FOLDS:
        evaluate_dir(PRED_BASE / f"25D_fold{fold}_TTA_DS002", DS002_LABELS,
                     f"2.5D fold {fold} (DS002)")
    for thr in [0.50, 0.40, 0.30]:
        suffix = f"_thr{thr:.2f}" if thr != 0.5 else ""
        evaluate_dir(PRED_BASE / f"25D_5fold_ensemble_TTA_DS002{suffix}", DS002_LABELS,
                     f"2.5D 5-fold ens thr={thr:.2f} (DS002)")
    evaluate_dir(PRED_BASE / "25D_foldall_TTA_DS002", DS002_LABELS,
                 "2.5D fold_all final ckpt (DS002)")
    evaluate_dir(PRED_BASE / "25D_foldall_best_TTA_DS002", DS002_LABELS,
                 "2.5D fold_all best ckpt (DS002)")

    print(f"\n{'='*60}")
    print(f"  Reference: CNN+ResEncL ensemble DS001=0.7161, DS002=0.8015")
    print(f"  ResEncL fold_all DS001=0.6950, DS002=0.7898")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
