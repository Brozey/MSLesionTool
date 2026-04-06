#!/usr/bin/env python3
"""
msseg.cli – Command-line interface for MS lesion segmentation.

Usage:
    python -m msseg.cli --flair scan_flair.nii.gz --t1 scan_t1.nii.gz -o output.nii.gz
    python -m msseg.cli --flair f.nii.gz --t1 t.nii.gz -o seg.nii.gz --backend onnx
    python -m msseg.cli --batch /data/subjects/ --output-dir /data/results/
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from .constants import ARCHITECTURES, DEFAULT_BEST2, resolve_model_dir, _APP_DIR
from .io import load_nifti, write_nifti


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("msseg.cli")


def _find_model_root():
    """Search for the msseg/ model directory."""
    candidates = [_APP_DIR, os.getcwd()]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'msseg', 'cnn3d')):
            return c
    return None


def run_segmentation(flair_path, t1_path, output_path, backend="pytorch",
                     device="cuda", ensemble="best2", save_probs=False):
    """Run ensemble segmentation and save results."""
    log.info("Loading FLAIR: %s", flair_path)
    flair_arr, _, spacing, sitk_ref = load_nifti(flair_path)

    log.info("Loading T1: %s", t1_path)
    t1_arr, _, _, _ = load_nifti(t1_path)

    flair_arr = flair_arr.astype(np.float32)
    t1_arr = t1_arr.astype(np.float32)

    # Build ensemble config
    if ensemble == "best2":
        ensemble_config = DEFAULT_BEST2
    elif ensemble == "all5":
        ensemble_config = {k: tuple(range(5)) for k in ARCHITECTURES}
    else:
        raise ValueError(f"Unknown ensemble mode: {ensemble}")

    img = np.stack([flair_arr, t1_arr], axis=0).astype(np.float32)
    props = {'spacing': list(spacing)}

    all_probs = []
    total = sum(len(folds) for folds in ensemble_config.values())
    done = 0

    for arch_key, folds in ensemble_config.items():
        subdir = ARCHITECTURES[arch_key][0]
        display = ARCHITECTURES[arch_key][2]
        model_dir = resolve_model_dir(subdir)

        if model_dir is None:
            log.warning("%s model not found, skipping", display)
            continue

        for fold in folds:
            done += 1

            if backend == "onnx":
                from .inference_ort import create_ort_predictor, check_onnx_available
                if not check_onnx_available(model_dir, fold):
                    log.warning("ONNX model not found for %s fold %d, skipping", display, fold)
                    continue
                ort_device = "onnx-cuda" if device == "cuda" else "onnx-cpu"
                log.info("[%d/%d] %s fold %d (ONNX, %s)", done, total, display, fold, ort_device)
                predictor, _ = create_ort_predictor(
                    arch_key, model_dir, folds=(fold,), device_str=ort_device)
            else:
                from .inference import create_predictor
                log.info("[%d/%d] %s fold %d (PyTorch, %s)", done, total, display, fold, device)
                predictor, device = create_predictor(arch_key, model_dir, device, folds=(fold,))
                if device == "cuda":
                    predictor.network.half()
                else:
                    predictor.network.float()

            _seg, probs = predictor.predict_single_npy_array(
                img, props, None, None, True)
            all_probs.append(probs)

    if not all_probs:
        log.error("No models produced predictions!")
        return

    # Ensemble averaging
    prob_stack = np.stack(all_probs, axis=0)
    mean_prob = np.mean(prob_stack, axis=0)

    # Handle multi-class → binary merge
    if mean_prob.shape[0] > 2:
        bg = mean_prob[0:1]
        fg = np.sum(mean_prob[1:], axis=0, keepdims=True)
        mean_prob = np.concatenate([bg, fg], axis=0)

    seg = np.argmax(mean_prob, axis=0).astype(np.uint8)

    # Stats
    voxel_vol = float(np.prod(spacing))
    lesion_voxels = int(np.sum(seg > 0))
    lesion_vol_mm3 = lesion_voxels * voxel_vol
    log.info("Lesion voxels: %d (%.1f mm³ / %.2f mL)",
             lesion_voxels, lesion_vol_mm3, lesion_vol_mm3 / 1000)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    import SimpleITK as sitk
    write_nifti(seg, output_path, ref_image=sitk_ref, spacing=spacing)
    log.info("Saved segmentation: %s", output_path)

    if save_probs:
        prob_path = output_path.replace('.nii', '_probs.nii')
        prob_fg = (mean_prob[1] * 1000).astype(np.int16)
        write_nifti(prob_fg, prob_path, ref_image=sitk_ref, spacing=spacing)
        log.info("Saved probabilities: %s", prob_path)

    # Save stats JSON
    stats_path = output_path.replace('.nii.gz', '_stats.json').replace('.nii', '_stats.json')
    stats = {
        "lesion_voxels": lesion_voxels,
        "lesion_volume_mm3": round(lesion_vol_mm3, 2),
        "lesion_volume_mL": round(lesion_vol_mm3 / 1000, 4),
        "spacing": list(spacing),
        "shape": list(seg.shape),
        "ensemble": ensemble,
        "backend": backend,
        "models": done,
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    log.info("Saved stats: %s", stats_path)


def main():
    parser = argparse.ArgumentParser(
        description="MSLesionTool – MS Lesion Segmentation (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--flair", help="Path to FLAIR NIfTI volume")
    group.add_argument("--batch", help="Directory containing patient subdirectories")

    parser.add_argument("--t1", help="Path to T1 NIfTI volume (required with --flair)")
    parser.add_argument("-o", "--output", help="Output segmentation path (with --flair)")
    parser.add_argument("--output-dir", help="Output directory (with --batch)")

    # Options
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch",
                        help="Inference backend (default: pytorch)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--ensemble", choices=["best2", "all5"], default="best2",
                        help="Ensemble mode (default: best2)")
    parser.add_argument("--save-probs", action="store_true",
                        help="Also save probability maps")

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        from .inference import detect_best_device
        args.device = detect_best_device()
        log.info("Auto-detected device: %s", args.device)

    t0 = time.time()

    if args.flair:
        if not args.t1:
            parser.error("--t1 is required when using --flair")
        output = args.output or args.flair.replace('.nii', '_seg.nii')
        run_segmentation(
            args.flair, args.t1, output,
            backend=args.backend, device=args.device,
            ensemble=args.ensemble, save_probs=args.save_probs,
        )
    else:
        # Batch mode
        if not args.output_dir:
            parser.error("--output-dir is required when using --batch")
        os.makedirs(args.output_dir, exist_ok=True)

        from .io import find_nifti_files_recursive, auto_assign_sequences
        patients = sorted(d for d in os.listdir(args.batch)
                          if os.path.isdir(os.path.join(args.batch, d)))

        log.info("Found %d patient directories", len(patients))
        for i, patient in enumerate(patients):
            log.info("="*60)
            log.info("[%d/%d] Processing: %s", i+1, len(patients), patient)
            pdir = os.path.join(args.batch, patient)
            niftis = find_nifti_files_recursive(pdir)
            assigned = auto_assign_sequences(niftis)
            if not assigned["FLAIR"] or not assigned["T1"]:
                log.warning("  Could not find FLAIR+T1, skipping")
                continue
            output = os.path.join(args.output_dir, f"{patient}_seg.nii.gz")
            run_segmentation(
                assigned["FLAIR"], assigned["T1"], output,
                backend=args.backend, device=args.device,
                ensemble=args.ensemble, save_probs=args.save_probs,
            )

    elapsed = time.time() - t0
    log.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
