#!/usr/bin/env python3
"""
master_inference.py
===================
Master inference pipeline for MS lesion segmentation.

Supports three modes:

  1. **from-softmax** (single model): Threshold pre-computed softmax maps.
  2. **full-pipeline** (single model): Run nnU-Net inference + threshold.
  3. **ensemble**: Average softmax from multiple model directories, then threshold.

Optimal configurations determined from systematic experiments:

  Single model (DS001/DS002 ResEncL, TTA):
      threshold 0.30 → DS001 DSC 0.6814, DS002 DSC 0.7793

  Ensemble (dataset-specific + DS003 combined model, TTA):
      threshold 0.45 → DS001 DSC 0.7026 (+3.1%), DS002 DSC 0.7951 (+2.7%)

  DS003 model alone (TTA):
      threshold 0.50 → DS001 DSC 0.7026, DS002 DSC 0.7923

Usage:
    # Single model — from cached softmax (no GPU):
    python scripts/inference/master_inference.py --from-softmax \\
        --softmax-dir predictions/DS001_ResEncL_3D_TTA \\
        --output-dir predictions/DS001_final

    # Ensemble — average softmax from two models:
    python scripts/inference/master_inference.py --ensemble \\
        --softmax-dirs predictions/DS001_ResEncL_3D_TTA \\
                       predictions/DS001_DS003_ResEncL_3D_TTA \\
        --output-dir predictions/DS001_ensemble \\
        --threshold 0.45

    # Full pipeline (runs nnU-Net inference):
    python scripts/inference/master_inference.py \\
        --dataset Dataset001_MSLesSeg \\
        --input-dir data/nnUNet_raw/Dataset001_MSLesSeg/imagesTs \\
        --output-dir predictions/DS001_final \\
        --model-dir data/nnUNet_results/Dataset001_MSLesSeg/...

Author: MS Lesion Segmentation Pipeline
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.30   # Optimal for single model
ENSEMBLE_THRESHOLD = 0.45  # Optimal for ensemble
SOFTMAX_AXIS_PERM = (2, 1, 0)  # nnU-Net NPZ (z,y,x) → NIfTI (x,y,z)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing from softmax
# ──────────────────────────────────────────────────────────────────────────────

def postprocess_softmax(
    softmax_dir: Path,
    output_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    ref_nifti_dir: Path | None = None,
) -> list[Path]:
    """Apply threshold to softmax maps and save as NIfTI.

    Args:
        softmax_dir: Dir with .nii.gz predictions + .npz softmax maps
        output_dir: Where to save final binary predictions
        threshold: Probability threshold for foreground class
        ref_nifti_dir: If set, use these NIfTI headers (otherwise use pred .nii.gz)

    Returns:
        List of output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(softmax_dir.glob("*.nii.gz"))
    if not nii_files:
        print(f"  ERROR: No .nii.gz files in {softmax_dir}")
        return []

    outputs = []
    for nii_path in nii_files:
        npz_path = softmax_dir / nii_path.name.replace(".nii.gz", ".npz")

        if not npz_path.exists():
            # No softmax — fall back to hard prediction (copy as-is)
            print(f"  [WARN] No softmax for {nii_path.name}, using hard prediction")
            out_path = output_dir / nii_path.name
            shutil.copy2(nii_path, out_path)
            outputs.append(out_path)
            continue

        # Load reference NIfTI for header/affine
        ref_path = nii_path
        if ref_nifti_dir and (ref_nifti_dir / nii_path.name).exists():
            ref_path = ref_nifti_dir / nii_path.name

        ref_nii = nib.load(str(ref_path))

        # Load softmax, transpose to NIfTI orientation, threshold
        npz_data = np.load(str(npz_path))
        key = "softmax" if "softmax" in npz_data else "probabilities"
        fg_prob = npz_data[key][1]  # foreground probability
        fg_prob = np.transpose(fg_prob, SOFTMAX_AXIS_PERM)

        assert fg_prob.shape == ref_nii.shape, (
            f"Shape mismatch after transpose: softmax {fg_prob.shape} vs "
            f"NIfTI {ref_nii.shape} for {nii_path.name}"
        )

        binary_seg = (fg_prob >= threshold).astype(np.uint8)

        # Save with same header/affine as reference
        out_nii = nib.Nifti1Image(binary_seg, ref_nii.affine, ref_nii.header)
        out_path = output_dir / nii_path.name
        nib.save(out_nii, str(out_path))
        outputs.append(out_path)

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble from multiple softmax directories
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_softmax(
    softmax_dirs: list[Path],
    output_dir: Path,
    threshold: float = ENSEMBLE_THRESHOLD,
) -> list[Path]:
    """Average softmax probabilities from multiple model directories, threshold.

    Each directory should contain matching .nii.gz + .npz file pairs.
    The NIfTI from the first directory is used for header/affine reference.

    Args:
        softmax_dirs: List of directories containing .npz softmax maps
        output_dir: Where to save final binary predictions
        threshold: Probability threshold for foreground class

    Returns:
        List of output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find common cases across all softmax directories
    common_stems = None
    for sd in softmax_dirs:
        stems = {f.stem.replace('.nii', '') for f in sd.glob("*.npz")}
        common_stems = stems if common_stems is None else common_stems & stems
    common_stems = sorted(common_stems)

    if not common_stems:
        print(f"  ERROR: No common cases across {len(softmax_dirs)} directories")
        return []

    print(f"  Ensembling {len(softmax_dirs)} models, {len(common_stems)} cases")

    outputs = []
    for case_stem in common_stems:
        nii_name = case_stem + ".nii.gz"

        # Load reference NIfTI from first directory
        ref_nii_path = softmax_dirs[0] / nii_name
        if not ref_nii_path.exists():
            print(f"  [WARN] No NIfTI ref for {case_stem}, skipping")
            continue
        ref_nii = nib.load(str(ref_nii_path))

        # Average softmax across models
        avg_prob = None
        for sd in softmax_dirs:
            npz_path = sd / f"{case_stem}.npz"
            npz_data = np.load(str(npz_path))
            key = "softmax" if "softmax" in npz_data else "probabilities"
            fg_prob = npz_data[key][1].astype(np.float32)
            fg_prob = np.transpose(fg_prob, SOFTMAX_AXIS_PERM)

            if avg_prob is None:
                avg_prob = fg_prob
            else:
                avg_prob += fg_prob

        avg_prob /= len(softmax_dirs)

        assert avg_prob.shape == ref_nii.shape, (
            f"Shape mismatch: averaged softmax {avg_prob.shape} vs "
            f"NIfTI {ref_nii.shape} for {case_stem}"
        )

        binary_seg = (avg_prob >= threshold).astype(np.uint8)

        out_nii = nib.Nifti1Image(binary_seg, ref_nii.affine, ref_nii.header)
        out_path = output_dir / nii_name
        nib.save(out_nii, str(out_path))
        outputs.append(out_path)

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# Full inference pipeline (nnU-Net → softmax → threshold)
# ──────────────────────────────────────────────────────────────────────────────

def run_full_inference(
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    model_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    folds: str = "0",
    num_processes: int = 4,
) -> list[Path]:
    """Run nnU-Net inference with TTA, then apply softmax thresholding.

    This is the complete pipeline for new/unseen data.
    """
    import subprocess

    # Step 1: Run nnU-Net prediction WITH TTA and softmax
    tta_dir = output_dir.parent / f"{output_dir.name}_tta_softmax"
    tta_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Step 1: nnU-Net inference with TTA")
    print(f"    Input:  {input_dir}")
    print(f"    Model:  {model_dir}")
    print(f"    Output: {tta_dir}")

    # Ensure env vars are set
    env = os.environ.copy()
    env.setdefault("nnUNet_raw", str(Path(input_dir).parent.parent.parent))
    env.setdefault("nnUNet_results", str(model_dir.parent.parent.parent))

    cmd = [
        sys.executable, "-m", "nnunetv2.inference.predict_from_raw_data",
        "-i", str(input_dir),
        "-o", str(tta_dir),
        "-d", dataset,
        "-tr", "nnUNetTrainer_WandB",
        "-p", "nnUNetResEncUNetLPlans",
        "-c", "3d_fullres",
        "-f", folds,
        "--save_probabilities",
        "-npp", str(num_processes),
        # NO --disable_tta → TTA enabled
    ]

    print(f"    Command: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: nnU-Net inference failed (exit code {result.returncode})")
        return []
    print(f"    Inference completed in {time.time()-t0:.0f}s")

    # Step 2: Post-process softmax maps
    print(f"\n  Step 2: Softmax thresholding (threshold={threshold})")
    outputs = postprocess_softmax(tta_dir, output_dir, threshold)

    print(f"\n  Pipeline complete: {len(outputs)} predictions in {output_dir}")

    # Copy plans.json and dataset.json if present
    for meta in ["plans.json", "dataset.json"]:
        src = tta_dir / meta
        if src.exists():
            shutil.copy2(src, output_dir / meta)

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Master inference pipeline for MS lesion segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Softmax threshold (default: {DEFAULT_THRESHOLD} single, "
                             f"{ENSEMBLE_THRESHOLD} ensemble)")

    # Mode 1: From pre-computed softmax (single model)
    parser.add_argument("--from-softmax", action="store_true",
                        help="Use pre-computed TTA softmax maps (no GPU needed)")
    parser.add_argument("--softmax-dir", type=Path,
                        help="Directory with .nii.gz + .npz softmax maps")

    # Mode 2: Ensemble from multiple softmax directories
    parser.add_argument("--ensemble", action="store_true",
                        help="Average softmax from multiple model directories")
    parser.add_argument("--softmax-dirs", type=Path, nargs="+",
                        help="Multiple directories with .npz softmax maps")

    # Mode 3: Full pipeline
    parser.add_argument("--dataset", type=str,
                        help="nnU-Net dataset name (e.g. Dataset001_MSLesSeg)")
    parser.add_argument("--input-dir", type=Path,
                        help="Input images directory")
    parser.add_argument("--model-dir", type=Path,
                        help="nnU-Net model directory")
    parser.add_argument("--folds", type=str, default="0",
                        help="Fold(s) to use (default: 0)")

    # Common
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for final predictions")

    args = parser.parse_args()

    # Determine mode and default threshold
    if args.ensemble:
        mode = "ensemble"
        threshold = args.threshold if args.threshold is not None else ENSEMBLE_THRESHOLD
    elif args.from_softmax:
        mode = "from-softmax"
        threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    else:
        mode = "full-pipeline"
        threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD

    print("=" * 70)
    print("  MS Lesion Segmentation — Master Inference Pipeline")
    print(f"  Mode: {mode}")
    print(f"  Strategy: TTA + softmax threshold @ {threshold}")
    if mode == "ensemble":
        print(f"  Models: {len(args.softmax_dirs)} directories")
    print("=" * 70)

    t_start = time.time()

    if mode == "ensemble":
        if not args.softmax_dirs or len(args.softmax_dirs) < 2:
            parser.error("--ensemble requires --softmax-dirs with ≥2 directories")
        outputs = ensemble_softmax(
            args.softmax_dirs, args.output_dir, threshold)
    elif mode == "from-softmax":
        if not args.softmax_dir:
            parser.error("--from-softmax requires --softmax-dir")
        outputs = postprocess_softmax(
            args.softmax_dir, args.output_dir, threshold)
    else:
        if not all([args.dataset, args.input_dir, args.model_dir]):
            parser.error("Full pipeline requires --dataset, --input-dir, --model-dir")
        outputs = run_full_inference(
            args.dataset, args.input_dir, args.output_dir,
            args.model_dir, threshold, args.folds)

    elapsed = time.time() - t_start
    print(f"\n  Done: {len(outputs)} files in {elapsed:.1f}s")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
