#!/usr/bin/env python3
"""
run_xai_experimental.py
========================
Experimental XAI methods for personal exploration (NOT thesis figures).
Implements three additional explainability methods:

  1. Integrated Gradients — axiomatic attribution (Sundararajan et al., 2017)
  2. SmoothGrad          — noise-averaged saliency (Smilkov et al., 2017)
  3. LRP (Layer-wise Relevance Propagation) — via zennit library

Usage:
    python run_xai_experimental.py                             # all 3 methods, 3 representative subjects
    python run_xai_experimental.py --methods intgrad           # Integrated Gradients only
    python run_xai_experimental.py --methods smoothgrad        # SmoothGrad only
    python run_xai_experimental.py --methods lrp               # LRP only
    python run_xai_experimental.py --max-subjects 1            # single subject
    python run_xai_experimental.py --models cnn                # CNN only
    python run_xai_experimental.py --subjects MSL_0094 MSL_0097 MSL_0100
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("nnUNet_raw", str(PROJECT_ROOT / "data" / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(PROJECT_ROOT / "data" / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(PROJECT_ROOT / "data" / "nnUNet_results"))

import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("xai_experimental")


# ═══════════════════════════════════════════════════════════════════════════════
# Reuse infra from existing XAI runner
# ═══════════════════════════════════════════════════════════════════════════════
from run_xai_analysis import (  # noqa: E402
    MODELS, RAW_BASE, RESULTS_BASE, OUTPUT_BASE,
    TEST_DATASET_ID, TEST_DATASET_NAME,
    load_model, get_test_subjects, load_volume, load_label,
    make_brain_mask, find_best_slices, save_heatmap_nifti,
)

EXPERIMENT_OUT = PROJECT_ROOT / "results" / "xai_experimental"

# Representative subjects (high lesion load, used in thesis XAI figures)
REPRESENTATIVE = ["MSL_0094", "MSL_0097", "MSL_0100"]


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Integrated Gradients (Sundararajan et al. 2017)
# ═══════════════════════════════════════════════════════════════════════════════
def run_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
    n_steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution for 3D segmentation.

    Interpolates from a baseline (default: zero tensor) to the input along
    a straight path, accumulating gradients at each step.

    Returns (D, H, W) heatmap in [0, 1].
    """
    device = input_tensor.device

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Precompute the interpolation delta
    delta = input_tensor - baseline  # (1, 2, D, H, W)

    accumulated_grads = torch.zeros_like(input_tensor)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * delta
        interpolated = interpolated.clone().detach().requires_grad_(True)

        output = model(interpolated)
        if isinstance(output, (list, tuple)):
            output = output[0]

        # Sum lesion class logits across all spatial locations
        target_score = output[0, target_class].sum()
        target_score.backward()

        accumulated_grads += interpolated.grad.detach()

        model.zero_grad()

    # Average gradients and multiply by delta (path integral approximation)
    avg_grads = accumulated_grads / (n_steps + 1)
    attributions = (delta * avg_grads).squeeze(0)  # (2, D, H, W)

    # Sum across both input channels to get single attribution map
    attr_map = attributions.sum(dim=0).cpu().numpy()  # (D, H, W)

    # Take absolute value (both positive and negative attributions matter)
    attr_map = np.abs(attr_map)

    # Normalize to [0, 1]
    a_min, a_max = attr_map.min(), attr_map.max()
    if a_max - a_min > 1e-8:
        attr_map = (attr_map - a_min) / (a_max - a_min)
    else:
        attr_map = np.zeros_like(attr_map)

    return attr_map.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: SmoothGrad (Smilkov et al. 2017)
# ═══════════════════════════════════════════════════════════════════════════════
def run_smoothgrad(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
    n_samples: int = 50,
    noise_level: float = 0.15,
) -> np.ndarray:
    """
    Compute SmoothGrad saliency map for 3D segmentation.

    Averages vanilla gradient saliency over n_samples noisy copies of the
    input. The noise standard deviation is noise_level * (input max - input min).

    Returns (D, H, W) heatmap in [0, 1].
    """
    device = input_tensor.device

    # Scale noise to the input range
    sigma = noise_level * (input_tensor.max() - input_tensor.min()).item()

    accumulated_grads = torch.zeros_like(input_tensor)

    for i in range(n_samples):
        noisy_input = input_tensor + sigma * torch.randn_like(input_tensor)
        noisy_input = noisy_input.clone().detach().requires_grad_(True)

        output = model(noisy_input)
        if isinstance(output, (list, tuple)):
            output = output[0]

        target_score = output[0, target_class].sum()
        target_score.backward()

        accumulated_grads += noisy_input.grad.detach()

        model.zero_grad()

    avg_grads = accumulated_grads / n_samples  # (1, 2, D, H, W)

    # Sum channels, take absolute value
    saliency = avg_grads.squeeze(0).sum(dim=0).cpu().numpy()  # (D, H, W)
    saliency = np.abs(saliency)

    # Normalize
    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 1e-8:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: LRP (Layer-wise Relevance Propagation) via zennit
# ═══════════════════════════════════════════════════════════════════════════════
def _check_zennit():
    """Check if zennit is installed."""
    try:
        import zennit  # noqa: F401
        return True
    except ImportError:
        return False


def run_lrp(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """
    Compute LRP attribution using zennit's EpsilonPlusFlat composite.

    This composite applies:
      - Epsilon rule for dense/conv layers (numerically stable)
      - ZPlus rule for lower layers (positive contributions only)
      - Flat rule for the first layer

    Returns (D, H, W) heatmap in [0, 1].
    """
    from zennit.composites import EpsilonPlusFlat
    from zennit.attribution import Gradient

    device = input_tensor.device

    inp = input_tensor.clone().detach().requires_grad_(True)

    composite = EpsilonPlusFlat()

    with Gradient(model=model, composite=composite) as attributor:
        output, attribution = attributor(inp, attr_output=None)

        if isinstance(output, (list, tuple)):
            output = output[0]

        # Create a one-hot-like output target for the lesion class
        target_output = torch.zeros_like(output)
        target_output[0, target_class] = output[0, target_class].detach()

        # Backward with the target
        output.backward(gradient=target_output)

        # Attribution is in inp.grad
        relevance = inp.grad.detach()  # (1, 2, D, H, W)

    # Sum channels, absolute value
    attr_map = relevance.squeeze(0).sum(dim=0).cpu().numpy()
    attr_map = np.abs(attr_map)

    # Normalize
    a_min, a_max = attr_map.min(), attr_map.max()
    if a_max - a_min > 1e-8:
        attr_map = (attr_map - a_min) / (a_max - a_min)
    else:
        attr_map = np.zeros_like(attr_map)

    return attr_map.astype(np.float32)


def run_lrp_manual(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Fallback manual LRP-epsilon via gradient × input trick.

    When zennit is not available, this approximates LRP using the
    input × gradient formulation (equivalent to LRP-0 for ReLU networks).

    Returns (D, H, W) heatmap in [0, 1].
    """
    inp = input_tensor.clone().detach().requires_grad_(True)

    output = model(inp)
    if isinstance(output, (list, tuple)):
        output = output[0]

    target_score = output[0, target_class].sum()
    target_score.backward()

    # Gradient × Input approximation
    relevance = (inp.grad * inp).detach()  # (1, 2, D, H, W)

    model.zero_grad()

    attr_map = relevance.squeeze(0).sum(dim=0).cpu().numpy()
    attr_map = np.abs(attr_map)

    a_min, a_max = attr_map.min(), attr_map.max()
    if a_max - a_min > 1e-8:
        attr_map = (attr_map - a_min) / (a_max - a_min)
    else:
        attr_map = np.zeros_like(attr_map)

    return attr_map.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization — side-by-side comparison panel
# ═══════════════════════════════════════════════════════════════════════════════
def save_experimental_panel(
    flair: np.ndarray,
    label: Optional[np.ndarray],
    heatmaps: Dict[str, np.ndarray],
    subject_id: str,
    slice_indices: List[int],
    output_dir: Path,
    model_key: str,
) -> Path:
    """
    Save comparison panel: FLAIR | GT | method1 | method2 | method3
    One row per axial slice.
    """
    n_slices = len(slice_indices)
    has_label = label is not None
    n_cols = (2 if has_label else 1) + len(heatmaps)
    method_names = list(heatmaps.keys())

    fig, axes = plt.subplots(n_slices, n_cols, figsize=(4 * n_cols, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row, sl in enumerate(slice_indices):
        col = 0

        flair_sl = flair[:, :, sl].T
        axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
        if row == 0:
            axes[row, col].set_title("FLAIR", fontsize=11)
        axes[row, col].set_ylabel(f"z={sl}", fontsize=9)
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        col += 1

        if has_label:
            axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
            lbl_sl = label[:, :, sl].T.astype(float)
            axes[row, col].imshow(lbl_sl, cmap="Reds", alpha=0.5, origin="lower",
                                  vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title("Ground Truth", fontsize=11)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            col += 1

        for method_name in method_names:
            hmap = heatmaps[method_name]
            hmap_sl = hmap[:, :, sl].T

            axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
            axes[row, col].imshow(hmap_sl, cmap="jet", alpha=0.5,
                                  origin="lower", vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(method_name, fontsize=11)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            col += 1

    fig.suptitle(f"{subject_id} — {model_key} — Experimental XAI", fontsize=14, y=1.01)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{subject_id}_{model_key}_experimental_xai.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def save_all_methods_comparison(
    flair: np.ndarray,
    label: Optional[np.ndarray],
    new_heatmaps: Dict[str, np.ndarray],
    existing_heatmaps: Dict[str, np.ndarray],
    subject_id: str,
    slice_indices: List[int],
    output_dir: Path,
    model_key: str,
) -> Path:
    """
    Save a 6-method mega-panel: FLAIR | GT | GradCAM | RISE | IntGrad | SmoothGrad | LRP
    """
    all_maps = {}
    all_maps.update(existing_heatmaps)
    all_maps.update(new_heatmaps)

    n_slices = len(slice_indices)
    has_label = label is not None
    n_cols = (2 if has_label else 1) + len(all_maps)
    method_names = list(all_maps.keys())

    fig, axes = plt.subplots(n_slices, n_cols, figsize=(3.5 * n_cols, 3.5 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row, sl in enumerate(slice_indices):
        col = 0

        flair_sl = flair[:, :, sl].T
        axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
        if row == 0:
            axes[row, col].set_title("FLAIR", fontsize=10)
        axes[row, col].set_ylabel(f"z={sl}", fontsize=8)
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        col += 1

        if has_label:
            axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
            lbl_sl = label[:, :, sl].T.astype(float)
            axes[row, col].imshow(lbl_sl, cmap="Reds", alpha=0.5, origin="lower",
                                  vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title("GT", fontsize=10)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            col += 1

        for method_name in method_names:
            hmap = all_maps[method_name]
            hmap_sl = hmap[:, :, sl].T

            axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
            axes[row, col].imshow(hmap_sl, cmap="jet", alpha=0.5,
                                  origin="lower", vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(method_name, fontsize=10)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            col += 1

    fig.suptitle(f"{subject_id} — {model_key} — All XAI Methods", fontsize=13, y=1.01)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{subject_id}_{model_key}_all_xai_methods.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Experimental XAI methods (personal exploration)")
    parser.add_argument("--methods", nargs="+", default=["intgrad", "smoothgrad", "lrp"],
                        choices=["intgrad", "smoothgrad", "lrp"],
                        help="Methods to run (default: all three)")
    parser.add_argument("--models", nargs="+", default=["resencl"],
                        choices=list(MODELS.keys()),
                        help="Models (default: resencl)")
    parser.add_argument("--subjects", nargs="+", default=REPRESENTATIVE,
                        help="Subject IDs to process (default: 3 representative)")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ig-steps", type=int, default=50,
                        help="Integrated Gradients interpolation steps (default: 50)")
    parser.add_argument("--sg-samples", type=int, default=50,
                        help="SmoothGrad noise samples (default: 50)")
    parser.add_argument("--sg-noise", type=float, default=0.15,
                        help="SmoothGrad noise level (default: 0.15)")
    parser.add_argument("--n-slices", type=int, default=5)
    parser.add_argument("--no-mega-panel", action="store_true",
                        help="Skip the combined old+new methods panel")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Methods: %s", ", ".join(args.methods))
    logger.info("Models: %s", ", ".join(args.models))

    # Check zennit availability
    has_zennit = _check_zennit()
    if "lrp" in args.methods:
        if has_zennit:
            logger.info("zennit available — using EpsilonPlusFlat composite")
        else:
            logger.warning("zennit not installed — falling back to gradient×input LRP approximation")

    # Get all test subjects, filter to requested ones
    all_subjects = get_test_subjects()
    subject_dict = {s[0]: s for s in all_subjects}

    subjects = []
    for sid in args.subjects:
        if sid in subject_dict:
            subjects.append(subject_dict[sid])
        else:
            logger.warning("Subject %s not found, skipping", sid)

    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    logger.info("Subjects: %s", [s[0] for s in subjects])

    for model_key in args.models:
        model_cfg = MODELS[model_key]
        logger.info("\n" + "=" * 70)
        logger.info("  Model: %s", model_cfg["label"])
        logger.info("=" * 70)

        model = load_model(model_key, device)
        model_out = EXPERIMENT_OUT / model_key

        for subj_idx, (subject_id, flair_path, t1_path, label_path) in enumerate(subjects):
            logger.info("\n  [%d/%d] Subject: %s", subj_idx + 1, len(subjects), subject_id)

            input_tensor, raw_flair, raw_t1, crop_slices = load_volume(
                flair_path, t1_path, device)
            label = load_label(label_path)
            if label is not None:
                label = label[crop_slices]
            logger.info("    Volume shape: %s", list(input_tensor.shape))

            flair_crop = raw_flair[crop_slices]
            brain_mask = make_brain_mask(flair_crop).astype(np.float32)

            new_heatmaps: Dict[str, np.ndarray] = {}

            # --- Integrated Gradients ---
            if "intgrad" in args.methods:
                logger.info("    Running Integrated Gradients (%d steps)...", args.ig_steps)
                t0 = time.time()
                with torch.enable_grad():
                    ig_map = run_integrated_gradients(
                        model, input_tensor, n_steps=args.ig_steps)
                dt = time.time() - t0
                logger.info("    Integrated Gradients done (%.1fs)", dt)
                ig_map *= brain_mask
                new_heatmaps["IntGrad"] = ig_map
                save_heatmap_nifti(ig_map, flair_path,
                                   model_out / "nifti" / f"{subject_id}_intgrad.nii.gz")

            # --- SmoothGrad ---
            if "smoothgrad" in args.methods:
                logger.info("    Running SmoothGrad (%d samples, noise=%.2f)...",
                            args.sg_samples, args.sg_noise)
                t0 = time.time()
                with torch.enable_grad():
                    sg_map = run_smoothgrad(
                        model, input_tensor,
                        n_samples=args.sg_samples, noise_level=args.sg_noise)
                dt = time.time() - t0
                logger.info("    SmoothGrad done (%.1fs)", dt)
                sg_map *= brain_mask
                new_heatmaps["SmoothGrad"] = sg_map
                save_heatmap_nifti(sg_map, flair_path,
                                   model_out / "nifti" / f"{subject_id}_smoothgrad.nii.gz")

            # --- LRP ---
            if "lrp" in args.methods:
                logger.info("    Running LRP...")
                t0 = time.time()
                with torch.enable_grad():
                    if has_zennit:
                        lrp_map = run_lrp(model, input_tensor)
                    else:
                        lrp_map = run_lrp_manual(model, input_tensor)
                dt = time.time() - t0
                logger.info("    LRP done (%.1fs)", dt)
                lrp_map *= brain_mask
                new_heatmaps["LRP"] = lrp_map
                save_heatmap_nifti(lrp_map, flair_path,
                                   model_out / "nifti" / f"{subject_id}_lrp.nii.gz")

            # --- Experimental methods panel ---
            if new_heatmaps:
                slices = find_best_slices(label, flair_crop, n_slices=args.n_slices)
                png_path = save_experimental_panel(
                    flair_crop, label, new_heatmaps, subject_id, slices,
                    model_out / "png", model_key)
                logger.info("    Saved experimental panel: %s", png_path)

                # --- Mega panel (old + new methods together) ---
                if not args.no_mega_panel:
                    existing_heatmaps: Dict[str, np.ndarray] = {}
                    existing_dir = OUTPUT_BASE / model_key / "nifti"

                    for method_name, file_suffix in [
                        ("Grad-CAM", "gradcam"),
                        ("RISE", "rise"),
                    ]:
                        nifti_path = existing_dir / f"{subject_id}_{file_suffix}.nii.gz"
                        if nifti_path.exists():
                            hmap = np.asarray(nib.load(str(nifti_path)).dataobj,
                                              dtype=np.float32)
                            # Crop to match
                            hmap = hmap[crop_slices]
                            # Pad if needed
                            if hmap.shape != flair_crop.shape:
                                pad_widths = []
                                for dim in range(3):
                                    target = flair_crop.shape[dim]
                                    current = hmap.shape[dim]
                                    if current < target:
                                        before = (target - current) // 2
                                        after = target - current - before
                                        pad_widths.append((before, after))
                                    else:
                                        pad_widths.append((0, 0))
                                hmap = np.pad(hmap, pad_widths, mode="constant")
                            existing_heatmaps[method_name] = hmap
                            logger.info("    Loaded existing %s heatmap", method_name)

                    if existing_heatmaps:
                        mega_path = save_all_methods_comparison(
                            flair_crop, label, new_heatmaps, existing_heatmaps,
                            subject_id, slices, model_out / "png", model_key)
                        logger.info("    Saved mega panel: %s", mega_path)

            # Cleanup
            del input_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("\n" + "=" * 70)
    logger.info("Experimental XAI complete. Output: %s", EXPERIMENT_OUT)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
