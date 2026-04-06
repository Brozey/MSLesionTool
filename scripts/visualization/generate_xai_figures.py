#!/usr/bin/env python3
"""
Generate thesis-quality XAI figures from pre-computed heatmaps.

Produces:
  fig_gradcam_comparison.pdf  — GradCAM: ResEncL vs CNN side-by-side (3 subjects)
  fig_rise_heatmap.pdf        — 3D RISE saliency maps (3 subjects)
  fig_xai_methods_comparison.pdf — All 3 methods compared on same subject

Usage:
  python scripts/visualization/generate_xai_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nibabel as nib
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT
XAI_DIR = ROOT / "results" / "xai"
RAW_DIR = ROOT / "data" / "nnUNet_raw" / "Dataset001_MSLesSeg"
FIG_DIR = ROOT / "thesis_outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Subjects with high lesion load for best visuals
REPRESENTATIVE_SUBJECTS = ["MSL_0094", "MSL_0097", "MSL_0100"]
PATCH_SIZE = (160, 192, 160)


def load_and_crop(path: Path) -> np.ndarray:
    """Load NIfTI and center-crop to patch size."""
    data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    crop_slices = []
    for dim, ps in enumerate(PATCH_SIZE):
        s = data.shape[dim]
        if s >= ps:
            start = (s - ps) // 2
            crop_slices.append(slice(start, start + ps))
        else:
            crop_slices.append(slice(0, s))
    return data[tuple(crop_slices)]


def find_lesion_slices(label: np.ndarray, n: int = 3) -> list[int]:
    """Find axial slices with most lesion voxels."""
    sums = label.sum(axis=(0, 1))
    indices = np.argsort(sums)[::-1][:n]
    return sorted(indices)


def make_brain_mask(flair: np.ndarray, dilate_voxels: int = 3) -> np.ndarray:
    """Create a binary brain foreground mask from a FLAIR volume.

    Thresholds slightly above zero (mean - 2*std of nonzero voxels),
    then dilates a few voxels to avoid clipping at the brain edge.
    Returns a boolean mask with the same shape as *flair*.
    """
    nonzero = flair[flair > 0]
    if nonzero.size == 0:
        return np.ones(flair.shape, dtype=bool)
    thr = max(0, nonzero.mean() - 2.0 * nonzero.std())
    mask = flair > thr
    if dilate_voxels > 0:
        struct = ndimage.generate_binary_structure(mask.ndim, 1)
        mask = ndimage.binary_dilation(mask, struct, iterations=dilate_voxels)
    return mask


def save_fig(fig, name: str):
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    # Also save PNG for quick preview
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=150)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: GradCAM comparison — ResEncL vs CNN on same subjects
# ═══════════════════════════════════════════════════════════════════════════════
def fig_gradcam_comparison():
    """2 rows × N_subjects cols: ResEncL GradCAM vs CNN GradCAM."""
    subjects = REPRESENTATIVE_SUBJECTS
    n_subj = len(subjects)

    fig, axes = plt.subplots(3, n_subj, figsize=(4 * n_subj, 11))
    if n_subj == 1:
        axes = axes[:, np.newaxis]

    for col, subj in enumerate(subjects):
        # Load FLAIR and label
        flair = load_and_crop(RAW_DIR / "imagesTs" / f"{subj}_0000.nii.gz")
        label = load_and_crop(RAW_DIR / "labelsTs" / f"{subj}.nii.gz")

        # Load GradCAMs
        gcam_resencl_path = XAI_DIR / "resencl" / "nifti" / f"{subj}_gradcam.nii.gz"
        gcam_cnn_path = XAI_DIR / "cnn" / "nifti" / f"{subj}_gradcam.nii.gz"

        if not gcam_resencl_path.exists() or not gcam_cnn_path.exists():
            print(f"  Skipping {subj}: missing GradCAM NIfTI")
            continue

        gcam_resencl = np.asarray(nib.load(str(gcam_resencl_path)).dataobj, dtype=np.float32)
        gcam_cnn = np.asarray(nib.load(str(gcam_cnn_path)).dataobj, dtype=np.float32)

        # Brain masking: zero out non-brain regions to remove padding artifacts
        brain_mask = make_brain_mask(flair).astype(np.float32)
        gcam_resencl *= brain_mask
        gcam_cnn *= brain_mask

        # Pick best slice
        best_slices = find_lesion_slices(label, n=1)
        sl = best_slices[0]

        flair_sl = flair[:, :, sl].T
        label_sl = label[:, :, sl].T

        # Row 0: FLAIR + GT
        axes[0, col].imshow(flair_sl, cmap="gray", origin="lower")
        axes[0, col].imshow(label_sl, cmap="Reds", alpha=0.4, origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"{subj} (z={sl})", fontsize=11)
        if col == 0:
            axes[0, col].set_ylabel("FLAIR + GT", fontsize=11)

        # Row 1: ResEncL GradCAM
        axes[1, col].imshow(flair_sl, cmap="gray", origin="lower")
        im1 = axes[1, col].imshow(gcam_resencl[:, :, sl].T, cmap="jet", alpha=0.5,
                                   origin="lower", vmin=0, vmax=1)
        if col == 0:
            axes[1, col].set_ylabel("ResEncL Grad-CAM", fontsize=11)

        # Row 2: CNN GradCAM
        axes[2, col].imshow(flair_sl, cmap="gray", origin="lower")
        im2 = axes[2, col].imshow(gcam_cnn[:, :, sl].T, cmap="jet", alpha=0.5,
                                   origin="lower", vmin=0, vmax=1)
        if col == 0:
            axes[2, col].set_ylabel("CNN Grad-CAM", fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label="Activation intensity")

    fig.suptitle("Grad-CAM Comparison: ResEncL-3D vs CNN-3D", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    save_fig(fig, "fig_gradcam_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: RISE saliency maps
# ═══════════════════════════════════════════════════════════════════════════════
def fig_rise_heatmap():
    """Show RISE saliency alongside FLAIR+GT for representative subjects."""
    subjects = REPRESENTATIVE_SUBJECTS[:3]
    n_subj = len(subjects)

    fig, axes = plt.subplots(2, n_subj, figsize=(4 * n_subj, 7))
    if n_subj == 1:
        axes = axes[:, np.newaxis]

    for col, subj in enumerate(subjects):
        flair = load_and_crop(RAW_DIR / "imagesTs" / f"{subj}_0000.nii.gz")
        label = load_and_crop(RAW_DIR / "labelsTs" / f"{subj}.nii.gz")
        rise_path = XAI_DIR / "resencl" / "nifti" / f"{subj}_rise.nii.gz"

        if not rise_path.exists():
            print(f"  Skipping {subj}: no RISE NIfTI")
            continue

        rise = np.asarray(nib.load(str(rise_path)).dataobj, dtype=np.float32)

        # Brain masking: zero out non-brain regions to remove padding artifacts
        brain_mask = make_brain_mask(flair).astype(np.float32)
        rise *= brain_mask

        sl = find_lesion_slices(label, n=1)[0]
        flair_sl = flair[:, :, sl].T
        label_sl = label[:, :, sl].T

        # Row 0: FLAIR + GT
        axes[0, col].imshow(flair_sl, cmap="gray", origin="lower")
        axes[0, col].imshow(label_sl, cmap="Reds", alpha=0.4, origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"{subj} (z={sl})", fontsize=11)
        if col == 0:
            axes[0, col].set_ylabel("FLAIR + GT", fontsize=11)

        # Row 1: RISE
        axes[1, col].imshow(flair_sl, cmap="gray", origin="lower")
        im = axes[1, col].imshow(rise[:, :, sl].T, cmap="jet", alpha=0.5,
                                  origin="lower", vmin=0, vmax=1)
        if col == 0:
            axes[1, col].set_ylabel("3D-RISE saliency", fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Saliency")

    fig.suptitle("3D RISE Saliency Maps (ResEncL-3D, 500 masks)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    save_fig(fig, "fig_rise_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: All XAI methods compared on same subject
# ═══════════════════════════════════════════════════════════════════════════════
def fig_xai_methods_comparison():
    """Single subject, 2 slices, comparing all available methods."""
    subj = REPRESENTATIVE_SUBJECTS[0]
    flair = load_and_crop(RAW_DIR / "imagesTs" / f"{subj}_0000.nii.gz")
    label = load_and_crop(RAW_DIR / "labelsTs" / f"{subj}.nii.gz")
    slices = find_lesion_slices(label, n=2)

    # Collect available methods
    methods = {}
    for name, fname in [("Grad-CAM", "gradcam"), ("3D-RISE", "rise"), ("Occlusion", "occlusion")]:
        p = XAI_DIR / "resencl" / "nifti" / f"{subj}_{fname}.nii.gz"
        if p.exists():
            methods[name] = np.asarray(nib.load(str(p)).dataobj, dtype=np.float32)

    if not methods:
        print("  No XAI method outputs found; skipping comparison figure")
        return

    # Brain masking: zero out non-brain regions to remove padding artifacts
    brain_mask = make_brain_mask(flair).astype(np.float32)
    for mname in methods:
        methods[mname] *= brain_mask

    n_methods = len(methods)
    n_slices = len(slices)
    n_cols = 1 + n_methods  # FLAIR+GT + methods

    fig, axes = plt.subplots(n_slices, n_cols, figsize=(4 * n_cols, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row, sl in enumerate(slices):
        flair_sl = flair[:, :, sl].T
        label_sl = label[:, :, sl].T

        # Col 0: FLAIR + GT
        axes[row, 0].imshow(flair_sl, cmap="gray", origin="lower")
        axes[row, 0].imshow(label_sl, cmap="Reds", alpha=0.4, origin="lower", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"z = {sl}", fontsize=10)
        if row == 0:
            axes[row, 0].set_title("FLAIR + GT", fontsize=11)

        for col_idx, (mname, hmap) in enumerate(methods.items(), start=1):
            axes[row, col_idx].imshow(flair_sl, cmap="gray", origin="lower")
            im = axes[row, col_idx].imshow(hmap[:, :, sl].T, cmap="jet", alpha=0.5,
                                            origin="lower", vmin=0, vmax=1)
            if row == 0:
                axes[row, col_idx].set_title(mname, fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Importance")

    fig.suptitle(f"XAI Method Comparison — {subj} (ResEncL-3D)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    save_fig(fig, "fig_xai_methods_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating XAI thesis figures...")
    fig_gradcam_comparison()
    fig_rise_heatmap()
    fig_xai_methods_comparison()
    print("Done.")
