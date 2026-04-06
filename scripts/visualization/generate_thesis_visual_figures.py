#!/usr/bin/env python3
"""
generate_thesis_visual_figures.py
=================================
Generate thesis-quality visual figures for Chapter 5 (Results) and Chapter 6 (XAI).

Produces:
  1. fig_best_predictions.pdf    — Best performing subjects (GT vs Pred, multi-slice)
  2. fig_worst_predictions.pdf   — Worst performing subjects showing failures
  3. fig_small_lesion_failures.pdf — Close-up of small lesion misses vs successes
  4. fig_segmentation_gallery.pdf — Gallery of diverse segmentation results
  5. fig_xai_gradcam_gallery.pdf — Extended GradCAM gallery (best heatmaps)
  6. fig_xai_rise_gallery.pdf   — Extended RISE saliency gallery
  7. fig_rfdetr_bboxes.pdf       — RF-DETR bounding box detection examples
  8. fig_slice_overview.pdf      — Full brain slices showing lesion distribution

Usage:
  python scripts/visualization/generate_thesis_visual_figures.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from scipy import ndimage
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT
RAW_DIR = ROOT / "data" / "nnUNet_raw" / "Dataset001_MSLesSeg"
PRED_BASE = ROOT / "results" / "predictions"
XAI_DIR = ROOT / "results" / "xai"
RFDETR_DIR = ROOT / "archive" / "rfdetr_smoke" / "Dataset003_Combined"
OUT_DIR = ROOT / "thesis_outputs" / "figures" / "visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# DS001 test subjects
SUBJECTS = [f"MSL_{i:04d}" for i in range(94, 116)]
PATCH_SIZE = (160, 192, 160)

# Best ensemble predictions directory (CNN-3D + ResEncL-2.5D-chfix + ResEncL-3D, thr=0.40)
# We need to build ensemble from softmax — or use the single best model predictions
# Since per-subject metrics exist for DS001_ResEncL_3D, use those + DS003 model
BASELINE_PRED_DIR = PRED_BASE / "DS001_ResEncL_3D"  # baseline hard preds
DS003_PRED_DIR = PRED_BASE / "DS001_DS003_ResEncL_3D_TTA"  # best single model


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI file as float32."""
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def center_crop(vol: np.ndarray, target: tuple = PATCH_SIZE) -> np.ndarray:
    """Center-crop volume to target shape."""
    slices = []
    for dim, ps in enumerate(target):
        s = vol.shape[dim]
        if s >= ps:
            start = (s - ps) // 2
            slices.append(slice(start, start + ps))
        else:
            slices.append(slice(0, s))
    return vol[tuple(slices)]


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] with percentile clipping."""
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score between two binary masks."""
    inter = np.sum((pred > 0) & (gt > 0))
    total = np.sum(pred > 0) + np.sum(gt > 0)
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def find_lesion_slices(label: np.ndarray, n: int = 5) -> list[int]:
    """Find axial slices with most lesion voxels, spread out."""
    sums = label.sum(axis=(0, 1))
    ranked = np.argsort(sums)[::-1]
    selected = []
    min_gap = max(3, label.shape[2] // (n * 2))
    for s in ranked:
        if len(selected) >= n:
            break
        if sums[s] == 0:
            break
        if all(abs(s - prev) >= min_gap for prev in selected):
            selected.append(int(s))
    if len(selected) < n:
        for s in ranked:
            if len(selected) >= n:
                break
            if s not in selected and sums[s] > 0:
                selected.append(int(s))
    return sorted(selected[:n])


def make_brain_mask(flair: np.ndarray, dilate_voxels: int = 3) -> np.ndarray:
    """Create a binary brain foreground mask from a FLAIR volume."""
    nonzero = flair[flair > 0]
    if nonzero.size == 0:
        return np.ones(flair.shape, dtype=bool)
    thr = max(0, nonzero.mean() - 2.0 * nonzero.std())
    mask = flair > thr
    if dilate_voxels > 0:
        struct = ndimage.generate_binary_structure(mask.ndim, 1)
        mask = ndimage.binary_dilation(mask, struct, iterations=dilate_voxels)
    return mask


def find_small_lesion_slice(gt: np.ndarray, pred: np.ndarray, find_miss: bool = True):
    """
    Find a slice containing a small lesion that was missed (find_miss=True)
    or correctly detected (find_miss=False).
    Returns (slice_idx, lesion_centroid_yx, lesion_size_voxels).
    """
    labeled, n_lesions = ndimage.label(gt > 0)
    results = []
    for i in range(1, n_lesions + 1):
        mask = labeled == i
        size = mask.sum()
        if 3 <= size <= 50:  # small lesions: 3-50 voxels
            overlap = np.sum(mask & (pred > 0))
            detected = overlap > 0
            if (find_miss and not detected) or (not find_miss and detected):
                com = ndimage.center_of_mass(mask)
                results.append((int(com[2]), (int(com[1]), int(com[0])), size))
    results.sort(key=lambda x: x[2])  # sort by size
    return results


def load_subject_data(subj: str, pred_dir: Optional[Path] = None):
    """Load FLAIR, GT, and prediction for a subject."""
    flair_path = RAW_DIR / "imagesTs" / f"{subj}_0000.nii.gz"
    gt_path = RAW_DIR / "labelsTs" / f"{subj}.nii.gz"

    if pred_dir is None:
        pred_dir = DS003_PRED_DIR
    pred_path = pred_dir / f"{subj}.nii.gz"

    if not all(p.exists() for p in [flair_path, gt_path, pred_path]):
        return None

    flair = load_nifti(flair_path)
    gt = load_nifti(gt_path)
    pred = load_nifti(pred_path)

    return flair, gt, pred


def save_fig(fig, name: str):
    """Save as both PDF and PNG."""
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {pdf_path} + PNG")


def compute_all_dice():
    """Compute per-subject Dice for DS003 model on DS001 test set."""
    results = []
    for subj in SUBJECTS:
        data = load_subject_data(subj, DS003_PRED_DIR)
        if data is None:
            continue
        _, gt, pred = data
        d = dice_score(pred, gt)
        results.append((subj, d))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Best Predictions
# ═════════════════════════════════════════════════════════════════════════════
def fig_best_predictions(dice_results):
    """GT vs Prediction for top-3 performing subjects."""
    print("\n[1/8] Generating best predictions figure...")

    top3 = dice_results[:3]
    n_slices = 3

    fig, axes = plt.subplots(len(top3) * n_slices, 4,
                              figsize=(14, 3.2 * len(top3) * n_slices),
                              facecolor="black")

    col_titles = ["FLAIR", "Ground Truth", "Prediction", "Overlay (TP/FN/FP)"]
    row_idx = 0

    for subj, dsc in top3:
        data = load_subject_data(subj)
        if data is None:
            continue
        flair, gt, pred = data
        slices = find_lesion_slices(gt, n_slices)

        for i, sl in enumerate(slices):
            flair_sl = np.rot90(flair[:, :, sl])
            gt_sl = np.rot90((gt[:, :, sl] > 0).astype(np.uint8))
            pred_sl = np.rot90((pred[:, :, sl] > 0).astype(np.uint8))
            flair_norm = normalize_image(flair_sl)

            # FLAIR
            axes[row_idx, 0].imshow(flair_norm, cmap="gray")
            if i == 0:
                axes[row_idx, 0].set_ylabel(f"{subj}\nDSC={dsc:.3f}",
                                             color="white", fontsize=10, fontweight="bold")
            else:
                axes[row_idx, 0].set_ylabel(f"Slice {sl}", color="white", fontsize=9)

            # GT overlay
            axes[row_idx, 1].imshow(flair_norm, cmap="gray")
            gt_masked = np.ma.masked_where(gt_sl == 0, gt_sl)
            axes[row_idx, 1].imshow(gt_masked, cmap=ListedColormap(["black", "#00ff00"]),
                                     alpha=0.6, vmin=0, vmax=1)

            # Prediction overlay
            axes[row_idx, 2].imshow(flair_norm, cmap="gray")
            pred_masked = np.ma.masked_where(pred_sl == 0, pred_sl)
            axes[row_idx, 2].imshow(pred_masked, cmap=ListedColormap(["black", "#ff3333"]),
                                     alpha=0.6, vmin=0, vmax=1)

            # Overlay
            axes[row_idx, 3].imshow(flair_norm, cmap="gray")
            overlay = np.zeros((*flair_norm.shape, 4), dtype=np.float32)
            tp = (gt_sl > 0) & (pred_sl > 0)
            fn = (gt_sl > 0) & (pred_sl == 0)
            fp = (gt_sl == 0) & (pred_sl > 0)
            overlay[tp] = [1.0, 1.0, 0.0, 0.7]
            overlay[fn] = [0.0, 1.0, 0.0, 0.7]
            overlay[fp] = [1.0, 0.0, 0.0, 0.7]
            axes[row_idx, 3].imshow(overlay)

            if row_idx == 0:
                for col, title in enumerate(col_titles):
                    axes[0, col].set_title(title, color="white", fontsize=12, fontweight="bold")

            for col in range(4):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])
                for spine in axes[row_idx, col].spines.values():
                    spine.set_visible(False)

            row_idx += 1

    legend_patches = [
        mpatches.Patch(color="#ffff00", label="True Positive"),
        mpatches.Patch(color="#00ff00", label="False Negative (missed)"),
        mpatches.Patch(color="#ff0000", label="False Positive"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, facecolor="black", edgecolor="gray",
               labelcolor="white", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Best Segmentation Results — DS003 ResEncL-3D on DS001 Test Set",
                 color="white", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, "fig_best_predictions")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Worst Predictions
# ═════════════════════════════════════════════════════════════════════════════
def fig_worst_predictions(dice_results):
    """GT vs Prediction for bottom-3 performing subjects."""
    print("\n[2/8] Generating worst predictions figure...")

    bottom3 = dice_results[-3:]
    n_slices = 3

    fig, axes = plt.subplots(len(bottom3) * n_slices, 4,
                              figsize=(14, 3.2 * len(bottom3) * n_slices),
                              facecolor="black")

    col_titles = ["FLAIR", "Ground Truth", "Prediction", "Overlay (TP/FN/FP)"]
    row_idx = 0

    for subj, dsc in bottom3:
        data = load_subject_data(subj)
        if data is None:
            continue
        flair, gt, pred = data
        slices = find_lesion_slices(gt, n_slices)

        for i, sl in enumerate(slices):
            flair_sl = np.rot90(flair[:, :, sl])
            gt_sl = np.rot90((gt[:, :, sl] > 0).astype(np.uint8))
            pred_sl = np.rot90((pred[:, :, sl] > 0).astype(np.uint8))
            flair_norm = normalize_image(flair_sl)

            axes[row_idx, 0].imshow(flair_norm, cmap="gray")
            if i == 0:
                axes[row_idx, 0].set_ylabel(f"{subj}\nDSC={dsc:.3f}",
                                             color="white", fontsize=10, fontweight="bold")
            else:
                axes[row_idx, 0].set_ylabel(f"Slice {sl}", color="white", fontsize=9)

            axes[row_idx, 1].imshow(flair_norm, cmap="gray")
            gt_masked = np.ma.masked_where(gt_sl == 0, gt_sl)
            axes[row_idx, 1].imshow(gt_masked, cmap=ListedColormap(["black", "#00ff00"]),
                                     alpha=0.6, vmin=0, vmax=1)

            axes[row_idx, 2].imshow(flair_norm, cmap="gray")
            pred_masked = np.ma.masked_where(pred_sl == 0, pred_sl)
            axes[row_idx, 2].imshow(pred_masked, cmap=ListedColormap(["black", "#ff3333"]),
                                     alpha=0.6, vmin=0, vmax=1)

            axes[row_idx, 3].imshow(flair_norm, cmap="gray")
            overlay = np.zeros((*flair_norm.shape, 4), dtype=np.float32)
            tp = (gt_sl > 0) & (pred_sl > 0)
            fn = (gt_sl > 0) & (pred_sl == 0)
            fp = (gt_sl == 0) & (pred_sl > 0)
            overlay[tp] = [1.0, 1.0, 0.0, 0.7]
            overlay[fn] = [0.0, 1.0, 0.0, 0.7]
            overlay[fp] = [1.0, 0.0, 0.0, 0.7]
            axes[row_idx, 3].imshow(overlay)

            if row_idx == 0:
                for col, title in enumerate(col_titles):
                    axes[0, col].set_title(title, color="white", fontsize=12, fontweight="bold")

            for col in range(4):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])
                for spine in axes[row_idx, col].spines.values():
                    spine.set_visible(False)

            row_idx += 1

    legend_patches = [
        mpatches.Patch(color="#ffff00", label="True Positive"),
        mpatches.Patch(color="#00ff00", label="False Negative (missed)"),
        mpatches.Patch(color="#ff0000", label="False Positive"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, facecolor="black", edgecolor="gray",
               labelcolor="white", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Worst Segmentation Results — DS003 ResEncL-3D on DS001 Test Set",
                 color="white", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, "fig_worst_predictions")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Small Lesion Failures vs Successes
# ═════════════════════════════════════════════════════════════════════════════
def fig_small_lesion_failures():
    """Zoomed-in views of small lesions: missed vs detected."""
    print("\n[3/8] Generating small lesion failures figure...")

    missed_examples = []
    detected_examples = []

    for subj in SUBJECTS:
        data = load_subject_data(subj)
        if data is None:
            continue
        flair, gt, pred = data

        misses = find_small_lesion_slice(gt, pred, find_miss=True)
        hits = find_small_lesion_slice(gt, pred, find_miss=False)

        for sl, centroid, size in misses[:2]:
            missed_examples.append((subj, flair, gt, pred, sl, centroid, size))
        for sl, centroid, size in hits[:2]:
            detected_examples.append((subj, flair, gt, pred, sl, centroid, size))

    # Pick up to 6 misses and 6 detections
    missed_examples = missed_examples[:6]
    detected_examples = detected_examples[:6]
    n_examples = max(len(missed_examples), len(detected_examples))

    if n_examples == 0:
        print("  No small lesion examples found, skipping.")
        return

    fig, axes = plt.subplots(n_examples, 4, figsize=(14, 3.0 * n_examples),
                              facecolor="black")
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    zoom_radius = 25  # pixels around centroid

    def render_zoomed(ax, flair_vol, gt_vol, pred_vol, sl, centroid, mode="overlay"):
        """Render zoomed view around a lesion centroid."""
        flair_sl = np.rot90(flair_vol[:, :, sl])
        gt_sl = np.rot90((gt_vol[:, :, sl] > 0).astype(np.uint8))
        pred_sl = np.rot90((pred_vol[:, :, sl] > 0).astype(np.uint8))
        flair_norm = normalize_image(flair_sl)

        # Map centroid through rot90
        h, w = flair_sl.shape
        cy, cx = centroid  # in original space
        # rot90 maps (r,c) -> (c, h-1-r) but for display we use rot90 on the whole image
        cy_rot = cx
        cx_rot = h - 1 - cy

        # Crop region
        y1 = max(0, cy_rot - zoom_radius)
        y2 = min(h, cy_rot + zoom_radius)
        x1 = max(0, cx_rot - zoom_radius)
        x2 = min(w, cx_rot + zoom_radius)

        flair_crop = flair_norm[y1:y2, x1:x2]
        gt_crop = gt_sl[y1:y2, x1:x2]
        pred_crop = pred_sl[y1:y2, x1:x2]

        if mode == "flair":
            ax.imshow(flair_crop, cmap="gray")
        elif mode == "gt":
            ax.imshow(flair_crop, cmap="gray")
            gt_m = np.ma.masked_where(gt_crop == 0, gt_crop)
            ax.imshow(gt_m, cmap=ListedColormap(["black", "#00ff00"]),
                      alpha=0.7, vmin=0, vmax=1)
        elif mode == "pred":
            ax.imshow(flair_crop, cmap="gray")
            pred_m = np.ma.masked_where(pred_crop == 0, pred_crop)
            ax.imshow(pred_m, cmap=ListedColormap(["black", "#ff3333"]),
                      alpha=0.7, vmin=0, vmax=1)
        elif mode == "overlay":
            ax.imshow(flair_crop, cmap="gray")
            overlay = np.zeros((*flair_crop.shape, 4), dtype=np.float32)
            tp = (gt_crop > 0) & (pred_crop > 0)
            fn = (gt_crop > 0) & (pred_crop == 0)
            fp = (gt_crop == 0) & (pred_crop > 0)
            overlay[tp] = [1.0, 1.0, 0.0, 0.8]
            overlay[fn] = [0.0, 1.0, 0.0, 0.8]
            overlay[fp] = [1.0, 0.0, 0.0, 0.8]
            ax.imshow(overlay)

    # Left side: missed lesions, Right side: detected
    col_titles = ["Missed — FLAIR", "Missed — GT+Pred", "Detected — FLAIR", "Detected — GT+Pred"]

    for row in range(n_examples):
        # Missed (cols 0-1)
        if row < len(missed_examples):
            subj, flair, gt, pred, sl, centroid, size = missed_examples[row]
            render_zoomed(axes[row, 0], flair, gt, pred, sl, centroid, "flair")
            render_zoomed(axes[row, 1], flair, gt, pred, sl, centroid, "overlay")
            axes[row, 0].set_ylabel(f"{subj}\n{size} vox", color="red",
                                     fontsize=9, fontweight="bold")
        else:
            axes[row, 0].set_facecolor("black")
            axes[row, 1].set_facecolor("black")

        # Detected (cols 2-3)
        if row < len(detected_examples):
            subj, flair, gt, pred, sl, centroid, size = detected_examples[row]
            render_zoomed(axes[row, 2], flair, gt, pred, sl, centroid, "flair")
            render_zoomed(axes[row, 3], flair, gt, pred, sl, centroid, "overlay")
            label_color = "#00cc00"
            axes[row, 2].set_ylabel(f"{subj}\n{size} vox", color=label_color,
                                     fontsize=9, fontweight="bold")
        else:
            axes[row, 2].set_facecolor("black")
            axes[row, 3].set_facecolor("black")

        if row == 0:
            for col, title in enumerate(col_titles):
                color = "red" if col < 2 else "#00cc00"
                axes[0, col].set_title(title, color=color, fontsize=11, fontweight="bold")

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#ffff00", label="True Positive"),
        mpatches.Patch(color="#00ff00", label="False Negative (missed)"),
        mpatches.Patch(color="#ff0000", label="False Positive"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, facecolor="black", edgecolor="gray",
               labelcolor="white", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Small Lesion Analysis — Missed vs. Correctly Detected (<50 voxels)",
                 color="white", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, "fig_small_lesion_failures")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: Segmentation Gallery (diverse results)
# ═════════════════════════════════════════════════════════════════════════════
def fig_segmentation_gallery(dice_results):
    """Compact gallery: 1 slice per subject, 6 subjects spread across DSC range."""
    print("\n[4/8] Generating segmentation gallery...")

    n = len(dice_results)
    # Pick subjects at various DSC ranks: best, ~25th%, median, ~75th%, worst, 2nd best
    indices = [0, n // 5, n // 3, n // 2, 2 * n // 3, n - 1]
    selected = [dice_results[i] for i in indices]

    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3.0 * len(selected)),
                              facecolor="black")

    col_titles = ["Ground Truth", "Prediction", "Overlay"]

    for row, (subj, dsc) in enumerate(selected):
        data = load_subject_data(subj)
        if data is None:
            continue
        flair, gt, pred = data

        # Pick single most informative slice
        sums = gt.sum(axis=(0, 1))
        best_sl = int(np.argmax(sums))

        flair_sl = np.rot90(flair[:, :, best_sl])
        gt_sl = np.rot90((gt[:, :, best_sl] > 0).astype(np.uint8))
        pred_sl = np.rot90((pred[:, :, best_sl] > 0).astype(np.uint8))
        flair_norm = normalize_image(flair_sl)

        # GT
        axes[row, 0].imshow(flair_norm, cmap="gray")
        gt_m = np.ma.masked_where(gt_sl == 0, gt_sl)
        axes[row, 0].imshow(gt_m, cmap=ListedColormap(["black", "#00ff00"]),
                             alpha=0.6, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{subj}\nDSC={dsc:.3f}", color="white",
                                 fontsize=10, fontweight="bold")

        # Pred
        axes[row, 1].imshow(flair_norm, cmap="gray")
        pred_m = np.ma.masked_where(pred_sl == 0, pred_sl)
        axes[row, 1].imshow(pred_m, cmap=ListedColormap(["black", "#ff3333"]),
                             alpha=0.6, vmin=0, vmax=1)

        # Overlay
        axes[row, 2].imshow(flair_norm, cmap="gray")
        overlay = np.zeros((*flair_norm.shape, 4), dtype=np.float32)
        tp = (gt_sl > 0) & (pred_sl > 0)
        fn = (gt_sl > 0) & (pred_sl == 0)
        fp = (gt_sl == 0) & (pred_sl > 0)
        overlay[tp] = [1.0, 1.0, 0.0, 0.7]
        overlay[fn] = [0.0, 1.0, 0.0, 0.7]
        overlay[fp] = [1.0, 0.0, 0.0, 0.7]
        axes[row, 2].imshow(overlay)

        if row == 0:
            for col, title in enumerate(col_titles):
                axes[0, col].set_title(title, color="white", fontsize=12, fontweight="bold")

        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#ffff00", label="True Positive"),
        mpatches.Patch(color="#00ff00", label="False Negative"),
        mpatches.Patch(color="#ff0000", label="False Positive"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, facecolor="black", edgecolor="gray",
               labelcolor="white", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Segmentation Gallery — Diverse Performance Range (DS001 Test Set)",
                 color="white", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, "fig_segmentation_gallery")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5: Extended GradCAM Gallery
# ═════════════════════════════════════════════════════════════════════════════
def fig_xai_gradcam_gallery():
    """GradCAM heatmaps for 4 subjects: FLAIR + GT + GradCAM overlay."""
    print("\n[5/8] Generating GradCAM gallery...")

    subjects = ["MSL_0094", "MSL_0097", "MSL_0100", "MSL_0095"]
    available = []
    for subj in subjects:
        gc_path = XAI_DIR / "resencl" / "nifti" / f"{subj}_gradcam.nii.gz"
        if gc_path.exists():
            available.append(subj)

    if not available:
        print("  No GradCAM data found, skipping.")
        return

    subjects = available[:4]
    fig, axes = plt.subplots(len(subjects), 4, figsize=(16, 3.5 * len(subjects)),
                              facecolor="black")
    if len(subjects) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["FLAIR", "Ground Truth", "ResEncL GradCAM", "CNN GradCAM"]

    for row, subj in enumerate(subjects):
        flair = load_nifti(RAW_DIR / "imagesTs" / f"{subj}_0000.nii.gz")
        gt = load_nifti(RAW_DIR / "labelsTs" / f"{subj}.nii.gz")

        gc_resencl_path = XAI_DIR / "resencl" / "nifti" / f"{subj}_gradcam.nii.gz"
        gc_cnn_path = XAI_DIR / "cnn" / "nifti" / f"{subj}_gradcam.nii.gz"

        gc_resencl = center_crop(load_nifti(gc_resencl_path)) if gc_resencl_path.exists() else None
        gc_cnn = center_crop(load_nifti(gc_cnn_path)) if gc_cnn_path.exists() else None

        flair = center_crop(flair)
        gt = center_crop(gt)

        # Brain masking: zero out non-brain regions to remove padding artifacts
        brain_mask = make_brain_mask(flair).astype(np.float32)
        if gc_resencl is not None:
            gc_resencl *= brain_mask
        if gc_cnn is not None:
            gc_cnn *= brain_mask

        # Pick best slice
        sums = gt.sum(axis=(0, 1))
        best_sl = int(np.argmax(sums))

        flair_sl = np.rot90(flair[:, :, best_sl])
        gt_sl = np.rot90((gt[:, :, best_sl] > 0).astype(np.uint8))
        flair_norm = normalize_image(flair_sl)

        # FLAIR
        axes[row, 0].imshow(flair_norm, cmap="gray")
        axes[row, 0].set_ylabel(subj, color="white", fontsize=11, fontweight="bold")

        # GT
        axes[row, 1].imshow(flair_norm, cmap="gray")
        gt_m = np.ma.masked_where(gt_sl == 0, gt_sl)
        axes[row, 1].imshow(gt_m, cmap=ListedColormap(["black", "#00ff00"]),
                             alpha=0.6, vmin=0, vmax=1)

        # ResEncL GradCAM
        if gc_resencl is not None:
            gc_sl = np.rot90(gc_resencl[:, :, best_sl])
            axes[row, 2].imshow(flair_norm, cmap="gray")
            gc_norm = gc_sl / (gc_sl.max() + 1e-8)
            gc_masked = np.ma.masked_where(gc_norm < 0.1, gc_norm)
            axes[row, 2].imshow(gc_masked, cmap="jet", alpha=0.6, vmin=0, vmax=1)
        else:
            axes[row, 2].imshow(flair_norm, cmap="gray")
            axes[row, 2].text(0.5, 0.5, "N/A", color="white", fontsize=14,
                              ha="center", va="center", transform=axes[row, 2].transAxes)

        # CNN GradCAM
        if gc_cnn is not None:
            gc_sl = np.rot90(gc_cnn[:, :, best_sl])
            axes[row, 3].imshow(flair_norm, cmap="gray")
            gc_norm = gc_sl / (gc_sl.max() + 1e-8)
            gc_masked = np.ma.masked_where(gc_norm < 0.1, gc_norm)
            axes[row, 3].imshow(gc_masked, cmap="jet", alpha=0.6, vmin=0, vmax=1)
        else:
            axes[row, 3].imshow(flair_norm, cmap="gray")
            axes[row, 3].text(0.5, 0.5, "N/A", color="white", fontsize=14,
                              ha="center", va="center", transform=axes[row, 3].transAxes)

        if row == 0:
            for col, title in enumerate(col_titles):
                axes[0, col].set_title(title, color="white", fontsize=12, fontweight="bold")

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    fig.suptitle("GradCAM Attention Maps — ResEncL vs CNN Architecture Comparison",
                 color="white", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_fig(fig, "fig_xai_gradcam_gallery")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6: Extended RISE Gallery
# ═════════════════════════════════════════════════════════════════════════════
def fig_xai_rise_gallery():
    """RISE saliency maps: FLAIR + GT + RISE heatmap for available subjects."""
    print("\n[6/8] Generating RISE gallery...")

    rise_subjects = []
    for subj in SUBJECTS:
        rise_path = XAI_DIR / "resencl" / "nifti" / f"{subj}_rise.nii.gz"
        if rise_path.exists():
            rise_subjects.append(subj)

    if not rise_subjects:
        print("  No RISE data found, skipping.")
        return

    subjects = rise_subjects[:4]
    n_slices = 2

    fig, axes = plt.subplots(len(subjects) * n_slices, 3,
                              figsize=(12, 3.0 * len(subjects) * n_slices),
                              facecolor="black")

    col_titles = ["FLAIR + GT Contour", "RISE Saliency Map", "RISE + GT Overlay"]
    row_idx = 0

    for subj in subjects:
        flair = center_crop(load_nifti(RAW_DIR / "imagesTs" / f"{subj}_0000.nii.gz"))
        gt = center_crop(load_nifti(RAW_DIR / "labelsTs" / f"{subj}.nii.gz"))
        rise = center_crop(load_nifti(XAI_DIR / "resencl" / "nifti" / f"{subj}_rise.nii.gz"))

        # Brain masking: zero out non-brain regions to remove padding artifacts
        brain_mask = make_brain_mask(flair).astype(np.float32)
        rise *= brain_mask

        slices = find_lesion_slices(gt, n_slices)

        for i, sl in enumerate(slices):
            flair_sl = np.rot90(flair[:, :, sl])
            gt_sl = np.rot90((gt[:, :, sl] > 0).astype(np.uint8))
            rise_sl = np.rot90(rise[:, :, sl])
            flair_norm = normalize_image(flair_sl)
            rise_norm = rise_sl / (rise_sl.max() + 1e-8)

            # FLAIR + GT contour
            axes[row_idx, 0].imshow(flair_norm, cmap="gray")
            axes[row_idx, 0].contour(gt_sl, levels=[0.5], colors=["#00ff00"], linewidths=1.5)
            if i == 0:
                axes[row_idx, 0].set_ylabel(subj, color="white", fontsize=10, fontweight="bold")

            # RISE only
            axes[row_idx, 1].imshow(rise_norm, cmap="inferno", vmin=0, vmax=1)

            # RISE + GT overlay
            axes[row_idx, 2].imshow(flair_norm, cmap="gray")
            rise_masked = np.ma.masked_where(rise_norm < 0.15, rise_norm)
            axes[row_idx, 2].imshow(rise_masked, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            axes[row_idx, 2].contour(gt_sl, levels=[0.5], colors=["#00ff00"], linewidths=1.5)

            if row_idx == 0:
                for col, title in enumerate(col_titles):
                    axes[0, col].set_title(title, color="white", fontsize=12, fontweight="bold")

            for col in range(3):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])
                for spine in axes[row_idx, col].spines.values():
                    spine.set_visible(False)

            row_idx += 1

    fig.suptitle("3D RISE Saliency Maps — Model Attention Regions (ResEncL-3D)",
                 color="white", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    save_fig(fig, "fig_xai_rise_gallery")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7: RF-DETR Bounding Box Examples
# ═════════════════════════════════════════════════════════════════════════════
def fig_rfdetr_bboxes():
    """Show RF-DETR bounding box annotations on 2D slices."""
    print("\n[7/8] Generating RF-DETR bounding box figure...")

    jsonl_path = RFDETR_DIR / "test.jsonl"
    img_dir = RFDETR_DIR / "images" / "test"

    if not jsonl_path.exists():
        print(f"  JSONL not found at {jsonl_path}, skipping.")
        return

    # Parse JSONL and find slices with many boxes
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d["boxes"]:
                entries.append(d)

    # Sort by number of boxes, pick diverse subjects
    entries.sort(key=lambda d: len(d["boxes"]), reverse=True)

    # Pick 6 entries from different subjects with varying box counts
    seen_subjects = set()
    selected = []
    for entry in entries:
        if entry["case_id"] not in seen_subjects and len(selected) < 8:
            selected.append(entry)
            seen_subjects.add(entry["case_id"])

    # Also add some with fewer boxes
    for entry in reversed(entries):
        if entry["case_id"] not in seen_subjects and len(selected) < 10:
            selected.append(entry)
            seen_subjects.add(entry["case_id"])

    selected = selected[:8]

    n_cols = 4
    n_rows = (len(selected) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.0 * n_rows),
                              facecolor="white")
    axes = axes.flatten()

    colors = plt.cm.Set1(np.linspace(0, 1, 10))

    for idx, entry in enumerate(selected):
        # Fix path: original path may point to old location
        img_name = Path(entry["image_path"]).name
        img_path = img_dir / img_name

        if not img_path.exists():
            axes[idx].text(0.5, 0.5, "Image\nnot found", ha="center", va="center",
                          transform=axes[idx].transAxes, fontsize=12)
            axes[idx].set_facecolor("lightgray")
            continue

        img = np.array(Image.open(img_path))
        axes[idx].imshow(img)

        # Draw bounding boxes
        for box_idx, box in enumerate(entry["boxes"]):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            color = colors[box_idx % len(colors)]
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2,
                                  edgecolor=color, facecolor="none")
            axes[idx].add_patch(rect)

        n_boxes = len(entry["boxes"])
        axes[idx].set_title(f"{entry['case_id']} z={entry['z']} ({n_boxes} lesions)",
                            fontsize=10, fontweight="bold")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    # Hide unused axes
    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("RF-DETR Lesion Detection — Ground Truth Bounding Boxes on 2D Slices\n"
                 "(RGB = [FLAIR, T1, FLAIR−T1])",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig_rfdetr_bboxes")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 8: Full Brain Slice Overview
# ═════════════════════════════════════════════════════════════════════════════
def fig_slice_overview():
    """Multi-view (axial/coronal/sagittal) of FLAIR + lesion map for 2 subjects."""
    print("\n[8/8] Generating slice overview figure...")

    subjects = ["MSL_0095", "MSL_0100"]  # High lesion load subjects

    fig, axes = plt.subplots(len(subjects), 3, figsize=(15, 5 * len(subjects)),
                              facecolor="black")

    view_titles = ["Axial", "Coronal", "Sagittal"]

    for row, subj in enumerate(subjects):
        data = load_subject_data(subj)
        if data is None:
            continue
        flair, gt, pred = data

        # Find center of lesion mass
        if gt.sum() > 0:
            com = ndimage.center_of_mass(gt > 0)
            cx, cy, cz = int(com[0]), int(com[1]), int(com[2])
        else:
            cx, cy, cz = [s // 2 for s in gt.shape]

        views = [
            (np.rot90(flair[:, :, cz]), np.rot90(gt[:, :, cz]), np.rot90(pred[:, :, cz])),  # axial
            (np.rot90(flair[:, cy, :]), np.rot90(gt[:, cy, :]), np.rot90(pred[:, cy, :])),    # coronal
            (np.rot90(flair[cx, :, :]), np.rot90(gt[cx, :, :]), np.rot90(pred[cx, :, :])),    # sagittal
        ]

        for col, (flair_v, gt_v, pred_v) in enumerate(views):
            flair_norm = normalize_image(flair_v)
            gt_bin = (gt_v > 0).astype(np.uint8)
            pred_bin = (pred_v > 0).astype(np.uint8)

            axes[row, col].imshow(flair_norm, cmap="gray")

            # Overlay: GT green contour, pred red filled
            overlay = np.zeros((*flair_norm.shape, 4), dtype=np.float32)
            tp = (gt_bin > 0) & (pred_bin > 0)
            fn = (gt_bin > 0) & (pred_bin == 0)
            fp = (gt_bin == 0) & (pred_bin > 0)
            overlay[tp] = [1.0, 1.0, 0.0, 0.6]
            overlay[fn] = [0.0, 1.0, 0.0, 0.6]
            overlay[fp] = [1.0, 0.0, 0.0, 0.6]
            axes[row, col].imshow(overlay)

            if row == 0:
                axes[0, col].set_title(view_titles[col], color="white",
                                        fontsize=13, fontweight="bold")
            if col == 0:
                dsc = dice_score(pred, gt)
                axes[row, 0].set_ylabel(f"{subj}\nDSC={dsc:.3f}", color="white",
                                         fontsize=11, fontweight="bold")

            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#ffff00", label="True Positive"),
        mpatches.Patch(color="#00ff00", label="False Negative"),
        mpatches.Patch(color="#ff0000", label="False Positive"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, facecolor="black", edgecolor="gray",
               labelcolor="white", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Multi-View Lesion Segmentation — Axial, Coronal, Sagittal",
                 color="white", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, "fig_slice_overview")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("Thesis Visual Figure Generator")
    print("=" * 70)

    # Compute dice scores for all subjects
    print("\nComputing per-subject Dice scores (DS003 ResEncL-3D on DS001 test)...")
    dice_results = compute_all_dice()

    print(f"\n  Subjects ranked by DSC:")
    for subj, dsc in dice_results:
        print(f"    {subj}: DSC = {dsc:.4f}")

    # Generate all figures
    fig_best_predictions(dice_results)
    fig_worst_predictions(dice_results)
    fig_small_lesion_failures()
    fig_segmentation_gallery(dice_results)
    fig_xai_gradcam_gallery()
    fig_xai_rise_gallery()
    fig_rfdetr_bboxes()
    fig_slice_overview()

    print("\n" + "=" * 70)
    print("All figures saved to:", OUT_DIR)
    print("=" * 70)
