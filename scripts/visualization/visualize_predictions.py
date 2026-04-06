#!/usr/bin/env python3
"""
visualize_predictions.py
========================
Generate thesis-quality side-by-side comparison panels:

    [ FLAIR slice ] | [ Ground Truth ] | [ Prediction ] | [ Overlay ]

For each experiment, picks the best / median / worst subjects by Dice score
and renders multi-slice PNG panels.  Also supports visualising ALL subjects.

Can be run:
  - After training (uses validation predictions in fold_0/validation/)
  - After inference (uses test predictions in predictions/)
  - Interactively for a single subject

Usage:
    python visualize_predictions.py                        # all available
    python visualize_predictions.py --mode validation      # validation only
    python visualize_predictions.py --mode test            # test set only
    python visualize_predictions.py --subject sub_0001     # single subject
    python visualize_predictions.py --top-k 5              # best/worst 5
    python visualize_predictions.py --all                  # every subject
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_helpers import (
    get_nnunet_env_paths,
    glob_nifti,
    load_config,
    logger,
    setup_logging,
    strip_nifti_ext,
)

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required:  pip install nibabel")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
except ImportError:
    raise ImportError("matplotlib is required:  pip install matplotlib")


# ──────────────────────────────────────────────────────────────────────────────
# Dice helper (for ranking subjects)
# ──────────────────────────────────────────────────────────────────────────────
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


# ──────────────────────────────────────────────────────────────────────────────
# Slice selection
# ──────────────────────────────────────────────────────────────────────────────
def pick_slices(gt_vol: np.ndarray, n_slices: int = 5) -> List[int]:
    """
    Pick axial slices that best show lesions.
    Strategy: find slices with the most lesion voxels, spread out.
    """
    # Sum lesion voxels per axial slice (axis 2 for nibabel RAS)
    axial_axis = 2
    lesion_per_slice = np.sum(gt_vol > 0, axis=(0, 1))

    if lesion_per_slice.max() == 0:
        # No lesions — pick evenly spaced slices
        total = gt_vol.shape[axial_axis]
        return list(np.linspace(total * 0.2, total * 0.8, n_slices, dtype=int))

    # Rank slices by lesion count
    ranked = np.argsort(lesion_per_slice)[::-1]

    # Greedily pick slices that are at least 5 apart
    selected = []
    min_gap = max(3, gt_vol.shape[axial_axis] // (n_slices * 2))
    for s in ranked:
        if len(selected) >= n_slices:
            break
        if all(abs(s - prev) >= min_gap for prev in selected):
            selected.append(int(s))

    # If we couldn't get enough, relax the gap constraint
    if len(selected) < n_slices:
        for s in ranked:
            if len(selected) >= n_slices:
                break
            if s not in selected:
                selected.append(int(s))

    return sorted(selected[:n_slices])


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────
def render_subject(
    flair_vol: np.ndarray,
    gt_vol: np.ndarray,
    pred_vol: np.ndarray,
    subject_name: str,
    experiment_name: str,
    dice: float,
    output_path: Path,
    n_slices: int = 5,
) -> None:
    """
    Render a multi-row panel for one subject.
    Each row = one axial slice.
    Columns: FLAIR | GT (green) | Prediction (red) | Overlay
    """
    slices = pick_slices(gt_vol, n_slices)

    fig, axes = plt.subplots(
        len(slices), 4,
        figsize=(16, 3.5 * len(slices)),
        facecolor="black",
    )

    if len(slices) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["FLAIR", "Ground Truth", "Prediction", "Overlay"]

    # Custom colormaps
    gt_cmap = ListedColormap(["black", "#00ff00"])   # green
    pred_cmap = ListedColormap(["black", "#ff3333"]) # red

    for row, sl in enumerate(slices):
        flair_slice = np.rot90(flair_vol[:, :, sl])
        gt_slice = np.rot90(gt_vol[:, :, sl] > 0).astype(np.uint8)
        pred_slice = np.rot90(pred_vol[:, :, sl] > 0).astype(np.uint8)

        # Normalize FLAIR for display
        vmin = np.percentile(flair_slice, 1)
        vmax = np.percentile(flair_slice, 99)
        flair_norm = np.clip((flair_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # Col 0: FLAIR
        axes[row, 0].imshow(flair_norm, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"Slice {sl}", color="white", fontsize=10)

        # Col 1: Ground Truth
        axes[row, 1].imshow(flair_norm, cmap="gray", vmin=0, vmax=1)
        gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
        axes[row, 1].imshow(gt_masked, cmap=gt_cmap, alpha=0.6, vmin=0, vmax=1)

        # Col 2: Prediction
        axes[row, 2].imshow(flair_norm, cmap="gray", vmin=0, vmax=1)
        pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[row, 2].imshow(pred_masked, cmap=pred_cmap, alpha=0.6, vmin=0, vmax=1)

        # Col 3: Overlay (GT green + Pred red + overlap yellow)
        axes[row, 3].imshow(flair_norm, cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*flair_norm.shape, 4), dtype=np.float32)

        # True Positive (yellow)
        tp = (gt_slice > 0) & (pred_slice > 0)
        overlay[tp] = [1.0, 1.0, 0.0, 0.7]

        # False Negative (green = missed by model)
        fn = (gt_slice > 0) & (pred_slice == 0)
        overlay[fn] = [0.0, 1.0, 0.0, 0.7]

        # False Positive (red = hallucinated)
        fp = (gt_slice == 0) & (pred_slice > 0)
        overlay[fp] = [1.0, 0.0, 0.0, 0.7]

        axes[row, 3].imshow(overlay)

        # Column titles on first row
        if row == 0:
            for col, title in enumerate(col_titles):
                axes[row, col].set_title(title, color="white", fontsize=13,
                                          fontweight="bold", pad=10)

        # Clean up axes
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    # Legend for overlay
    legend_patches = [
        mpatches.Patch(color="#ffff00", label="True Positive (overlap)"),
        mpatches.Patch(color="#00ff00", label="False Negative (missed)"),
        mpatches.Patch(color="#ff0000", label="False Positive (extra)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=11,
        facecolor="black",
        edgecolor="gray",
        labelcolor="white",
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        f"{experiment_name}  |  {subject_name}  |  Dice = {dice:.4f}",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Discovery: find available experiments
# ──────────────────────────────────────────────────────────────────────────────
def discover_experiments(
    raw_base: Path,
    results_base: Path,
    mode: str = "all",
) -> List[Dict[str, Any]]:
    """
    Find all (experiment_name, pred_dir, gt_dir, flair_dir) combos.
    mode: "validation" | "test" | "all"
    """
    experiments = []

    dataset_configs = {
        500: {
            "name": "RawFLAIR",
            "flair_tr": raw_base / "Dataset500_RawFLAIR" / "imagesTr",
            "flair_ts": raw_base / "Dataset500_RawFLAIR" / "imagesTs",
            "gt_tr": raw_base / "Dataset500_RawFLAIR" / "labelsTr",
            "gt_ts": raw_base / "Dataset500_RawFLAIR" / "labelsTs",
        },
        501: {
            "name": "SkullStrippedFLAIR",
            "flair_tr": raw_base / "Dataset501_SkullStrippedFLAIR" / "imagesTr",
            "flair_ts": raw_base / "Dataset501_SkullStrippedFLAIR" / "imagesTs",
            "gt_tr": raw_base / "Dataset501_SkullStrippedFLAIR" / "labelsTr",
            "gt_ts": raw_base / "Dataset501_SkullStrippedFLAIR" / "labelsTs",
        },
    }

    trainers = [
        ("CNN", "nnUNetTrainer__nnUNetPlans__3d_fullres"),
        ("ResEncL", "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"),
    ]

    # ── Validation predictions (inside fold results) ─────────────────────
    if mode in ("validation", "all"):
        for ds_id, ds_cfg in dataset_configs.items():
            for tag, trainer_dir in trainers:
                val_dir = (
                    results_base / f"Dataset{ds_id}_{ds_cfg['name']}"
                    / trainer_dir / "fold_0" / "validation"
                )
                if val_dir.is_dir() and list(val_dir.glob("*.nii.gz")):
                    experiments.append({
                        "name": f"DS{ds_id}_{tag}_validation",
                        "pred_dir": val_dir,
                        "gt_dir": ds_cfg["gt_tr"],
                        "flair_dir": ds_cfg["flair_tr"],
                        "flair_suffix": "_0000",
                    })

    # ── Test predictions (after inference) ───────────────────────────────
    if mode in ("test", "all"):
        pred_base = results_base / "predictions"
        ens_base = results_base / "ensembles"
        adapt_base = results_base / "adaptive_ensembles"

        for ds_id, ds_cfg in dataset_configs.items():
            for tag in ("CNN", "ResEncL"):
                pred_dir = pred_base / f"DS{ds_id}_{tag}"
                if pred_dir.is_dir() and list(pred_dir.glob("*.nii.gz")):
                    experiments.append({
                        "name": f"DS{ds_id}_{tag}_test",
                        "pred_dir": pred_dir,
                        "gt_dir": ds_cfg["gt_ts"],
                        "flair_dir": ds_cfg["flair_ts"],
                        "flair_suffix": "_0000",
                    })

            # Standard ensemble
            ens_dir = ens_base / f"DS{ds_id}_CNN_plus_ResEncL"
            if ens_dir.is_dir() and list(ens_dir.glob("*.nii.gz")):
                experiments.append({
                    "name": f"DS{ds_id}_Ensemble_test",
                    "pred_dir": ens_dir,
                    "gt_dir": ds_cfg["gt_ts"],
                    "flair_dir": ds_cfg["flair_ts"],
                    "flair_suffix": "_0000",
                })

            # Adaptive ensemble
            adapt_dir = adapt_base / f"DS{ds_id}_AdaptiveEnsemble"
            if adapt_dir.is_dir() and list(adapt_dir.glob("*.nii.gz")):
                experiments.append({
                    "name": f"DS{ds_id}_AdaptiveEns_test",
                    "pred_dir": adapt_dir,
                    "gt_dir": ds_cfg["gt_ts"],
                    "flair_dir": ds_cfg["flair_ts"],
                    "flair_suffix": "_0000",
                })

    return experiments


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(
    mode: str = "all",
    top_k: int = 3,
    n_slices: int = 5,
    subject_filter: Optional[str] = None,
    show_all: bool = False,
) -> None:
    setup_logging(log_file="logs/visualize_predictions.log")
    raw_base, _, results_base = get_nnunet_env_paths()
    raw_base = Path(raw_base)
    results_base = Path(results_base)

    output_base = results_base / "visualizations"
    output_base.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(raw_base, results_base, mode)

    if not experiments:
        logger.warning(
            "No predictions found yet. Run this after training completes "
            "(validation) or after Phase 3 inference (test)."
        )
        print("\nNo predictions found. Available after:")
        print("  - Training completes -> validation predictions")
        print("  - Phase 3 inference  -> test predictions")
        return

    logger.info("Found %d experiment(s) to visualize.", len(experiments))

    total_rendered = 0
    for exp in experiments:
        exp_name = exp["name"]
        pred_dir = exp["pred_dir"]
        gt_dir = exp["gt_dir"]
        flair_dir = exp["flair_dir"]
        flair_suffix = exp.get("flair_suffix", "_0000")

        logger.info("Processing: %s", exp_name)

        pred_files = sorted(glob_nifti(pred_dir, suffix=".nii.gz"))
        gt_files = glob_nifti(gt_dir, suffix=".nii.gz")
        gt_map = {strip_nifti_ext(f.name): f for f in gt_files}

        # Compute Dice for all subjects
        subject_scores: List[Tuple[str, float, Path]] = []
        for pf in pred_files:
            stem = strip_nifti_ext(pf.name)
            if stem not in gt_map:
                continue
            if subject_filter and subject_filter not in stem:
                continue

            pred_data = (np.asarray(nib.load(str(pf)).dataobj) > 0).astype(np.uint8)
            gt_data = (np.asarray(nib.load(str(gt_map[stem])).dataobj) > 0).astype(np.uint8)
            d = dice_score(pred_data, gt_data)
            subject_scores.append((stem, d, pf))

        if not subject_scores:
            logger.info("  No matching subjects found.")
            continue

        # Sort by Dice
        subject_scores.sort(key=lambda x: x[1])

        # Select subjects to render
        if show_all or subject_filter:
            to_render = subject_scores
        else:
            # Best k, worst k, median
            worst = subject_scores[:top_k]
            best = subject_scores[-top_k:]
            mid_idx = len(subject_scores) // 2
            median = subject_scores[max(0, mid_idx - top_k // 2):mid_idx + (top_k + 1) // 2]
            # Deduplicate
            seen = set()
            to_render = []
            for item in worst + median + best:
                if item[0] not in seen:
                    to_render.append(item)
                    seen.add(item[0])

        logger.info(
            "  %d subjects scored, rendering %d panels.",
            len(subject_scores), len(to_render),
        )

        exp_out = output_base / exp_name
        exp_out.mkdir(parents=True, exist_ok=True)

        for stem, d, pf in to_render:
            # Load volumes
            pred_nii = nib.load(str(pf))
            pred_vol = (np.asarray(pred_nii.dataobj) > 0).astype(np.uint8)
            gt_vol = (np.asarray(nib.load(str(gt_map[stem])).dataobj) > 0).astype(np.uint8)

            # Find FLAIR image
            flair_path = flair_dir / f"{stem}{flair_suffix}.nii.gz"
            if not flair_path.exists():
                # Try without suffix
                flair_path = flair_dir / f"{stem}.nii.gz"
            if not flair_path.exists():
                logger.warning("  FLAIR not found for %s, skipping.", stem)
                continue

            flair_vol = np.asarray(nib.load(str(flair_path)).dataobj).astype(np.float32)

            # Categorise for filename
            rank_idx = subject_scores.index((stem, d, pf))
            n_total = len(subject_scores)
            if rank_idx < top_k:
                category = "WORST"
            elif rank_idx >= n_total - top_k:
                category = "BEST"
            else:
                category = "MEDIAN"

            out_file = exp_out / f"{category}_dice{d:.4f}_{stem}.png"

            render_subject(
                flair_vol, gt_vol, pred_vol,
                subject_name=stem,
                experiment_name=exp_name,
                dice=d,
                output_path=out_file,
                n_slices=n_slices,
            )
            total_rendered += 1
            logger.info("    Saved: %s", out_file.name)

        # Save Dice ranking
        ranking_file = exp_out / "dice_ranking.txt"
        with open(ranking_file, "w") as fh:
            fh.write(f"Dice Ranking for {exp_name}\n")
            fh.write(f"{'='*60}\n")
            fh.write(f"{'Rank':>5}  {'Subject':<30}  {'Dice':>8}\n")
            fh.write(f"{'-'*60}\n")
            for i, (stem, d, _) in enumerate(subject_scores):
                fh.write(f"{i+1:>5}  {stem:<30}  {d:>8.4f}\n")
            fh.write(f"\nMean Dice: {np.mean([s[1] for s in subject_scores]):.4f}\n")
            fh.write(f"Std Dice:  {np.std([s[1] for s in subject_scores]):.4f}\n")
        logger.info("  Dice ranking saved to %s", ranking_file)

    print(f"\nRendered {total_rendered} panels to: {output_base}")
    print("Open the PNG files to inspect segmentation quality.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize segmentation predictions vs ground truth."
    )
    parser.add_argument(
        "--mode", choices=["validation", "test", "all"], default="all",
        help="Which predictions to visualize (default: all available)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of best/worst/median subjects to render (default: 3)",
    )
    parser.add_argument(
        "--n-slices", type=int, default=5,
        help="Number of axial slices per subject (default: 5)",
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Filter to a specific subject ID (partial match)",
    )
    parser.add_argument(
        "--all", action="store_true", dest="show_all",
        help="Render ALL subjects (not just best/worst/median)",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        top_k=args.top_k,
        n_slices=args.n_slices,
        subject_filter=args.subject,
        show_all=args.show_all,
    )
