#!/usr/bin/env python3
"""
analyze_lesion_sizes.py
=======================
Stratified lesion-level analysis across ablation experiments.

For each test subject, extracts individual connected components from the
ground-truth mask, categorises them by volume (small / medium / large),
and evaluates per-lesion detection and segmentation quality for every model.

This answers the question: "Is one architecture systematically better
for a particular lesion size range?"

Outputs
-------
- results/lesion_size_analysis.csv       Per-lesion metrics
- results/lesion_size_summary.csv        Aggregate metrics per model x size
- Console summary table

Usage:
    python analyze_lesion_sizes.py
    python analyze_lesion_sizes.py --config config/dataset_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.io_helpers import (
    get_nnunet_env_paths,
    glob_nifti,
    load_config,
    logger,
    setup_logging,
    strip_nifti_ext,
)
from utils.thesis_logger import PhaseTimer, init_thesis_logging

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required:  pip install nibabel")

from scipy.ndimage import label as connected_components


# ──────────────────────────────────────────────────────────────────────────────
# Lesion size categories (in voxels — converted to mL in the output)
# ──────────────────────────────────────────────────────────────────────────────
SIZE_BINS = {
    "small":  (1, 100),        # punctate lesions
    "medium": (101, 1000),     # juxtacortical patches
    "large":  (1001, np.inf),  # confluent plaques / Dawson's fingers
}


def classify_size(vol_voxels: int) -> str:
    """Return the size bin name for a lesion with *vol_voxels* voxels."""
    for name, (lo, hi) in SIZE_BINS.items():
        if lo <= vol_voxels <= hi:
            return name
    return "large"  # fallback


# ──────────────────────────────────────────────────────────────────────────────
# Per-lesion metrics
# ──────────────────────────────────────────────────────────────────────────────
def lesion_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between a single predicted and ground-truth lesion."""
    intersection = np.sum(pred_mask & gt_mask)
    total = np.sum(pred_mask) + np.sum(gt_mask)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def lesion_overlap(pred_mask: np.ndarray, gt_lesion_mask: np.ndarray) -> float:
    """Fraction of GT lesion voxels captured by prediction."""
    if np.sum(gt_lesion_mask) == 0:
        return 1.0
    return np.sum(pred_mask & gt_lesion_mask) / np.sum(gt_lesion_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyse_subject(
    pred_data: np.ndarray,
    gt_data: np.ndarray,
    voxel_spacing: Tuple[float, ...],
    experiment: str,
    subject: str,
) -> List[Dict[str, Any]]:
    """
    Extract connected components from GT, match to prediction, compute
    per-lesion metrics.

    Returns one row per GT lesion.
    """
    voxel_vol_mm3 = float(np.prod(voxel_spacing))
    voxel_vol_ml = voxel_vol_mm3 / 1000.0

    gt_labels, n_gt = connected_components(gt_data.astype(bool))
    pred_bool = pred_data.astype(bool)

    rows: List[Dict[str, Any]] = []
    for lesion_id in range(1, n_gt + 1):
        gt_lesion = gt_labels == lesion_id
        vol_voxels = int(np.sum(gt_lesion))
        vol_ml = vol_voxels * voxel_vol_ml
        size_cat = classify_size(vol_voxels)

        # Check if this GT lesion is detected (>= 1 voxel overlap with pred)
        overlap_voxels = int(np.sum(pred_bool & gt_lesion))
        detected = overlap_voxels > 0

        # Sensitivity (recall) for this lesion: what fraction is captured?
        recall = lesion_overlap(pred_bool, gt_lesion)

        # For a finer Dice, restrict prediction to the neighbourhood of
        # this GT lesion (dilated bounding box) to compute a local Dice.
        # This avoids penalising the prediction for lesions elsewhere.
        coords = np.argwhere(gt_lesion)
        margin = 5  # voxels
        slices = tuple(
            slice(max(0, coords[:, d].min() - margin),
                  min(gt_data.shape[d], coords[:, d].max() + margin + 1))
            for d in range(3)
        )
        local_pred = pred_bool[slices]
        local_gt = gt_lesion[slices]
        local_dice = lesion_dice(local_pred, local_gt)

        rows.append({
            "experiment": experiment,
            "subject": subject,
            "lesion_id": lesion_id,
            "size_category": size_cat,
            "volume_voxels": vol_voxels,
            "volume_ml": round(vol_ml, 4),
            "detected": int(detected),
            "overlap_voxels": overlap_voxels,
            "recall": round(recall, 4),
            "local_dice": round(local_dice, 4),
        })

    # Also count false-positive lesion components (present in pred but
    # not matching any GT lesion).  We label pred CCs and check for GT overlap.
    pred_labels, n_pred = connected_components(pred_bool)
    fp_count = 0
    fp_vol_total = 0
    for pid in range(1, n_pred + 1):
        pred_cc = pred_labels == pid
        if np.sum(pred_cc & gt_data.astype(bool)) == 0:
            fp_count += 1
            fp_vol_total += int(np.sum(pred_cc))

    # Attach false-positive summary as a special row
    rows.append({
        "experiment": experiment,
        "subject": subject,
        "lesion_id": -1,  # sentinel
        "size_category": "FP_summary",
        "volume_voxels": fp_vol_total,
        "volume_ml": round(fp_vol_total * voxel_vol_ml, 4),
        "detected": fp_count,  # reuse field = number of FP CCs
        "overlap_voxels": 0,
        "recall": 0.0,
        "local_dice": 0.0,
    })

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate summary
# ──────────────────────────────────────────────────────────────────────────────
def build_summary(
    all_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-experiment × per-size-category."""
    grouped: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        if r["size_category"] == "FP_summary":
            continue
        grouped[r["experiment"]][r["size_category"]].append(r)

    summary: List[Dict[str, Any]] = []
    for exp in sorted(grouped):
        for cat in ("small", "medium", "large"):
            lesions = grouped[exp].get(cat, [])
            n = len(lesions)
            if n == 0:
                summary.append({
                    "experiment": exp,
                    "size_category": cat,
                    "n_lesions": 0,
                    "detection_rate": 0.0,
                    "mean_recall": 0.0,
                    "mean_local_dice": 0.0,
                    "median_volume_ml": 0.0,
                })
                continue

            summary.append({
                "experiment": exp,
                "size_category": cat,
                "n_lesions": n,
                "detection_rate": round(np.mean([r["detected"] for r in lesions]), 4),
                "mean_recall": round(np.mean([r["recall"] for r in lesions]), 4),
                "mean_local_dice": round(np.mean([r["local_dice"] for r in lesions]), 4),
                "median_volume_ml": round(np.median([r["volume_ml"] for r in lesions]), 4),
            })

    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    """Pretty-print the summary table."""
    header = (
        f"{'Experiment':<40} {'Size':<8} {'N':>6} "
        f"{'DetRate':>8} {'Recall':>8} {'L.Dice':>8} {'Med Vol(mL)':>12}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in summary:
        print(
            f"{r['experiment']:<40} {r['size_category']:<8} {r['n_lesions']:>6} "
            f"{r['detection_rate']:>8.4f} {r['mean_recall']:>8.4f} "
            f"{r['mean_local_dice']:>8.4f} {r['median_volume_ml']:>12.4f}"
        )
    print(sep)


def compute_optimal_weights(summary: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    For each dataset (DS500, DS501), determine the optimal weight for each
    size category based on per-lesion recall.

    Returns a dict like:
        { "DS500": {"small": 0.65, "medium": 0.50, "large": 0.35},
          "DS501": {"small": 0.60, "medium": 0.50, "large": 0.40} }

    where the weight refers to the CNN model (ResEncL gets 1 - w).
    """
    # Group by dataset prefix
    per_ds: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in summary:
        exp = r["experiment"]
        # Extract dataset prefix (DS500 or DS501) and model tag (CNN or ResEncL)
        parts = exp.split("_")
        ds_prefix = parts[0]          # "DS500" or "DS501"
        model_tag = parts[-1]         # "CNN" or "ResEncL"
        cat = r["size_category"]
        per_ds[ds_prefix][(model_tag, cat)] = {
            "recall": r["mean_recall"],
            "dice": r["mean_local_dice"],
            "det_rate": r["detection_rate"],
        }

    optimal: Dict[str, Dict[str, float]] = {}
    for ds in sorted(per_ds):
        weights: Dict[str, float] = {}
        for cat in ("small", "medium", "large"):
            cnn_metrics = per_ds[ds].get(("CNN", cat), {})
            res_metrics = per_ds[ds].get(("ResEncL", cat), {})

            # Composite score: 0.5 * recall + 0.3 * dice + 0.2 * detection_rate
            cnn_score = (
                0.5 * cnn_metrics.get("recall", 0)
                + 0.3 * cnn_metrics.get("dice", 0)
                + 0.2 * cnn_metrics.get("det_rate", 0)
            )
            res_score = (
                0.5 * res_metrics.get("recall", 0)
                + 0.3 * res_metrics.get("dice", 0)
                + 0.2 * res_metrics.get("det_rate", 0)
            )

            total = cnn_score + res_score
            if total == 0:
                weights[cat] = 0.5
            else:
                weights[cat] = round(cnn_score / total, 4)

        optimal[ds] = weights

    return optimal


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(config_path: Optional[str] = None) -> None:
    init_thesis_logging(log_dir="logs")
    setup_logging(log_file="logs/lesion_size_analysis.log")
    cfg = load_config(config_path)
    raw_base, _, results_base = get_nnunet_env_paths()
    raw_base = Path(raw_base)
    results_base = Path(results_base)

    predictions_base = results_base / "predictions"
    ensemble_base = results_base / "ensembles"

    datasets = {
        500: {
            "name": "RawFLAIR",
            "labels_ts": raw_base / "Dataset500_RawFLAIR" / "labelsTs",
        },
        501: {
            "name": "SkullStrippedFLAIR",
            "labels_ts": raw_base / "Dataset501_SkullStrippedFLAIR" / "labelsTs",
        },
    }

    all_rows: List[Dict[str, Any]] = []

    with PhaseTimer("Lesion-size stratified analysis"):
        for ds_id, ds_info in datasets.items():
            gt_dir = ds_info["labels_ts"]
            if not gt_dir.is_dir():
                logger.warning("No labelsTs for dataset %d -> skipping.", ds_id)
                continue

            gt_files = glob_nifti(gt_dir, suffix=".nii.gz")
            gt_map = {strip_nifti_ext(f.name): f for f in gt_files}

            for tag in ("CNN", "ResEncL"):
                pred_dir = predictions_base / f"DS{ds_id}_{tag}"
                if not pred_dir.is_dir():
                    logger.info("  Predictions not found: %s -> skipping.", pred_dir)
                    continue

                exp_name = f"DS{ds_id}_{ds_info['name']}_{tag}"
                logger.info("Analysing lesion sizes: %s", exp_name)

                pred_files = glob_nifti(pred_dir, suffix=".nii.gz")
                for pf in pred_files:
                    stem = strip_nifti_ext(pf.name)
                    if stem not in gt_map:
                        continue

                    pred_nii = nib.load(str(pf))
                    gt_nii = nib.load(str(gt_map[stem]))
                    pred_data = (np.asarray(pred_nii.dataobj) > 0).astype(np.uint8)
                    gt_data = (np.asarray(gt_nii.dataobj) > 0).astype(np.uint8)
                    spacing = tuple(float(v) for v in pred_nii.header.get_zooms()[:3])

                    subject_rows = analyse_subject(
                        pred_data, gt_data, spacing, exp_name, stem
                    )
                    all_rows.extend(subject_rows)

    if not all_rows:
        logger.warning("No predictions found to analyse.")
        return

    # ── Save per-lesion CSV ──────────────────────────────────────────────
    csv_path = results_base / "lesion_size_analysis.csv"
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Per-lesion metrics saved to %s", csv_path)

    # ── Summary ──────────────────────────────────────────────────────────
    summary = build_summary(all_rows)
    print_summary(summary)

    summary_csv = results_base / "lesion_size_summary.csv"
    fieldnames_s = list(summary[0].keys())
    with open(summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames_s)
        writer.writeheader()
        writer.writerows(summary)
    logger.info("Summary saved to %s", summary_csv)

    # ── Compute optimal ensemble weights ─────────────────────────────────
    optimal_weights = compute_optimal_weights(summary)
    print("\n" + "=" * 60)
    print("  OPTIMAL ENSEMBLE WEIGHTS (CNN weight; ResEncL = 1 - w)")
    print("=" * 60)
    for ds, weights in optimal_weights.items():
        print(f"\n  {ds}:")
        for cat, w in weights.items():
            print(f"    {cat:>8s}:  CNN = {w:.4f}  |  ResEncL = {1-w:.4f}")
    print()

    # ── Save weights as JSON for adaptive_ensemble.py ────────────────────
    import json
    weights_path = results_base / "optimal_ensemble_weights.json"
    with open(weights_path, "w") as fh:
        json.dump(optimal_weights, fh, indent=2)
    logger.info("Optimal weights saved to %s", weights_path)
    logger.info(
        "Run adaptive_ensemble.py to generate scale-aware ensemble predictions."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lesion-size stratified analysis of segmentation models."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset_config.yaml")
    args = parser.parse_args()
    main(args.config)
