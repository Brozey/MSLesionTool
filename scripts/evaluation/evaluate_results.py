#!/usr/bin/env python3
"""
evaluate_results.py
===================
Compare segmentation performance across all ablation experiments using
**MetricsReloaded** -- a standardised framework for biomedical image
analysis validation (Maier-Hein et al., Nature Methods 2024).

Computed metrics (per-subject)
------------------------------
Overlap:
    DSC   -- Dice similarity coefficient
    IoU   -- Intersection over Union (Jaccard)
    clDSC -- Centreline Dice (topology-aware)

Detection:
    Sensitivity (Recall / TPR)
    PPV         (Precision)
    F-beta      (F1 by default)

Boundary / Distance:
    HD95  -- 95th-percentile Hausdorff distance  (mm)
    MASD  -- Mean average surface distance       (mm)
    NSD   -- Normalised surface distance  (tau = 2 mm)

Volume:
    AVD   -- Absolute volume difference ratio
    pred_volume_ml / gt_volume_ml

Outputs:
    <nnUNet_results>/ablation_metrics.csv       per-subject metrics
    <nnUNet_results>/ablation_summary.csv       per-experiment aggregates
    Console summary table

Usage:
    python evaluate_results.py
    python evaluate_results.py --config config/dataset_config.yaml
    python evaluate_results.py --nsd-tolerance 1.0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

try:
    from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures
except ImportError:
    raise ImportError(
        "MetricsReloaded is required:  pip install MetricsReloaded\n"
        "  https://github.com/csudre/MetricsReloaded"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
NSD_TOLERANCE_MM = 2.0   # default tolerance for Normalised Surface Distance
HD_PERCENTILE = 95       # Hausdorff percentile

# Metrics to compute -- order matches CSV columns
METRIC_NAMES = [
    "dice", "iou", "cldsc",
    "sensitivity", "ppv", "fbeta",
    "hd95", "masd", "nsd",
    "avd", "pred_volume_ml", "gt_volume_ml",
]


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation  (MetricsReloaded)
# ──────────────────────────────────────────────────────────────────────────────
def safe_metric(func, pred, gt):
    """Call a MetricsReloaded metric function, returning NaN on failure."""
    pred_empty = not np.any(pred)
    gt_empty = not np.any(gt)
    if pred_empty and gt_empty:
        fname = getattr(func, "__name__", str(func))
        if any(kw in fname for kw in ("hausdorff", "masd", "distance", "volume")):
            return 0.0
        return 1.0  # DSC, IoU, etc. = 1.0 for empty-vs-empty
    try:
        val = func()
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return float("nan")
        return float(val)
    except Exception as e:
        logger.debug("Metric %s failed: %s", func, e)
        return float("nan")


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    pixdim: list[float],
    nsd_tolerance: float = NSD_TOLERANCE_MM,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single binary prediction vs
    ground-truth pair using MetricsReloaded.

    Parameters
    ----------
    pred : binary uint8 array (H, W, D)
    gt   : binary uint8 array (H, W, D)
    pixdim : voxel spacing in mm  [sx, sy, sz]
    nsd_tolerance : tolerance in mm for NSD

    Returns
    -------
    dict with metric name -> value
    """
    bpm = BinaryPairwiseMeasures(
        pred, gt,
        connectivity_type=1,
        pixdim=pixdim,
        dict_args={
            "hd_perc": HD_PERCENTILE,
            "nsd": nsd_tolerance,
        },
    )

    # -- Overlap --
    dsc = safe_metric(bpm.dsc, pred, gt)
    iou = safe_metric(bpm.intersection_over_union, pred, gt)
    cldsc = safe_metric(bpm.centreline_dsc, pred, gt)

    # -- Detection --
    sens = safe_metric(bpm.sensitivity, pred, gt)
    ppv = safe_metric(bpm.positive_predictive_value, pred, gt)
    fb = safe_metric(bpm.fbeta, pred, gt)

    # -- Boundary / distance --
    hd95 = safe_metric(bpm.measured_hausdorff_distance_perc, pred, gt)
    masd = safe_metric(bpm.measured_masd, pred, gt)
    nsd = safe_metric(bpm.normalised_surface_distance, pred, gt)

    # -- Volume --
    avd = safe_metric(bpm.absolute_volume_difference_ratio, pred, gt)
    voxel_vol_ml = float(np.prod(pixdim)) / 1000.0  # mm^3 -> mL
    pred_vol = float(np.sum(pred)) * voxel_vol_ml
    gt_vol = float(np.sum(gt)) * voxel_vol_ml

    return {
        "dice": dsc,
        "iou": iou,
        "cldsc": cldsc,
        "sensitivity": sens,
        "ppv": ppv,
        "fbeta": fb,
        "hd95": hd95,
        "masd": masd,
        "nsd": nsd,
        "avd": avd,
        "pred_volume_ml": pred_vol,
        "gt_volume_ml": gt_vol,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation driver
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_folder(
    pred_dir: Path,
    gt_dir: Path,
    experiment_name: str,
    nsd_tolerance: float = NSD_TOLERANCE_MM,
) -> List[Dict[str, Any]]:
    """
    Evaluate all predicted segmentations in *pred_dir* against ground-truth
    labels in *gt_dir*.  Returns a list of per-subject metric dicts.
    """
    pred_files = glob_nifti(pred_dir, suffix=".nii.gz")
    gt_files = glob_nifti(gt_dir, suffix=".nii.gz")

    # Build lookup: stem -> path
    gt_map: Dict[str, Path] = {}
    for f in gt_files:
        gt_map[strip_nifti_ext(f.name)] = f

    results: List[Dict[str, Any]] = []
    for pf in sorted(pred_files):
        stem = strip_nifti_ext(pf.name)
        if stem not in gt_map:
            logger.warning("  No ground truth for prediction: %s", pf.name)
            continue

        pred_nii = nib.load(str(pf))
        gt_nii = nib.load(str(gt_map[stem]))
        pred_data = (np.asarray(pred_nii.dataobj) > 0).astype(np.uint8)
        gt_data = (np.asarray(gt_nii.dataobj) > 0).astype(np.uint8)

        pixdim = [float(v) for v in pred_nii.header.get_zooms()[:3]]

        metrics = compute_metrics(pred_data, gt_data, pixdim, nsd_tolerance)
        row = {"experiment": experiment_name, "subject": stem, **metrics}
        results.append(row)

        logger.info(
            "  %s  DSC=%.4f  IoU=%.4f  HD95=%.2f  NSD=%.4f",
            stem, metrics["dice"], metrics["iou"],
            metrics["hd95"], metrics["nsd"],
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
def print_summary_table(all_results: List[Dict[str, Any]]) -> list:
    """Print a formatted summary table grouped by experiment.
    Returns list of summary row dicts for CSV export."""
    from collections import defaultdict

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_results:
        grouped[r["experiment"]].append(r)

    # Header
    cols = ["Experiment", "N", "DSC", "IoU", "clDSC", "Sens", "PPV",
            "F1", "HD95", "MASD", "NSD", "AVD"]
    widths = [35, 4, 8, 8, 8, 8, 8, 8, 10, 10, 8, 8]
    header = "  ".join(f"{c:>{w}}" if i > 0 else f"{c:<{w}}"
                       for i, (c, w) in enumerate(zip(cols, widths)))
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    summary_rows = []
    for exp_name in sorted(grouped.keys()):
        rows = grouped[exp_name]
        n = len(rows)

        def mean_finite(key):
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            return np.mean(vals) if vals else float("nan")

        def std_finite(key):
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            return np.std(vals) if vals else float("nan")

        agg = {k: mean_finite(k) for k in METRIC_NAMES}
        agg_std = {k: std_finite(k) for k in METRIC_NAMES}

        print(
            f"{exp_name:<35}  {n:>4}  "
            f"{agg['dice']:>8.4f}  {agg['iou']:>8.4f}  {agg['cldsc']:>8.4f}  "
            f"{agg['sensitivity']:>8.4f}  {agg['ppv']:>8.4f}  {agg['fbeta']:>8.4f}  "
            f"{agg['hd95']:>10.2f}  {agg['masd']:>10.4f}  "
            f"{agg['nsd']:>8.4f}  {agg['avd']:>8.4f}"
        )

        summary_rows.append({
            "experiment": exp_name,
            "n_subjects": n,
            **{f"{k}_mean": agg[k] for k in METRIC_NAMES},
            **{f"{k}_std": agg_std[k] for k in METRIC_NAMES},
        })

    print(sep)
    return summary_rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(
    config_path: Optional[str] = None,
    nsd_tolerance: float = NSD_TOLERANCE_MM,
) -> None:
    init_thesis_logging(log_dir="logs")
    cfg = load_config(config_path)
    _, _, results_base = get_nnunet_env_paths()
    raw_base = Path(get_nnunet_env_paths()[0])

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

    all_results: List[Dict[str, Any]] = []

    # ── Evaluate individual model predictions ────────────────────────────
    with PhaseTimer("Evaluate individual models"):
        for ds_id, ds_info in datasets.items():
            gt_dir = ds_info["labels_ts"]
            if not gt_dir.is_dir():
                logger.warning("No labelsTs for dataset %d -- skipping.", ds_id)
                continue

            for tag in ("CNN", "ResEncL"):
                pred_dir = predictions_base / f"DS{ds_id}_{tag}"
                if not pred_dir.is_dir():
                    logger.info("  Predictions not found: %s -- skipping.",
                                pred_dir)
                    continue
                exp_name = f"DS{ds_id}_{ds_info['name']}_{tag}"
                logger.info(
                    "Evaluating %s  (%d subjects) ...", exp_name,
                    len(glob_nifti(pred_dir, suffix=".nii.gz")),
                )
                rows = evaluate_folder(
                    pred_dir, gt_dir, exp_name, nsd_tolerance
                )
                all_results.extend(rows)

    # ── Evaluate ensemble predictions ────────────────────────────────────
    with PhaseTimer("Evaluate ensemble models"):
        for ds_id, ds_info in datasets.items():
            gt_dir = ds_info["labels_ts"]
            if not gt_dir.is_dir():
                continue

            for ens_suffix in (
                "CNN_plus_ResEncL",
            ):
                ens_dir = ensemble_base / f"DS{ds_id}_{ens_suffix}"
                if not ens_dir.is_dir():
                    continue
                exp_name = f"DS{ds_id}_{ds_info['name']}_{ens_suffix}"
                logger.info("Evaluating %s ...", exp_name)
                rows = evaluate_folder(
                    ens_dir, gt_dir, exp_name, nsd_tolerance
                )
                all_results.extend(rows)

    if not all_results:
        logger.warning("No predictions found to evaluate.")
        return

    # ── Print summary ────────────────────────────────────────────────────
    summary_rows = print_summary_table(all_results)

    # ── Write per-subject CSV ────────────────────────────────────────────
    csv_path = results_base / "ablation_metrics.csv"
    fieldnames = ["experiment", "subject"] + METRIC_NAMES
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("Per-subject metrics saved to %s", csv_path)

    # ── Write summary CSV ────────────────────────────────────────────────
    summary_csv = results_base / "ablation_summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(summary_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("Summary metrics saved to %s", summary_csv)

    # ── Citation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Metrics computed using MetricsReloaded")
    print("  Maier-Hein, Reinke, Godau et al.")
    print("  'Metrics Reloaded: Recommendations for image analysis validation'")
    print("  Nature Methods, 2024")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ablation results using MetricsReloaded."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dataset_config.yaml",
    )
    parser.add_argument(
        "--nsd-tolerance", type=float, default=NSD_TOLERANCE_MM,
        help=f"NSD tolerance in mm (default: {NSD_TOLERANCE_MM})",
    )
    args = parser.parse_args()
    main(args.config, args.nsd_tolerance)
