#!/usr/bin/env python3
"""
evaluate_test_sets.py
=====================
Evaluate locally trained ResEncL 3D models on DS001 and DS002 test sets.
Computes standard metrics using MetricsReloaded and compares to challenge baselines.

Metrics computed:
    DSC   - Dice Similarity Coefficient
    HD95  - 95th-percentile Hausdorff Distance (mm)
    AVD   - Absolute Volume Difference ratio
    Sens  - Sensitivity (Recall)
    PPV   - Positive Predictive Value (Precision)
    F1    - F-beta (beta=1)
    NSD   - Normalised Surface Distance (tau=2mm)

Challenge references:
    DS001 (MSLesSeg-2024): Guarnera et al., Sci Data 2025 — nnU-Net V2 baseline
    DS002 (WMH Challenge 2017): Kuijf et al., IEEE TMI 2019 — top methods

Usage:
    python scripts/evaluation/evaluate_test_sets.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel required: pip install nibabel")

try:
    from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures
except ImportError:
    raise ImportError("MetricsReloaded required: pip install MetricsReloaded")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_BASE = REPO_ROOT / "results"

RAW_BASE = Path(os.environ.get(
    "nnUNet_raw",
    str(REPO_ROOT / "data" / "nnUNet_raw")
))

NSD_TOLERANCE_MM = 2.0
HD_PERCENTILE = 95

DATASETS = {
    "DS001_ResEncL_3D": {
        "pred_dir": RESULTS_BASE / "predictions" / "DS001_ResEncL_3D",
        "gt_dir": RAW_BASE / "Dataset001_MSLesSeg" / "labelsTs",
        "dataset_name": "DS001 (MSLesSeg-2024)",
        "challenge_name": "MSLesSeg-2024",
        "challenge_ref": "Guarnera et al., Sci Data 2025",
        # MSLesSeg-2024 nnU-Net V2 baseline from their paper
        # They report DSC ~0.614 for their nnU-Net baseline
        "challenge_metrics": {
            "dice": 0.614,
            "hd95": None,  # not reported in same format
            "ppv": None,
            "sensitivity": None,
        },
    },
    "DS002_ResEncL_3D": {
        "pred_dir": RESULTS_BASE / "predictions" / "DS002_ResEncL_3D",
        "gt_dir": RAW_BASE / "Dataset002_WMH" / "labelsTs",
        "dataset_name": "DS002 (WMH Challenge 2017)",
        "challenge_name": "WMH Challenge 2017",
        "challenge_ref": "Kuijf et al., IEEE TMI 2019",
        # WMH Challenge 2017 top-team results (training set eval)
        # Top method "sysu_media": DSC~0.80, H95~6.3mm, AVD~0.13, Recall~0.84
        # Method "k2": DSC~0.79, H95~7.6mm
        # Overall top-5 average: DSC~0.78
        "challenge_metrics": {
            "dice": 0.80,          # sysu_media (winner)
            "hd95": 6.30,          # sysu_media H95
            "avd": 0.13,           # sysu_media AVD
            "sensitivity": 0.84,   # sysu_media recall
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def safe_metric(func, pred, gt):
    """Call a MetricsReloaded metric function, returning NaN on failure."""
    pred_empty = not np.any(pred)
    gt_empty = not np.any(gt)
    if pred_empty and gt_empty:
        fname = getattr(func, "__name__", str(func))
        if any(kw in fname for kw in ("hausdorff", "masd", "distance", "volume")):
            return 0.0
        return 1.0
    try:
        val = func()
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return float("nan")
        return float(val)
    except Exception as e:
        print(f"  [WARN] Metric {func} failed: {e}")
        return float("nan")


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    pixdim: list[float],
    nsd_tolerance: float = NSD_TOLERANCE_MM,
) -> dict[str, float]:
    """Compute all evaluation metrics for a single prediction vs GT pair."""
    bpm = BinaryPairwiseMeasures(
        pred, gt,
        connectivity_type=1,
        pixdim=pixdim,
        dict_args={"hd_perc": HD_PERCENTILE, "nsd": nsd_tolerance},
    )

    dsc = safe_metric(bpm.dsc, pred, gt)
    iou = safe_metric(bpm.intersection_over_union, pred, gt)
    sens = safe_metric(bpm.sensitivity, pred, gt)
    ppv = safe_metric(bpm.positive_predictive_value, pred, gt)
    fb = safe_metric(bpm.fbeta, pred, gt)
    hd95 = safe_metric(bpm.measured_hausdorff_distance_perc, pred, gt)
    masd = safe_metric(bpm.measured_masd, pred, gt)
    nsd = safe_metric(bpm.normalised_surface_distance, pred, gt)

    # Compute AVD manually to avoid MetricsReloaded integer overflow
    n_pred = float(np.sum(pred))
    n_gt = float(np.sum(gt))
    if n_gt > 0:
        avd = abs(n_pred - n_gt) / n_gt
    elif n_pred == 0:
        avd = 0.0
    else:
        avd = float("nan")

    voxel_vol_ml = float(np.prod(pixdim)) / 1000.0
    pred_vol = float(np.sum(pred)) * voxel_vol_ml
    gt_vol = float(np.sum(gt)) * voxel_vol_ml

    return {
        "dice": dsc, "iou": iou,
        "sensitivity": sens, "ppv": ppv, "fbeta": fb,
        "hd95": hd95, "masd": masd, "nsd": nsd,
        "avd": avd,
        "pred_volume_ml": pred_vol, "gt_volume_ml": gt_vol,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation driver
# ──────────────────────────────────────────────────────────────────────────────

METRIC_KEYS = [
    "dice", "iou", "sensitivity", "ppv", "fbeta",
    "hd95", "masd", "nsd", "avd",
    "pred_volume_ml", "gt_volume_ml",
]


def evaluate_dataset(experiment_name: str, config: dict) -> list[dict]:
    """Evaluate all predictions for a dataset. Returns per-subject rows."""
    pred_dir = config["pred_dir"]
    gt_dir = config["gt_dir"]

    if not pred_dir.is_dir():
        print(f"  [SKIP] Prediction directory not found: {pred_dir}")
        return []
    if not gt_dir.is_dir():
        print(f"  [SKIP] Ground truth directory not found: {gt_dir}")
        return []

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    gt_map = {f.name: f for f in gt_dir.glob("*.nii.gz")}

    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {config['dataset_name']}")
    print(f"  Predictions: {pred_dir}")
    print(f"  Ground truth: {gt_dir}")
    print(f"  Cases: {len(pred_files)} predictions, {len(gt_map)} GT labels")
    print(f"{'=' * 70}")

    results = []
    for i, pf in enumerate(pred_files):
        stem = pf.name
        if stem not in gt_map:
            print(f"  [WARN] No GT for {pf.name}")
            continue

        pred_nii = nib.load(str(pf))
        gt_nii = nib.load(str(gt_map[stem]))
        pred_data = (np.asarray(pred_nii.dataobj) > 0).astype(np.uint8)
        gt_data = (np.asarray(gt_nii.dataobj) > 0).astype(np.uint8)
        pixdim = [float(v) for v in pred_nii.header.get_zooms()[:3]]

        metrics = compute_metrics(pred_data, gt_data, pixdim)
        row = {"experiment": experiment_name, "subject": stem, **metrics}
        results.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == len(pred_files):
            print(f"  [{i+1}/{len(pred_files)}] {stem}  "
                  f"DSC={metrics['dice']:.4f}  HD95={metrics['hd95']:.2f}mm  "
                  f"Sens={metrics['sensitivity']:.4f}  PPV={metrics['ppv']:.4f}")

    return results


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """Print formatted summary comparing our results to challenge baselines."""
    print("\n")
    print("=" * 90)
    print("  RESULTS SUMMARY — ResEncL 3D on Test Sets")
    print("=" * 90)

    for exp_name, rows in all_results.items():
        if not rows:
            continue
        cfg = DATASETS[exp_name]
        n = len(rows)

        def mean_f(key):
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            return np.mean(vals) if vals else float("nan")

        def std_f(key):
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            return np.std(vals) if vals else float("nan")

        def median_f(key):
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            return np.median(vals) if vals else float("nan")

        print(f"\n{'─' * 90}")
        print(f"  {cfg['dataset_name']} — {n} test subjects")
        print(f"  Challenge: {cfg['challenge_name']} ({cfg['challenge_ref']})")
        print(f"{'─' * 90}")

        header = f"{'Metric':<15} {'Ours (mean±std)':>20} {'Ours (median)':>15} {'Challenge':>15}"
        print(header)
        print("-" * 70)

        display_metrics = [
            ("DSC", "dice"),
            ("HD95 (mm)", "hd95"),
            ("Sensitivity", "sensitivity"),
            ("PPV", "ppv"),
            ("F1", "fbeta"),
            ("NSD", "nsd"),
            ("AVD", "avd"),
            ("MASD (mm)", "masd"),
        ]

        ch = cfg.get("challenge_metrics", {})
        for label, key in display_metrics:
            m = mean_f(key)
            s = std_f(key)
            med = median_f(key)
            ch_val = ch.get(key)
            ch_str = f"{ch_val:.4f}" if ch_val is not None else "—"

            # Highlight if we beat the challenge
            marker = ""
            if ch_val is not None and np.isfinite(m):
                if key in ("hd95", "masd", "avd"):  # lower is better
                    marker = " ✓" if m < ch_val else ""
                else:  # higher is better
                    marker = " ✓" if m > ch_val else ""

            print(f"  {label:<13} {m:>8.4f} ± {s:<8.4f} {med:>13.4f}  {ch_str:>13}{marker}")

        # Volume stats
        pred_vol = mean_f("pred_volume_ml")
        gt_vol = mean_f("gt_volume_ml")
        print(f"\n  Mean pred volume: {pred_vol:.2f} mL | Mean GT volume: {gt_vol:.2f} mL")


def save_csv(all_results: dict[str, list[dict]], output_dir: Path) -> None:
    """Save per-subject and summary CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-subject CSV
    all_rows = []
    for rows in all_results.values():
        all_rows.extend(rows)

    if all_rows:
        csv_path = output_dir / "test_set_metrics.csv"
        fieldnames = ["experiment", "subject"] + METRIC_KEYS
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Per-subject metrics saved to: {csv_path}")

    # Summary CSV
    summary_rows = []
    for exp_name, rows in all_results.items():
        if not rows:
            continue
        n = len(rows)
        agg = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            agg[f"{key}_mean"] = np.mean(vals) if vals else float("nan")
            agg[f"{key}_std"] = np.std(vals) if vals else float("nan")
            agg[f"{key}_median"] = np.median(vals) if vals else float("nan")
        summary_rows.append({"experiment": exp_name, "n_subjects": n, **agg})

    if summary_rows:
        csv_path = output_dir / "test_set_summary.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Summary metrics saved to: {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MS Lesion Segmentation — Test Set Evaluation")
    print("  MetricsReloaded (Maier-Hein et al., Nature Methods 2024)")
    print("=" * 70)

    t0 = time.time()
    all_results = {}

    for exp_name, cfg in DATASETS.items():
        rows = evaluate_dataset(exp_name, cfg)
        all_results[exp_name] = rows

    elapsed = time.time() - t0
    print(f"\nTotal evaluation time: {elapsed:.1f}s")

    print_summary(all_results)
    save_csv(all_results, RESULTS_BASE / "evaluation")

    print("\n" + "=" * 70)
    print("  Metrics computed using MetricsReloaded")
    print("  Maier-Hein, Reinke, Godau et al.")
    print("  'Metrics Reloaded: Recommendations for image analysis validation'")
    print("  Nature Methods, 2024")
    print("=" * 70)


if __name__ == "__main__":
    main()
