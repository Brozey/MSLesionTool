#!/usr/bin/env python3
"""
Evaluate model performance on DS002 (WMH) broken down by center.
Sites: Amsterdam (50), Singapore (30), Utrecht (30).
Reads mapping from data/nnUNet_raw/Dataset002_WMH/mapping.json.
Output printed to console and saved to results/evaluation/.
"""
import os
import sys
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
try:
    from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures
except ImportError:
    print("[ERROR] MetricsReloaded not installed. Evaluating with basic Dice.")
    BinaryPairwiseMeasures = None

def basic_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    if intersection == 0:
        return 0.0
    return 2. * intersection / (pred.sum() + gt.sum())

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_JSON = REPO_ROOT / "data/nnUNet_raw/Dataset002_WMH/mapping.json"
DS002_LABELS = REPO_ROOT / "data/nnUNet_raw/Dataset002_WMH/labelsTs"
PREDICTIONS_DIR = REPO_ROOT / "results/predictions"
EVAL_OUT = REPO_ROOT / "results/evaluation/per_center_ds002_analysis.csv"

# Models to evaluate (pre-computed .nii.gz predictions)
MODELS_TO_EVALUATE = [
    "CNN_3D_5fold_TTA_DS002",
    "25D_5fold_ensemble_TTA_DS002_thr0.40",
    "25D_5fold_ensemble_TTA_DS002",
    "ResEncL_3D_5fold_TTA_DS002",
]

# Best-2/arch ensemble (on-the-fly from softmax .npz files)
BEST2_ARCH_MODELS = [
    "CNN_3D_fold1",
    "CNN_3D_fold3",
    "ResEncL_3D_fold1",
    "ResEncL_3D_fold3",
    "25D_fold1",
    "25D_fold3",
]
BEST2_ARCH_THRESHOLD = 0.50

def load_mapping():
    with open(MAPPING_JSON, 'r') as f:
        mapping = json.load(f)

    # Create dict of case_id -> site for test cases
    case_to_site = {}
    for case in mapping:
        if case.get('split') == 'test':
            case_to_site[case['case_id']] = case.get('site', 'Unknown')
    return case_to_site

def evaluate_model_per_center(model_dir, case_to_site):
    if not model_dir.exists():
        print(f"  [WARN] Model directory missing: {model_dir.name}")
        return None

    print(f"Evaluating: {model_dir.name}")

    site_metrics = defaultdict(list)

    pred_files = sorted(model_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"  [WARN] No predictions found in: {model_dir.name}")
        return None

    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")
        if case_id not in case_to_site:
            continue

        site = case_to_site[case_id]
        gt_path = DS002_LABELS / pred_path.name

        if not gt_path.exists():
            continue

        pred_img = nib.load(str(pred_path))
        pred = (pred_img.get_fdata() > 0).astype(np.uint8)

        gt_img = nib.load(str(gt_path))
        gt = (gt_img.get_fdata() > 0).astype(np.uint8)

        if BinaryPairwiseMeasures:
            pixdim = list(gt_img.header.get_zooms()[:3])
            bpm = BinaryPairwiseMeasures(pred, gt, connectivity_type=1, pixdim=pixdim)
            dice = bpm.dsc()
        else:
            dice = basic_dice(pred, gt)

        site_metrics[site].append(dice)

    # Calculate means
    results = {"Model": model_dir.name}
    all_dices = []

    for site, dices in site_metrics.items():
        mean_dice = float(np.mean(dices))
        results[f"{site} Dice"] = mean_dice
        results[f"{site} N"] = len(dices)
        all_dices.extend(dices)
        print(f"  {site:12s} N={len(dices):<3d} Dice: {mean_dice:.4f} ± {np.std(dices):.4f}")

    if all_dices:
        results["Overall Dice"] = float(np.mean(all_dices))
        print(f"  {'Overall':12s} N={len(all_dices):<3d} Dice: {results['Overall Dice']:.4f}")

    return results


def load_prob(npz_path):
    """Load softmax probability for foreground class from .npz file."""
    d = np.load(npz_path)
    key = "softmax" if "softmax" in d.files else "probabilities"
    arr = d[key][1].astype(np.float32)
    return np.transpose(arr, (2, 1, 0))


def evaluate_ensemble_per_center(model_names, threshold, case_to_site):
    """Evaluate on-the-fly ensemble from softmax .npz files, broken down by center."""
    label = f"Best-2/arch ({len(model_names)} models)"
    print(f"Evaluating: {label}")

    site_metrics = defaultdict(list)
    gt_files = sorted(DS002_LABELS.glob("*.nii.gz"))

    for gt_path in gt_files:
        case_id = gt_path.name.replace(".nii.gz", "")
        if case_id not in case_to_site:
            continue

        site = case_to_site[case_id]

        # Load and average softmax predictions
        probs = []
        for mn in model_names:
            pred_dir = PREDICTIONS_DIR / f"{mn}_TTA_DS002"
            npz = pred_dir / f"{case_id}.npz"
            if not npz.exists():
                npz = pred_dir / f"{case_id}.nii.gz.npz"
            if not npz.exists():
                continue
            probs.append(load_prob(npz))

        if len(probs) != len(model_names):
            continue

        avg_prob = np.mean(probs, axis=0)
        pred = (avg_prob >= threshold).astype(np.uint8)

        gt_img = nib.load(str(gt_path))
        gt = (gt_img.get_fdata() > 0).astype(np.uint8)

        if BinaryPairwiseMeasures:
            pixdim = list(gt_img.header.get_zooms()[:3])
            bpm = BinaryPairwiseMeasures(pred, gt, connectivity_type=1, pixdim=pixdim)
            dice = bpm.dsc()
        else:
            dice = basic_dice(pred, gt)

        site_metrics[site].append(dice)

    results = {"Model": label}
    all_dices = []

    for site, dices in site_metrics.items():
        mean_dice = float(np.mean(dices))
        results[f"{site} Dice"] = mean_dice
        results[f"{site} N"] = len(dices)
        all_dices.extend(dices)
        print(f"  {site:12s} N={len(dices):<3d} Dice: {mean_dice:.4f} ± {np.std(dices):.4f}")

    if all_dices:
        results["Overall Dice"] = float(np.mean(all_dices))
        print(f"  {'Overall':12s} N={len(all_dices):<3d} Dice: {results['Overall Dice']:.4f}")

    return results


def main():
    print("Loading test cases mapping...")
    case_to_site = load_mapping()
    print(f"Found {len(case_to_site)} test cases mapped to sites.")
    for site, count in pd.Series(list(case_to_site.values())).value_counts().items():
        print(f"  {site}: {count}")
    print()

    all_results = []

    for model_name in MODELS_TO_EVALUATE:
        model_dir = PREDICTIONS_DIR / model_name
        res = evaluate_model_per_center(model_dir, case_to_site)
        if res is not None:
            all_results.append(res)
            print()

    # Evaluate Best-2/arch ensemble (on-the-fly from softmax)
    res = evaluate_ensemble_per_center(BEST2_ARCH_MODELS, BEST2_ARCH_THRESHOLD, case_to_site)
    if res is not None:
        all_results.append(res)
        print()

    if all_results:
        df = pd.DataFrame(all_results)

        # Order columns logically
        base_cols = ["Model", "Overall Dice"]
        site_cols = []
        for site in sorted(set(s for s in case_to_site.values())):
            if f"{site} Dice" in df.columns:
                site_cols.extend([f"{site} Dice", f"{site} N"])

        df = df[base_cols + site_cols]
        df = df.sort_values("Overall Dice", ascending=False)

        EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(EVAL_OUT, index=False)
        print(f"\nSaved results to: {EVAL_OUT}")
        print("\nSummary Table:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
