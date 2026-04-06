#!/usr/bin/env python3
"""Validation-based ensemble selection: pick models by val EMA Dice, evaluate on test."""
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from itertools import combinations
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_test_sets import compute_metrics

PRED = REPO_ROOT / "results" / "predictions"
RAW_BASE = Path(os.environ.get("nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw")))
GT_DS001 = RAW_BASE / "Dataset001_MSLesSeg" / "labelsTs"
GT_DS002 = RAW_BASE / "Dataset002_WMH" / "labelsTs"

# Validation EMA fg_dice for all 15 models (from checkpoint_best)
VAL_EMA = {
    "CNN_3D_fold0": 0.8291,
    "CNN_3D_fold1": 0.8418,
    "CNN_3D_fold2": 0.8242,
    "CNN_3D_fold3": 0.8356,
    "CNN_3D_fold4": 0.8259,
    "ResEncL_3D_fold0": 0.8135,
    "ResEncL_3D_fold1": 0.8430,
    "ResEncL_3D_fold2": 0.8334,
    "ResEncL_3D_fold3": 0.8344,
    "ResEncL_3D_fold4": 0.8317,
    "25D_fold0": 0.8385,
    "25D_fold1": 0.8582,
    "25D_fold2": 0.8507,
    "25D_fold3": 0.8522,
    "25D_fold4": 0.8476,
}

# Sort by val EMA descending
RANKED = sorted(VAL_EMA.items(), key=lambda x: -x[1])


def load_prob(npz_path):
    d = np.load(npz_path)
    key = "softmax" if "softmax" in d.files else "probabilities"
    arr = d[key][1].astype(np.float32)
    return np.transpose(arr, (2, 1, 0))


def fast_dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = float(np.sum(pred & gt))
    denom = 2 * tp + float(np.sum(pred & ~gt)) + float(np.sum(~pred & gt))
    return 2 * tp / denom if denom > 0 else 1.0


def get_pred_dir(model_name, suffix):
    return PRED / f"{model_name}_TTA{suffix}"


def evaluate_combo(model_names, gt_dir, suffix, thresholds, full_metrics=False):
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    n = len(gt_files)

    best_thr, best_dice = 0.5, 0.0
    for thr in thresholds:
        dices = []
        for gt_path in gt_files:
            stem = gt_path.name.replace(".nii.gz", "")
            probs = []
            for mn in model_names:
                d = get_pred_dir(mn, suffix)
                npz = d / f"{stem}.npz"
                if not npz.exists():
                    npz = d / f"{stem}.nii.gz.npz"
                probs.append(load_prob(npz))
            avg = np.mean(probs, axis=0)
            pred = (avg >= thr).astype(np.uint8)
            gt = (nib.load(str(gt_path)).get_fdata() > 0.5).astype(np.uint8)
            dices.append(fast_dice(pred, gt))
        md = np.mean(dices)
        if md > best_dice:
            best_dice = md
            best_thr = thr

    result = {"dice": best_dice, "threshold": best_thr}

    if full_metrics:
        all_m = {"dice": [], "nsd": [], "hd95": [], "sensitivity": [], "ppv": []}
        for gt_path in gt_files:
            stem = gt_path.name.replace(".nii.gz", "")
            probs = []
            for mn in model_names:
                d = get_pred_dir(mn, suffix)
                npz = d / f"{stem}.npz"
                if not npz.exists():
                    npz = d / f"{stem}.nii.gz.npz"
                probs.append(load_prob(npz))
            avg = np.mean(probs, axis=0)
            pred = (avg >= best_thr).astype(np.uint8)
            img = nib.load(str(gt_path))
            gt = (img.get_fdata() > 0.5).astype(np.uint8)
            pixdim = list(img.header.get_zooms()[:3])
            m = compute_metrics(pred, gt, pixdim)
            for k in all_m:
                all_m[k].append(m[k])
        result.update({k: np.mean(v) for k, v in all_m.items()})
        result["dice_std"] = np.std(all_m["dice"])

    return result


def main():
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

    print("=" * 70)
    print("  Validation EMA Dice ranking (all 15 models)")
    print("=" * 70)
    for i, (name, ema) in enumerate(RANKED):
        print(f"  {i+1:2d}. {name:<20s}  EMA={ema:.4f}")

    for ds_key, gt_dir, suffix in [("DS001", GT_DS001, ""), ("DS002", GT_DS002, "_DS002")]:
        print(f"\n{'='*70}")
        print(f"  {ds_key} — Validation-selected ensembles")
        print(f"{'='*70}")

        # Strategy 1: Top-K by global val EMA ranking
        print(f"\n  --- Strategy 1: Top-K models by validation EMA ---")
        for k in [3, 4, 5, 6, 7, 8, 10, 15]:
            selected = [name for name, _ in RANKED[:k]]
            r = evaluate_combo(selected, gt_dir, suffix, thresholds)
            arch_counts = {}
            for s in selected:
                arch = s.rsplit("_fold", 1)[0]
                arch_counts[arch] = arch_counts.get(arch, 0) + 1
            arch_str = ", ".join(f"{a}:{c}" for a, c in sorted(arch_counts.items()))
            print(f"  Top-{k:2d}: DSC={r['dice']:.4f} thr={r['threshold']:.2f}  ({arch_str})")

        # Strategy 2: Best N folds per architecture (1, 2, 3 from each)
        print(f"\n  --- Strategy 2: Best N folds per architecture ---")
        for n_per_arch in [1, 2, 3]:
            selected = []
            for arch in ["CNN_3D", "ResEncL_3D", "25D"]:
                arch_folds = [(name, ema) for name, ema in RANKED if name.startswith(arch)]
                arch_folds.sort(key=lambda x: -x[1])
                selected.extend([name for name, _ in arch_folds[:n_per_arch]])
            r = evaluate_combo(selected, gt_dir, suffix, thresholds)
            print(f"  Best-{n_per_arch}/arch ({len(selected)} models): DSC={r['dice']:.4f} thr={r['threshold']:.2f}  {selected}")

        # Strategy 3: All 3 architectures full 5-fold (= triple ensemble)
        all_15 = [name for name, _ in RANKED]
        r = evaluate_combo(all_15, gt_dir, suffix, thresholds)
        print(f"\n  All 15 models: DSC={r['dice']:.4f} thr={r['threshold']:.2f}")

        # Full metrics for top strategies
        print(f"\n  --- Full metrics for best strategies ---")

        # Top-K that performed best
        for label, models in [
            ("Top-3 global", [n for n, _ in RANKED[:3]]),
            ("Top-4 global", [n for n, _ in RANKED[:4]]),
            ("Top-5 global", [n for n, _ in RANKED[:5]]),
            ("Best-1/arch", []),  # will fill below
            ("Best-2/arch", []),
            ("All 3 arches 5-fold", all_15),
        ]:
            if "Best-1" in label:
                models = []
                for arch in ["CNN_3D", "ResEncL_3D", "25D"]:
                    arch_folds = [(n, e) for n, e in RANKED if n.startswith(arch)]
                    arch_folds.sort(key=lambda x: -x[1])
                    models.append(arch_folds[0][0])
            elif "Best-2" in label:
                models = []
                for arch in ["CNN_3D", "ResEncL_3D", "25D"]:
                    arch_folds = [(n, e) for n, e in RANKED if n.startswith(arch)]
                    arch_folds.sort(key=lambda x: -x[1])
                    models.extend([n for n, _ in arch_folds[:2]])

            r = evaluate_combo(models, gt_dir, suffix, thresholds, full_metrics=True)
            arch_str = ", ".join(models)
            print(f"\n  {label} ({len(models)} models) @ thr={r['threshold']:.2f}")
            print(f"    Models: {arch_str}")
            print(f"    DSC:  {r['dice']:.4f} +/- {r.get('dice_std', 0):.4f}")
            print(f"    NSD:  {r.get('nsd', 0):.4f}")
            print(f"    HD95: {r.get('hd95', 0):.4f}")
            print(f"    Sens: {r.get('sensitivity', 0):.4f}")
            print(f"    PPV:  {r.get('ppv', 0):.4f}")


if __name__ == "__main__":
    main()
