#!/usr/bin/env python3
"""Evaluate all 7 architecture-level foldall ensemble combinations."""
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
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]


def load_prob(npz_path):
    d = np.load(npz_path)
    key = "softmax" if "softmax" in d.files else "probabilities"
    arr = d[key][1].astype(np.float32)
    return np.transpose(arr, (2, 1, 0))


def fast_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 1.0


def run_dataset(ds_key, gt_dir, suffix):
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    n = len(gt_files)
    print(f"\n{'='*60}")
    print(f"  {ds_key} ({n} cases) — Architecture-level foldall combos")
    print(f"{'='*60}")

    # Load architecture-level softmax (average of 5 per-fold softmax)
    arch_probs = {}
    for arch_name, fold_dirs in [
        ("CNN_3D", [f"CNN_3D_fold{i}_TTA{suffix}" for i in range(5)]),
        ("ResEncL_3D", [f"ResEncL_3D_fold{i}_TTA{suffix}" for i in range(5)]),
        ("25D", [f"25D_fold{i}_TTA{suffix}" for i in range(5)]),
    ]:
        all_probs = []
        for gt_path in gt_files:
            stem = gt_path.name.replace(".nii.gz", "")
            fold_probs = []
            for fd in fold_dirs:
                npz = PRED / fd / f"{stem}.npz"
                if not npz.exists():
                    npz = PRED / fd / f"{stem}.nii.gz.npz"
                fold_probs.append(load_prob(npz))
            avg = np.mean(fold_probs, axis=0)
            all_probs.append(avg)
        arch_probs[arch_name] = all_probs
        print(f"  Loaded {arch_name}: {len(all_probs)} cases (avg of 5 folds)")

    # Load GT masks
    gt_masks = []
    gt_pixdims = []
    for gt_path in gt_files:
        img = nib.load(str(gt_path))
        gt_masks.append((img.get_fdata() > 0.5).astype(np.uint8))
        gt_pixdims.append(list(img.header.get_zooms()[:3]))

    # Enumerate all 7 combos × 5 thresholds (fast dice)
    arch_names = list(arch_probs.keys())
    results = []
    for r in range(1, len(arch_names) + 1):
        for combo in combinations(range(len(arch_names)), r):
            combo_name = " + ".join(arch_names[i] for i in combo)
            for thr in THRESHOLDS:
                dices = []
                for ci in range(n):
                    avg_prob = np.mean(
                        [arch_probs[arch_names[i]][ci] for i in combo], axis=0
                    )
                    pred = (avg_prob >= thr).astype(np.uint8)
                    dices.append(fast_dice(pred, gt_masks[ci]))
                mean_dice = np.mean(dices)
                results.append((combo_name, thr, mean_dice, len(combo)))

    # Sort and print fast metrics
    results.sort(key=lambda x: -x[2])
    print(f"\n  {'Combination':<45} {'Thr':>5}   {'DSC':>7}")
    print(f"  {'-'*45} {'-'*5}   {'-'*7}")
    for i, (name, thr, dice, nc) in enumerate(results[:21]):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {name:<45} {thr:.2f}   {dice:.4f}{marker}")

    # Full metrics for top-3
    print(f"\n  Full metrics for top-3:")
    seen = set()
    count = 0
    for name, thr, dice, nc in results:
        if count >= 3:
            break
        key = (name, thr)
        if key in seen:
            continue
        seen.add(key)
        count += 1

        combo_idx = [i for i, an in enumerate(arch_names) if an in name.split(" + ")]
        full_dices, full_nsd, full_hd95, full_sens, full_ppv = [], [], [], [], []
        for ci in range(n):
            avg_prob = np.mean(
                [arch_probs[arch_names[i]][ci] for i in combo_idx], axis=0
            )
            pred = (avg_prob >= thr).astype(np.uint8)
            m = compute_metrics(pred, gt_masks[ci], gt_pixdims[ci])
            full_dices.append(m["dice"])
            full_nsd.append(m["nsd"])
            full_hd95.append(m["hd95"])
            full_sens.append(m["sensitivity"])
            full_ppv.append(m["ppv"])

        print(f"\n  #{count}: {name} @ thr={thr:.2f}")
        print(f"    DSC:  {np.mean(full_dices):.4f} +/- {np.std(full_dices):.4f}")
        print(f"    NSD:  {np.mean(full_nsd):.4f}")
        print(f"    HD95: {np.mean(full_hd95):.4f}")
        print(f"    Sens: {np.mean(full_sens):.4f}")
        print(f"    PPV:  {np.mean(full_ppv):.4f}")


if __name__ == "__main__":
    run_dataset("DS001", GT_DS001, "")
    run_dataset("DS002", GT_DS002, "_DS002")
