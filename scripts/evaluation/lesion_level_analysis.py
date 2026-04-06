#!/usr/bin/env python3
"""
lesion_level_analysis.py
========================
Per-lesion detection analysis based on lesion size.

For each subject:
  1. Extract individual connected-component lesions from GT
  2. Classify each by volume (tiny/small/medium/large)
  3. Determine if each GT lesion was detected (TP) or missed (FN)
  4. Find false-positive predicted lesions (FP)
  5. Report detection rate (lesion-wise sensitivity) by size bin

Size bins (standard in MS literature):
  - Tiny:   < 10 mm³   (< ~10 voxels at 1mm iso)
  - Small:  10–100 mm³  (~10–100 voxels)
  - Medium: 100–1000 mm³
  - Large:  > 1000 mm³  (> 1 mL)

A GT lesion is "detected" if the prediction overlaps ≥1 voxel of it.

Usage:
    python scripts/evaluation/lesion_level_analysis.py
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_BASE = Path(os.environ.get(
    "nnUNet_results", str(REPO_ROOT / "data" / "nnUNet_results")))
RAW_BASE = Path(os.environ.get(
    "nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw")))

DATASETS = {
    "DS001_ResEncL_3D": {
        "pred_dir": RESULTS_BASE / "predictions" / "DS001_ResEncL_3D",
        "gt_dir": RAW_BASE / "Dataset001_MSLesSeg" / "labelsTs",
        "name": "DS001 (MSLesSeg-2024)",
    },
    "DS002_ResEncL_3D": {
        "pred_dir": RESULTS_BASE / "predictions" / "DS002_ResEncL_3D",
        "gt_dir": RAW_BASE / "Dataset002_WMH" / "labelsTs",
        "name": "DS002 (WMH Challenge 2017)",
    },
}

# Volume bins in mm³
SIZE_BINS = [
    ("Tiny (<10 mm³)",    0,    10),
    ("Small (10-100 mm³)", 10,  100),
    ("Medium (100-1000 mm³)", 100, 1000),
    ("Large (>1000 mm³)",  1000, float("inf")),
]

# Overlap threshold for detection
DETECTION_OVERLAP_VOXELS = 1


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────

def get_bin(volume_mm3: float) -> str:
    """Return the size bin name for a given volume."""
    for name, lo, hi in SIZE_BINS:
        if lo <= volume_mm3 < hi:
            return name
    return SIZE_BINS[-1][0]


def analyze_lesions(
    gt: np.ndarray,
    pred: np.ndarray,
    pixdim: list[float],
    connectivity: int = 2,  # 26-connectivity for 3D
) -> dict:
    """
    Perform lesion-level detection analysis for one subject.

    Handles split/merge topology:
      - 1:1   — one GT CC matched to exactly one pred CC (and vice versa)
      - split — one GT CC overlaps multiple pred CCs (model fragmented the lesion)
      - merge — multiple GT CCs share a single pred CC (model joined lesions)
      - complex — GT CC is part of a many-to-many mapping (both split and merge)

    Per-lesion Dice uses the union of ALL pred CCs overlapping the GT CC,
    which correctly penalizes merges (inflated denominator) and captures
    fragments from splits.

    Returns dict with:
      gt_lesions: list of {id, volume_mm3, volume_vx, bin, detected, overlap_vx,
                           lesion_dice, match_type, n_pred_overlapping}
      fp_lesions: list of {id, volume_mm3, volume_vx, bin}
      tp, fn, fp counts
      match_counts: {one_to_one, split, merge, complex, fn} — topology summary
    """
    voxel_vol_mm3 = float(np.prod(pixdim))

    # Label GT connected components
    gt_labeled, n_gt = ndimage.label(gt, structure=ndimage.generate_binary_structure(3, connectivity))
    # Label pred connected components
    pred_labeled, n_pred = ndimage.label(pred, structure=ndimage.generate_binary_structure(3, connectivity))

    # ── Build bipartite overlap graph ──
    # gt_to_pred[i] = set of pred CC ids overlapping GT CC i (≥1 voxel)
    # pred_to_gt[j] = set of GT CC ids overlapping pred CC j (≥1 voxel)
    gt_to_pred = {i: set() for i in range(1, n_gt + 1)}
    pred_to_gt = {j: set() for j in range(1, n_pred + 1)}

    # Vectorized overlap matrix construction
    gt_flat = gt_labeled.ravel()
    pred_flat = pred_labeled.ravel()
    # Only look at voxels where both have labels
    overlap_mask = (gt_flat > 0) & (pred_flat > 0)
    gt_overlap_ids = gt_flat[overlap_mask]
    pred_overlap_ids = pred_flat[overlap_mask]
    for gi, pi in zip(gt_overlap_ids, pred_overlap_ids):
        gt_to_pred[gi].add(pi)
        pred_to_gt[pi].add(gi)

    # ── Classify each GT lesion ──
    gt_lesions = []
    for i in range(1, n_gt + 1):
        mask = gt_labeled == i
        vol_vx = int(np.sum(mask))
        vol_mm3 = vol_vx * voxel_vol_mm3
        pred_ids_overlapping = gt_to_pred[i]
        overlap_vx = int(np.sum(mask & (pred_labeled > 0)))
        detected = overlap_vx >= DETECTION_OVERLAP_VOXELS

        # Match type classification
        n_pred_overlap = len(pred_ids_overlapping)
        if not detected:
            match_type = "fn"
        elif n_pred_overlap == 1:
            pid = next(iter(pred_ids_overlapping))
            if len(pred_to_gt[pid]) == 1:
                match_type = "one_to_one"
            else:
                match_type = "merge"  # this pred CC also covers other GT CCs
        else:
            # GT overlaps multiple pred CCs — check if any of those also cover other GT CCs
            any_shared = any(len(pred_to_gt[pid]) > 1 for pid in pred_ids_overlapping)
            if any_shared:
                match_type = "complex"  # both split and merge
            else:
                match_type = "split"  # fragmented but each fragment is unique to this GT

        # Per-lesion Dice: union of all overlapping pred CCs
        if detected and pred_ids_overlapping:
            pred_mask = np.zeros_like(pred, dtype=bool)
            for pid in pred_ids_overlapping:
                pred_mask |= (pred_labeled == pid)
            intersection = int(np.sum(mask & pred_mask))
            pred_sum = int(np.sum(pred_mask))
            lesion_dice = 2 * intersection / (vol_vx + pred_sum) if (vol_vx + pred_sum) > 0 else 0.0
        else:
            lesion_dice = 0.0

        gt_lesions.append({
            "id": i,
            "volume_vx": vol_vx,
            "volume_mm3": vol_mm3,
            "bin": get_bin(vol_mm3),
            "detected": detected,
            "overlap_vx": overlap_vx,
            "lesion_dice": float(lesion_dice),
            "match_type": match_type,
            "n_pred_overlapping": n_pred_overlap,
        })

    # Find FP lesions (pred components with no GT overlap)
    fp_lesions = []
    if n_pred > 0:
        pred_sizes = np.zeros(n_pred + 1, dtype=np.int64)
        np.add.at(pred_sizes, pred_flat, 1)
        for j in range(1, n_pred + 1):
            if not pred_to_gt[j]:  # no GT overlap at all
                vol_vx = int(pred_sizes[j])
                fp_lesions.append({
                    "id": j,
                    "volume_vx": vol_vx,
                    "volume_mm3": vol_vx * voxel_vol_mm3,
                    "bin": get_bin(vol_vx * voxel_vol_mm3),
                })

    tp = sum(1 for l in gt_lesions if l["detected"])
    fn = sum(1 for l in gt_lesions if not l["detected"])
    fp = len(fp_lesions)

    # Match topology summary
    match_counts = {
        "one_to_one": sum(1 for l in gt_lesions if l["match_type"] == "one_to_one"),
        "split": sum(1 for l in gt_lesions if l["match_type"] == "split"),
        "merge": sum(1 for l in gt_lesions if l["match_type"] == "merge"),
        "complex": sum(1 for l in gt_lesions if l["match_type"] == "complex"),
        "fn": fn,
    }

    return {
        "gt_lesions": gt_lesions,
        "fp_lesions": fp_lesions,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "match_counts": match_counts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(exp_name: str, cfg: dict, all_subject_results: list[dict]):
    """Print formatted lesion-level detection summary by size bin."""

    # Aggregate across all subjects
    all_gt = []
    all_fp = []
    total_tp = total_fn = total_fp = 0
    for sr in all_subject_results:
        all_gt.extend(sr["gt_lesions"])
        all_fp.extend(sr["fp_lesions"])
        total_tp += sr["tp"]
        total_fn += sr["fn"]
        total_fp += sr["fp"]

    n_subjects = len(all_subject_results)
    n_gt_total = len(all_gt)

    print(f"\n{'=' * 80}")
    print(f"  LESION-LEVEL DETECTION ANALYSIS: {cfg['name']}")
    print(f"  {n_subjects} subjects, {n_gt_total} GT lesions, "
          f"{total_tp} TP, {total_fn} FN, {total_fp} FP")
    print(f"{'=' * 80}")

    # Overall lesion-wise metrics
    lesion_sens = total_tp / max(1, total_tp + total_fn)
    lesion_ppv = total_tp / max(1, total_tp + total_fp)
    lesion_f1 = 2 * lesion_sens * lesion_ppv / max(1e-9, lesion_sens + lesion_ppv)
    print(f"\n  Overall lesion-wise:")
    print(f"    Sensitivity (detection rate): {lesion_sens:.4f} ({total_tp}/{total_tp+total_fn})")
    print(f"    PPV (precision):              {lesion_ppv:.4f} ({total_tp}/{total_tp+total_fp})")
    print(f"    F1:                           {lesion_f1:.4f}")

    # By size bin
    print(f"\n  {'Size Bin':<25} {'#GT':>5} {'#TP':>5} {'#FN':>5} {'DetRate':>8} "
          f"{'MeanDice':>9} {'#FP':>5} {'MedVol':>10}")
    print(f"  {'-'*75}")

    for bin_name, _, _ in SIZE_BINS:
        gt_in_bin = [l for l in all_gt if l["bin"] == bin_name]
        tp_in_bin = sum(1 for l in gt_in_bin if l["detected"])
        fn_in_bin = sum(1 for l in gt_in_bin if not l["detected"])
        det_rate = tp_in_bin / max(1, len(gt_in_bin))

        dices = [l["lesion_dice"] for l in gt_in_bin if l["detected"]]
        mean_dice = np.mean(dices) if dices else 0.0

        fp_in_bin = [l for l in all_fp if l["bin"] == bin_name]

        vols = [l["volume_mm3"] for l in gt_in_bin]
        med_vol = np.median(vols) if vols else 0.0

        print(f"  {bin_name:<25} {len(gt_in_bin):>5} {tp_in_bin:>5} {fn_in_bin:>5} "
              f"{det_rate:>7.1%} {mean_dice:>9.4f} {len(fp_in_bin):>5} {med_vol:>8.1f} mm³")

    # Missed lesion analysis
    missed = [l for l in all_gt if not l["detected"]]
    if missed:
        missed_vols = [l["volume_mm3"] for l in missed]
        print(f"\n  Missed lesions (FN={len(missed)}):")
        print(f"    Volume range: {min(missed_vols):.1f} – {max(missed_vols):.1f} mm³")
        print(f"    Volume mean:  {np.mean(missed_vols):.1f} mm³ | "
              f"median: {np.median(missed_vols):.1f} mm³")
        # How many missed lesions are tiny?
        for bin_name, _, _ in SIZE_BINS:
            n = sum(1 for l in missed if l["bin"] == bin_name)
            if n > 0:
                print(f"    {bin_name}: {n} missed ({n/len(missed)*100:.1f}%)")

    # FP analysis
    if all_fp:
        fp_vols = [l["volume_mm3"] for l in all_fp]
        print(f"\n  False positive lesions (FP={len(all_fp)}):")
        print(f"    Volume range: {min(fp_vols):.1f} – {max(fp_vols):.1f} mm³")
        print(f"    Volume mean:  {np.mean(fp_vols):.1f} mm³ | "
              f"median: {np.median(fp_vols):.1f} mm³")
        for bin_name, _, _ in SIZE_BINS:
            n = sum(1 for l in all_fp if l["bin"] == bin_name)
            if n > 0:
                print(f"    {bin_name}: {n} FP ({n/len(all_fp)*100:.1f}%)")

    return {
        "total_gt": n_gt_total, "total_tp": total_tp, "total_fn": total_fn,
        "total_fp": total_fp, "lesion_sens": lesion_sens, "lesion_ppv": lesion_ppv,
        "lesion_f1": lesion_f1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(datasets: dict = None, tag: str = "") -> dict:
    """Run lesion-level analysis on specified datasets. Returns results dict."""
    if datasets is None:
        datasets = DATASETS

    print("=" * 80)
    print(f"  Lesion-Level Detection Analysis by Size{f' [{tag}]' if tag else ''}")
    print("=" * 80)

    t0 = time.time()
    all_results = {}

    for exp_name, cfg in datasets.items():
        pred_dir = cfg["pred_dir"]
        gt_dir = cfg["gt_dir"]

        if not pred_dir.is_dir() or not gt_dir.is_dir():
            print(f"  [SKIP] Missing dirs for {exp_name}")
            continue

        pred_files = sorted(pred_dir.glob("*.nii.gz"))
        gt_map = {f.name: f for f in gt_dir.glob("*.nii.gz")}

        subject_results = []
        for i, pf in enumerate(pred_files):
            if pf.name not in gt_map:
                continue
            pred_nii = nib.load(str(pf))
            gt_nii = nib.load(str(gt_map[pf.name]))
            pred_data = (np.asarray(pred_nii.dataobj) > 0).astype(np.uint8)
            gt_data = (np.asarray(gt_nii.dataobj) > 0).astype(np.uint8)
            pixdim = [float(v) for v in pred_nii.header.get_zooms()[:3]]

            result = analyze_lesions(gt_data, pred_data, pixdim)
            result["subject"] = pf.name
            subject_results.append(result)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(pred_files)}] processed...")

        summary = print_summary(exp_name, cfg, subject_results)
        all_results[exp_name] = {
            "subject_results": subject_results,
            "summary": summary,
        }

    elapsed = time.time() - t0
    print(f"\nTotal analysis time: {elapsed:.1f}s")

    # Save per-lesion CSV
    out_dir = RESULTS_BASE / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"lesion_level_analysis{f'_{tag}' if tag else ''}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "experiment", "subject", "lesion_type", "lesion_id",
            "volume_vx", "volume_mm3", "bin", "detected", "overlap_vx", "lesion_dice",
        ])
        writer.writeheader()
        for exp_name, data in all_results.items():
            for sr in data["subject_results"]:
                for l in sr["gt_lesions"]:
                    writer.writerow({
                        "experiment": exp_name, "subject": sr["subject"],
                        "lesion_type": "GT", "lesion_id": l["id"],
                        "volume_vx": l["volume_vx"], "volume_mm3": f"{l['volume_mm3']:.2f}",
                        "bin": l["bin"], "detected": l["detected"],
                        "overlap_vx": l["overlap_vx"], "lesion_dice": f"{l['lesion_dice']:.4f}",
                    })
                for l in sr["fp_lesions"]:
                    writer.writerow({
                        "experiment": exp_name, "subject": sr["subject"],
                        "lesion_type": "FP", "lesion_id": l["id"],
                        "volume_vx": l["volume_vx"], "volume_mm3": f"{l['volume_mm3']:.2f}",
                        "bin": l["bin"], "detected": "", "overlap_vx": "", "lesion_dice": "",
                    })
    print(f"  Per-lesion CSV saved to: {csv_path}")

    return all_results


if __name__ == "__main__":
    run_analysis()
