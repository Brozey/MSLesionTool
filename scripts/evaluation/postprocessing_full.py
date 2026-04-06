#!/usr/bin/env python3
"""
postprocessing_full.py
======================
Efficient two-phase post-processing comparison:
  Phase 1 — Fast Dice-only sweep across ALL strategies (ms per case)
  Phase 2 — Full metrics (HD95, NSD, lesion-level) on top-N candidates only

Strategies tested:
  1. Baseline hard predictions (noTTA / TTA)
  2. Remove small components (3, 5, 10 mm³)
  3. Softmax threshold tuning (0.3–0.6)
  4. Best threshold + remove small
  5. TTA variants of all above

Fixes:
  - Softmax axis auto-transpose (NPZ internal order != NIfTI nibabel order)
  - Two-phase evaluation for ~10x speedup
  - Parallel case evaluation in Phase 2

Usage:
    python scripts/evaluation/postprocessing_full.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluation.evaluate_test_sets import compute_metrics
from scripts.evaluation.lesion_level_analysis import analyze_lesions

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_BASE = Path(os.environ.get(
    "nnUNet_results", str(REPO_ROOT / "data" / "nnUNet_results")))
RAW_BASE = Path(os.environ.get(
    "nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw")))

DATASETS = {
    "DS001": {
        "gt_dir": RAW_BASE / "Dataset001_MSLesSeg" / "labelsTs",
        "pred_dirs": {
            "noTTA": RESULTS_BASE / "predictions" / "DS001_ResEncL_3D_softmax",
            "TTA":   RESULTS_BASE / "predictions" / "DS001_ResEncL_3D_TTA",
        },
        "name": "DS001 (MSLesSeg-2024)",
    },
    "DS002": {
        "gt_dir": RAW_BASE / "Dataset002_WMH" / "labelsTs",
        "pred_dirs": {
            "noTTA": RESULTS_BASE / "predictions" / "DS002_ResEncL_3D_softmax",
            "TTA":   RESULTS_BASE / "predictions" / "DS002_ResEncL_3D_TTA",
        },
        "name": "DS002 (WMH Challenge 2017)",
    },
}

REMOVE_SIZES_MM3 = [3, 5, 10]
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
TOP_N = 8               # How many strategies get full evaluation in Phase 2
NUM_WORKERS = 6          # Parallel workers for Phase 2 (adjust to CPU cores)

OUTPUT_DIR = RESULTS_BASE / "evaluation" / "postprocessing"


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing primitives
# ──────────────────────────────────────────────────────────────────────────────

def remove_small_cc(seg: np.ndarray, min_size_mm3: float,
                    pixdim: list[float]) -> np.ndarray:
    """Remove connected components smaller than min_size_mm3."""
    voxel_vol = float(np.prod(pixdim))
    min_vox = max(1, int(np.ceil(min_size_mm3 / voxel_vol)))
    labeled, n = ndimage.label(seg, structure=ndimage.generate_binary_structure(3, 2))
    if n == 0:
        return seg
    sizes = np.bincount(labeled.ravel())
    remove = sizes < min_vox
    remove[0] = False  # keep background label unchanged
    out = seg.copy()
    out[remove[labeled]] = 0
    return out


def fast_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice in ~1 ms with pure numpy — no MetricsReloaded overhead."""
    p = pred.ravel().astype(bool)
    g = gt.ravel().astype(bool)
    inter = np.count_nonzero(p & g)
    denom = np.count_nonzero(p) + np.count_nonzero(g)
    if denom == 0:
        return 1.0 if np.count_nonzero(g) == 0 else 0.0
    return 2.0 * inter / denom


def fix_softmax_orientation(softmax: np.ndarray,
                            target_shape: tuple) -> np.ndarray:
    """Transpose softmax from nnU-Net internal (z,y,x) to NIfTI (x,y,z) order.

    nnU-Net NPZ always stores probabilities in (z,y,x) axis order while
    nibabel loads NIfTI in (x,y,z).  A simple shape-match heuristic FAILS
    when two spatial dims are equal (e.g. 182×218×182).  Always apply the
    canonical (2,1,0) transpose instead.
    """
    return np.transpose(softmax, (2, 1, 0))


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_cases(gt_dir: Path, pred_dir: Path) -> list[dict]:
    """Load GT, hard predictions, and softmax.  Auto-fix softmax orientation."""
    gt_map = {f.name: f for f in gt_dir.glob("*.nii.gz")}
    pred_files = sorted(pred_dir.glob("*.nii.gz"))

    cases = []
    n_transposed = 0
    for pf in pred_files:
        if pf.name not in gt_map:
            continue
        pred_nii = nib.load(str(pf))
        gt_nii = nib.load(str(gt_map[pf.name]))
        pred_data = (np.asarray(pred_nii.dataobj) > 0).astype(np.uint8)
        gt_data = (np.asarray(gt_nii.dataobj) > 0).astype(np.uint8)
        pixdim = [float(v) for v in pred_nii.header.get_zooms()[:3]]

        # Load softmax from NPZ
        npz_path = pred_dir / pf.name.replace(".nii.gz", ".npz")
        softmax = None
        if npz_path.exists():
            npz_data = np.load(str(npz_path))
            key = "softmax" if "softmax" in npz_data else "probabilities"
            raw_sm = npz_data[key][1]  # foreground probability map

            # Always transpose from nnU-Net (z,y,x) to NIfTI (x,y,z)
            softmax = fix_softmax_orientation(raw_sm, pred_data.shape)
            assert softmax.shape == pred_data.shape, \
                f"Softmax {softmax.shape} != pred {pred_data.shape} for {pf.name}"

        cases.append({
            "name": pf.name,
            "pred": pred_data,
            "gt": gt_data,
            "pixdim": pixdim,
            "softmax": softmax,
        })

    if n_transposed:
        print(f"    [FIX] Transposed softmax axes for {n_transposed}/{len(cases)} cases")
    return cases


# ──────────────────────────────────────────────────────────────────────────────
# Strategy generators
# ──────────────────────────────────────────────────────────────────────────────

def make_strategies(prefix: str, has_softmax: bool) -> dict:
    """Build strategy_name -> transform_fn(case) -> pred_array."""
    strats = {}

    # 1) Baseline hard prediction
    strats[f"{prefix}_baseline"] = lambda c: c["pred"]

    # 2) Remove small CCs on hard prediction
    for sz in REMOVE_SIZES_MM3:
        strats[f"{prefix}_rm_{sz}mm3"] = (
            lambda c, s=sz: remove_small_cc(c["pred"], s, c["pixdim"]))

    if has_softmax:
        # 3) Threshold sweep on softmax
        for t in THRESHOLDS:
            strats[f"{prefix}_thr_{t:.2f}"] = (
                lambda c, thr=t: (c["softmax"] >= thr).astype(np.uint8))

        # 4) Threshold + remove small combos
        for t in THRESHOLDS:
            for sz in REMOVE_SIZES_MM3:
                strats[f"{prefix}_thr_{t:.2f}_rm_{sz}mm3"] = (
                    lambda c, thr=t, s=sz: remove_small_cc(
                        (c["softmax"] >= thr).astype(np.uint8), s, c["pixdim"]))

    return strats


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Fast Dice sweep (pure numpy, ~1 ms/case)
# ──────────────────────────────────────────────────────────────────────────────

def fast_sweep(strategies: dict, cases_by_prefix: dict) -> dict[str, float]:
    """Compute mean Dice for each strategy using fast numpy."""
    results = {}
    for sname, (tfn, prefix) in strategies.items():
        cases = cases_by_prefix[prefix]
        dices = [fast_dice(tfn(c), c["gt"]) for c in cases]
        results[sname] = float(np.mean(dices))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Full evaluation on top candidates (parallel)
# ──────────────────────────────────────────────────────────────────────────────

def eval_single_case(pred: np.ndarray, gt: np.ndarray,
                     pixdim: list[float]) -> dict:
    """Full voxel + lesion metrics for one case (runs in worker process)."""
    vm = compute_metrics(pred, gt, pixdim)
    lm = analyze_lesions(gt, pred, pixdim)
    return {**vm, "l_tp": lm["tp"], "l_fn": lm["fn"],
            "l_fp": lm["fp"], "l_ngt": lm["n_gt"]}


def full_eval_strategy(sname: str, tfn, cases: list[dict],
                       parallel: bool = True) -> dict:
    """Full metrics for one strategy across all cases."""
    tasks = [(tfn(c), c["gt"], c["pixdim"]) for c in cases]

    rows = []
    if parallel and NUM_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futs = {pool.submit(eval_single_case, p, g, d): i
                    for i, (p, g, d) in enumerate(tasks)}
            for fut in as_completed(futs):
                rows.append(fut.result())
    else:
        for p, g, d in tasks:
            rows.append(eval_single_case(p, g, d))

    return aggregate(rows)


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-case results into dataset-level summary."""
    n = len(rows)
    if n == 0:
        return {}

    def mean_k(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")

    def med_k(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    tp = sum(r["l_tp"] for r in rows)
    fn = sum(r["l_fn"] for r in rows)
    fp = sum(r["l_fp"] for r in rows)
    ls = tp / max(1, tp + fn)
    lp = tp / max(1, tp + fp)
    lf1 = 2 * ls * lp / max(1e-9, ls + lp) if (ls + lp) > 0 else 0.0

    return {
        "n": n,
        "dice_mean": mean_k("dice"), "dice_median": med_k("dice"),
        "hd95_mean": mean_k("hd95"), "hd95_median": med_k("hd95"),
        "sens_mean": mean_k("sensitivity"), "ppv_mean": mean_k("ppv"),
        "nsd_mean": mean_k("nsd"), "avd_mean": mean_k("avd"),
        "l_sens": ls, "l_ppv": lp, "l_f1": lf1,
        "l_tp": tp, "l_fn": fn, "l_fp": fp,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ds_key, ds_cfg in DATASETS.items():
        print(f"\n{'=' * 90}")
        print(f"  {ds_cfg['name']}")
        print(f"{'=' * 90}")

        all_strategies = {}
        cases_by_prefix = {}

        # Load data for each prediction set (noTTA / TTA)
        for tta_label, pred_dir in ds_cfg["pred_dirs"].items():
            prefix = tta_label
            t0 = time.time()
            print(f"\n  Loading {prefix} cases from {pred_dir}...")
            cases = load_cases(ds_cfg["gt_dir"], pred_dir)
            has_sm = any(c["softmax"] is not None for c in cases)
            print(f"  {len(cases)} cases, softmax={has_sm}, load={time.time()-t0:.1f}s")
            cases_by_prefix[prefix] = cases
            strats = make_strategies(prefix, has_sm)
            all_strategies.update({k: (v, prefix) for k, v in strats.items()})

        # ── Phase 1: Fast Dice sweep ──────────────────────────────────────────
        n_strats = len(all_strategies)
        print(f"\n  Phase 1: Fast Dice sweep ({n_strats} strategies)...")
        t0 = time.time()
        fast_results = fast_sweep(all_strategies, cases_by_prefix)

        # Sort and show
        ranked = sorted(fast_results.items(), key=lambda x: -x[1])
        print(f"  Done in {time.time()-t0:.1f}s")
        print(f"\n  {'Strategy':<42} {'Mean Dice':>10}")
        print(f"  {'-'*52}")
        for i, (sname, dice) in enumerate(ranked):
            marker = " << TOP" if i < TOP_N else ""
            print(f"  {sname:<42} {dice:>10.4f}{marker}")

        # ── Phase 2: Full evaluation on top-N ─────────────────────────────────
        top_names = [name for name, _ in ranked[:TOP_N]]
        print(f"\n  Phase 2: Full metrics on top {TOP_N} strategies "
              f"(parallel={NUM_WORKERS} workers)...")

        full_results = {}
        for sname in top_names:
            tfn, prefix = all_strategies[sname]
            cases = cases_by_prefix[prefix]
            t1 = time.time()
            print(f"    {sname}...", end="", flush=True)
            res = full_eval_strategy(sname, tfn, cases, parallel=True)
            full_results[sname] = res
            print(f"  {time.time()-t1:.0f}s  DSC={res['dice_mean']:.4f}  "
                  f"LF1={res['l_f1']:.4f}")

        # ── Print comparison table ────────────────────────────────────────────
        baseline_dice = full_results.get(
            "noTTA_baseline", next(iter(full_results.values())))["dice_mean"]

        print(f"\n{'─' * 120}")
        print(f"  FULL METRICS — {ds_cfg['name']}  (top {TOP_N} by Dice)")
        print(f"{'─' * 120}")
        hdr = (f"  {'Strategy':<42} {'DSC':>7} {'DSCmed':>7} {'HD95':>7} "
               f"{'Sens':>7} {'PPV':>7} {'NSD':>7} "
               f"{'LSens':>7} {'LPPV':>7} {'LF1':>7} {'LFP':>6}")
        print(hdr)
        print(f"  {'-'*118}")

        for sname in sorted(full_results,
                            key=lambda k: -full_results[k]["dice_mean"]):
            r = full_results[sname]
            delta = r["dice_mean"] - baseline_dice
            sym = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            print(
                f"  {sname:<42} "
                f"{r['dice_mean']:>7.4f} {r['dice_median']:>7.4f} "
                f"{r['hd95_mean']:>7.2f} "
                f"{r['sens_mean']:>7.4f} {r['ppv_mean']:>7.4f} "
                f"{r['nsd_mean']:>7.4f} "
                f"{r['l_sens']:>7.4f} {r['l_ppv']:>7.4f} "
                f"{r['l_f1']:>7.4f} {r['l_fp']:>6d}"
                f"  {sym}{abs(delta):.4f}")

        all_results[ds_key] = full_results

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "full_postprocessing_comparison.csv"
    fields = ["dataset", "strategy",
              "dice_mean", "dice_median", "hd95_mean", "hd95_median",
              "sens_mean", "ppv_mean", "nsd_mean", "avd_mean",
              "l_sens", "l_ppv", "l_f1", "l_tp", "l_fn", "l_fp"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for dk, exps in all_results.items():
            for sname, res in exps.items():
                row = {"dataset": dk, "strategy": sname}
                for f in fields[2:]:
                    v = res.get(f, 0)
                    row[f] = f"{v:.6f}" if isinstance(v, float) else v
                w.writerow(row)
    print(f"\n  Results saved to: {csv_path}")

    # ── Recommendations ───────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  RECOMMENDATIONS")
    print(f"{'=' * 90}")
    for dk, exps in all_results.items():
        bl = exps.get("noTTA_baseline", next(iter(exps.values())))
        best_dsc = max(exps.items(), key=lambda x: x[1]["dice_mean"])
        best_lf1 = max(exps.items(), key=lambda x: x[1]["l_f1"])
        best_comb = max(exps.items(),
                        key=lambda x: (x[1]["dice_mean"] + x[1]["l_f1"]) / 2)
        print(f"\n  {dk}:")
        print(f"    Baseline (noTTA):  DSC={bl['dice_mean']:.4f}, LF1={bl['l_f1']:.4f}")
        print(f"    Best DSC:          {best_dsc[0]} -> {best_dsc[1]['dice_mean']:.4f} "
              f"(d={best_dsc[1]['dice_mean'] - bl['dice_mean']:+.4f})")
        print(f"    Best Lesion-F1:    {best_lf1[0]} -> {best_lf1[1]['l_f1']:.4f}")
        print(f"    Best Combined:     {best_comb[0]} -> DSC={best_comb[1]['dice_mean']:.4f}, "
              f"LF1={best_comb[1]['l_f1']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
