#!/usr/bin/env python3
"""
postprocessing_experiments.py
=============================
Systematically test post-processing strategies on DS001/DS002 predictions.

Strategies tested:
  1. baseline       — Raw network output (no post-processing)
  2. remove_tiny_N  — Remove connected components < N mm³
  3. remove_tiny_N_fill_M  — Remove tiny CCs + fill small holes < M mm³
  4. threshold_T    — Re-threshold softmax at T instead of 0.5

For threshold experiments, we need the softmax/probability maps from nnUNet.
If npz files are not available, threshold experiments are skipped.

Each strategy is evaluated with both:
  - Voxel-level metrics (DSC, HD95, Sens, PPV, NSD)
  - Lesion-level metrics (detection rate, lesion-wise F1)

Usage:
    python scripts/evaluation/postprocessing_experiments.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

# Add parent so we can import our modules
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_test_sets import compute_metrics, METRIC_KEYS
from scripts.evaluation.lesion_level_analysis import analyze_lesions, SIZE_BINS, get_bin

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_BASE = REPO_ROOT / "results"
RAW_BASE = Path(os.environ.get(
    "nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw")))

DATASETS = {
    "DS001": {
        "pred_dir": RESULTS_BASE / "predictions" / "DS001_ResEncL_3D",
        "gt_dir": RAW_BASE / "Dataset001_MSLesSeg" / "labelsTs",
        "name": "DS001 (MSLesSeg-2024)",
    },
    "DS002": {
        "pred_dir": RESULTS_BASE / "predictions" / "DS002_ResEncL_3D",
        "gt_dir": RAW_BASE / "Dataset002_WMH" / "labelsTs",
        "name": "DS002 (WMH Challenge 2017)",
    },
}

# Post-processing parameter grid
# Remove connected components smaller than N mm³
REMOVE_SIZES_MM3 = [3, 5, 10, 15, 20, 30, 50]

# Fill holes smaller than M mm³ (binary morphological closing)
FILL_SIZES_MM3 = [5, 10, 20]

# Threshold values (only if softmax .npz available)
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

OUTPUT_DIR = RESULTS_BASE / "evaluation" / "postprocessing"


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing functions
# ──────────────────────────────────────────────────────────────────────────────

def remove_small_components(
    seg: np.ndarray,
    min_size_mm3: float,
    pixdim: list[float],
    connectivity: int = 2,
) -> np.ndarray:
    """Remove connected components smaller than min_size_mm3."""
    voxel_vol = float(np.prod(pixdim))
    min_voxels = max(1, int(np.ceil(min_size_mm3 / voxel_vol)))

    labeled, n = ndimage.label(seg, structure=ndimage.generate_binary_structure(3, connectivity))
    if n == 0:
        return seg

    # Count sizes using bincount
    sizes = np.bincount(labeled.ravel())
    # Build mask of labels to REMOVE (small foreground components)
    remove = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        if sizes[i] < min_voxels:
            remove[i] = True

    # Zero out small components
    result = seg.copy()
    result[remove[labeled]] = 0
    return result


def fill_small_holes(
    seg: np.ndarray,
    max_hole_mm3: float,
    pixdim: list[float],
    connectivity: int = 2,
) -> np.ndarray:
    """Fill small holes (connected components of background enclosed by foreground).

    Only fills background components that do NOT touch the volume border
    (i.e., true internal holes).
    """
    voxel_vol = float(np.prod(pixdim))
    max_voxels = max(1, int(np.ceil(max_hole_mm3 / voxel_vol)))

    # Invert: holes are foreground in inverted
    inv = (1 - seg).astype(np.uint8)
    labeled, n = ndimage.label(inv, structure=ndimage.generate_binary_structure(3, connectivity))
    if n == 0:
        return seg

    # Find which labels touch the border (these are outer background, not holes)
    border_labels = set()
    for ax in range(3):
        for side in [0, seg.shape[ax] - 1]:
            slc = [slice(None)] * 3
            slc[ax] = side
            border_labels.update(np.unique(labeled[tuple(slc)]))
    border_labels.discard(0)

    sizes = np.bincount(labeled.ravel())
    result = seg.copy()
    for i in range(1, n + 1):
        # Only fill internal holes (not touching border) that are small enough
        if i not in border_labels and sizes[i] <= max_voxels:
            result[labeled == i] = 1  # fill the hole

    return result


def apply_threshold(softmax: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold to softmax probability map."""
    return (softmax >= threshold).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single(pred: np.ndarray, gt: np.ndarray, pixdim: list[float]) -> dict:
    """Compute both voxel-level and lesion-level metrics for one case."""
    voxel_metrics = compute_metrics(pred, gt, pixdim)
    lesion_results = analyze_lesions(gt, pred, pixdim)
    return {
        **voxel_metrics,
        "lesion_tp": lesion_results["tp"],
        "lesion_fn": lesion_results["fn"],
        "lesion_fp": lesion_results["fp"],
        "lesion_n_gt": lesion_results["n_gt"],
    }


def aggregate_results(rows: list[dict]) -> dict:
    """Aggregate per-subject results into summary stats."""
    n = len(rows)
    if n == 0:
        return {}

    agg = {}
    for key in ["dice", "hd95", "sensitivity", "ppv", "nsd", "avd"]:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        agg[f"{key}_mean"] = np.mean(vals) if vals else float("nan")
        agg[f"{key}_median"] = np.median(vals) if vals else float("nan")

    total_tp = sum(r["lesion_tp"] for r in rows)
    total_fn = sum(r["lesion_fn"] for r in rows)
    total_fp = sum(r["lesion_fp"] for r in rows)
    total_gt = sum(r["lesion_n_gt"] for r in rows)

    agg["lesion_sensitivity"] = total_tp / max(1, total_tp + total_fn)
    agg["lesion_ppv"] = total_tp / max(1, total_tp + total_fp)
    ls = agg["lesion_sensitivity"]
    lp = agg["lesion_ppv"]
    agg["lesion_f1"] = 2 * ls * lp / max(1e-9, ls + lp)
    agg["lesion_tp"] = total_tp
    agg["lesion_fn"] = total_fn
    agg["lesion_fp"] = total_fp
    agg["lesion_n_gt"] = total_gt
    agg["n_subjects"] = n

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def load_cases(cfg: dict) -> list[dict]:
    """Load all prediction/GT pairs for a dataset."""
    pred_dir = cfg["pred_dir"]
    gt_dir = cfg["gt_dir"]
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    gt_map = {f.name: f for f in gt_dir.glob("*.nii.gz")}

    cases = []
    for pf in pred_files:
        if pf.name not in gt_map:
            continue
        pred_nii = nib.load(str(pf))
        gt_nii = nib.load(str(gt_map[pf.name]))
        pred_data = (np.asarray(pred_nii.dataobj) > 0).astype(np.uint8)
        gt_data = (np.asarray(gt_nii.dataobj) > 0).astype(np.uint8)
        pixdim = [float(v) for v in pred_nii.header.get_zooms()[:3]]

        # Check for softmax .npz
        npz_path = pf.with_suffix("").with_suffix(".npz")  # remove .nii.gz, add .npz
        softmax = None
        if npz_path.exists():
            npz = np.load(str(npz_path))
            if "softmax" in npz:
                softmax = npz["softmax"][1]  # class 1 (foreground) probability
            elif "probabilities" in npz:
                softmax = npz["probabilities"][1]

        cases.append({
            "name": pf.name,
            "pred": pred_data,
            "gt": gt_data,
            "pixdim": pixdim,
            "softmax": softmax,
        })

    return cases


def run_experiment(
    strategy_name: str,
    cases: list[dict],
    postprocess_fn,
    save_dir: Path = None,
) -> dict:
    """Run a single post-processing experiment on all cases.

    postprocess_fn: callable(case_dict) -> np.ndarray (processed prediction)
    """
    rows = []
    for case in cases:
        processed = postprocess_fn(case)
        metrics = evaluate_single(processed, case["gt"], case["pixdim"])
        metrics["subject"] = case["name"]
        rows.append(metrics)

        # Optionally save processed predictions
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            # Load the original NIfTI to get the header/affine
            pred_path = None
            for ds_cfg in DATASETS.values():
                p = ds_cfg["pred_dir"] / case["name"]
                if p.exists():
                    pred_path = p
                    break
            if pred_path:
                ref_nii = nib.load(str(pred_path))
                out_nii = nib.Nifti1Image(processed, ref_nii.affine, ref_nii.header)
                nib.save(out_nii, str(save_dir / case["name"]))

    return aggregate_results(rows)


def run_all_experiments():
    """Run all post-processing experiments for all datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ds_key, ds_cfg in DATASETS.items():
        print(f"\n{'=' * 80}")
        print(f"  POST-PROCESSING EXPERIMENTS: {ds_cfg['name']}")
        print(f"{'=' * 80}")

        t0 = time.time()
        print("  Loading cases...")
        cases = load_cases(ds_cfg)
        print(f"  Loaded {len(cases)} cases in {time.time()-t0:.1f}s")
        has_softmax = any(c["softmax"] is not None for c in cases)
        print(f"  Softmax maps available: {has_softmax}")

        experiments = {}

        # ── 1. Baseline ──
        print("\n  [1] Baseline (no post-processing)...")
        result = run_experiment(
            "baseline", cases,
            lambda c: c["pred"],
        )
        experiments["baseline"] = result

        # ── 2. Remove small components ──
        for min_sz in REMOVE_SIZES_MM3:
            name = f"remove_lt_{min_sz}mm3"
            print(f"  [2] {name}...")
            result = run_experiment(
                name, cases,
                lambda c, ms=min_sz: remove_small_components(c["pred"], ms, c["pixdim"]),
            )
            experiments[name] = result

        # ── 3. Remove small + fill holes ──
        # Use the best remove_tiny threshold from step 2
        best_remove = max(
            [(k, v) for k, v in experiments.items() if k.startswith("remove_lt_")],
            key=lambda x: x[1]["dice_mean"]
        )
        best_remove_name, _ = best_remove
        best_remove_sz = int(best_remove_name.split("_")[2].replace("mm3", ""))
        print(f"\n  Best remove threshold: {best_remove_sz} mm³")

        for fill_sz in FILL_SIZES_MM3:
            name = f"remove_{best_remove_sz}_fill_{fill_sz}mm3"
            print(f"  [3] {name}...")
            result = run_experiment(
                name, cases,
                lambda c, rs=best_remove_sz, fs=fill_sz: fill_small_holes(
                    remove_small_components(c["pred"], rs, c["pixdim"]),
                    fs, c["pixdim"]
                ),
            )
            experiments[name] = result

        # ── 4. Threshold experiments (if softmax available) ──
        if has_softmax:
            for thr in THRESHOLDS:
                name = f"threshold_{thr:.2f}"
                print(f"  [4] {name}...")
                result = run_experiment(
                    name, cases,
                    lambda c, t=thr: apply_threshold(c["softmax"], t) if c["softmax"] is not None else c["pred"],
                )
                experiments[name] = result

            # ── 5. Best threshold + remove small ──
            thr_experiments = {k: v for k, v in experiments.items() if k.startswith("threshold_")}
            if thr_experiments:
                best_thr_name = max(thr_experiments, key=lambda k: thr_experiments[k]["dice_mean"])
                best_thr = float(best_thr_name.split("_")[1])
                print(f"\n  Best threshold: {best_thr}")

                for min_sz in [3, 5, 10, 15]:
                    name = f"thr_{best_thr:.2f}_remove_{min_sz}mm3"
                    print(f"  [5] {name}...")
                    result = run_experiment(
                        name, cases,
                        lambda c, t=best_thr, ms=min_sz: remove_small_components(
                            apply_threshold(c["softmax"], t) if c["softmax"] is not None else c["pred"],
                            ms, c["pixdim"]
                        ),
                    )
                    experiments[name] = result

        elapsed = time.time() - t0
        all_results[ds_key] = experiments

        # ── Print results table ──
        print(f"\n{'─' * 100}")
        print(f"  RESULTS: {ds_cfg['name']} ({elapsed:.0f}s)")
        print(f"{'─' * 100}")
        header = (f"{'Strategy':<35} {'DSC_m':>7} {'DSC_md':>7} {'HD95_m':>7} "
                  f"{'Sens_m':>7} {'PPV_m':>7} {'NSD_m':>7} "
                  f"{'L_Sens':>7} {'L_PPV':>7} {'L_F1':>7} {'L_FP':>6}")
        print(header)
        print("-" * 100)

        # Sort by DSC mean
        for name, res in sorted(experiments.items(), key=lambda x: -x[1].get("dice_mean", 0)):
            marker = " ***" if name != "baseline" and res.get("dice_mean", 0) > experiments["baseline"].get("dice_mean", 0) else ""
            print(
                f"  {name:<33} "
                f"{res.get('dice_mean', 0):>7.4f} "
                f"{res.get('dice_median', 0):>7.4f} "
                f"{res.get('hd95_mean', 0):>7.2f} "
                f"{res.get('sensitivity_mean', 0):>7.4f} "
                f"{res.get('ppv_mean', 0):>7.4f} "
                f"{res.get('nsd_mean', 0):>7.4f} "
                f"{res.get('lesion_sensitivity', 0):>7.4f} "
                f"{res.get('lesion_ppv', 0):>7.4f} "
                f"{res.get('lesion_f1', 0):>7.4f} "
                f"{res.get('lesion_fp', 0):>6d}"
                f"{marker}"
            )

    # ── Save all results to CSV ──
    csv_path = OUTPUT_DIR / "postprocessing_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["dataset", "strategy",
                      "dice_mean", "dice_median", "hd95_mean", "hd95_median",
                      "sensitivity_mean", "ppv_mean", "nsd_mean", "avd_mean",
                      "lesion_sensitivity", "lesion_ppv", "lesion_f1",
                      "lesion_tp", "lesion_fn", "lesion_fp", "lesion_n_gt"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ds_key, experiments in all_results.items():
            for name, res in experiments.items():
                row = {"dataset": ds_key, "strategy": name}
                for f in fieldnames[2:]:
                    row[f] = f"{res.get(f, 0):.6f}" if isinstance(res.get(f, 0), float) else res.get(f, 0)
                writer.writerow(row)
    print(f"\n  Results saved to: {csv_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_experiments()

    # Print final recommendation
    print("\n" + "=" * 80)
    print("  RECOMMENDATIONS")
    print("=" * 80)
    for ds_key, experiments in results.items():
        # Best by overall DSC mean
        best_dsc = max(experiments.items(), key=lambda x: x[1].get("dice_mean", 0))
        # Best by lesion F1
        best_lf1 = max(experiments.items(), key=lambda x: x[1].get("lesion_f1", 0))
        # Best by combined score (DSC + lesion_F1) / 2
        best_combined = max(experiments.items(),
                           key=lambda x: (x[1].get("dice_mean", 0) + x[1].get("lesion_f1", 0)) / 2)

        print(f"\n  {ds_key}:")
        print(f"    Best DSC:      {best_dsc[0]} (DSC={best_dsc[1]['dice_mean']:.4f})")
        print(f"    Best LesF1:    {best_lf1[0]} (LF1={best_lf1[1]['lesion_f1']:.4f})")
        print(f"    Best Combined: {best_combined[0]} "
              f"(DSC={best_combined[1]['dice_mean']:.4f}, LF1={best_combined[1]['lesion_f1']:.4f})")
