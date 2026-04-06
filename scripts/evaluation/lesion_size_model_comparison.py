#!/usr/bin/env python3
"""
lesion_size_model_comparison.py
===============================
Compare lesion detection performance by size bin across model configurations.

Supports two config types per model:
  - Path (str/Path)         → load hard predictions from NIfTI directory
  - SoftmaxConfig(dirs, thr) → average softmax NPZ files on-the-fly, then threshold

For each config computes:
  - Per-size-bin detection rate (sensitivity)
  - Per-size-bin lesion Dice
  - False positive counts by size
  - Overall lesion-level F1

Outputs formatted comparison tables for thesis.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluation.lesion_level_analysis import analyze_lesions, SIZE_BINS, get_bin

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────────
# Softmax ensemble config helper
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SoftmaxConfig:
    """Average these softmax NPZ dirs and threshold to produce a hard prediction."""
    softmax_dirs: List[Path]
    threshold: float = 0.50
    softmax_channel: int = 1        # lesion channel index
    axis_perm: tuple = (2, 1, 0)    # nnU-Net NPZ is (z,y,x) → NIfTI is (x,y,z)


def _load_softmax_mean(softmax_dirs: List[Path], filename_stem: str,
                        channel: int, axis_perm: tuple) -> np.ndarray | None:
    """Load and average softmax channel from multiple NPZ dirs for one case."""
    arrays = []
    for sd in softmax_dirs:
        npz_path = sd / f"{filename_stem}.npz"
        if not npz_path.exists():
            # export_prediction_from_logits uses {case_id}.nii.gz.npz
            npz_path = sd / f"{filename_stem}.nii.gz.npz"
        if not npz_path.exists():
            return None
        data = np.load(str(npz_path))
        key = list(data.files)[0]          # typically 'softmax'
        arr = data[key]                    # shape (C, z, y, x) or (C, x, y, z)
        prob = np.transpose(arr[channel], axis_perm).astype(np.float32)
        arrays.append(prob)
    if not arrays:
        return None
    return np.mean(arrays, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PRED_BASE = REPO_ROOT / "results" / "predictions"
SOFTMAX_BASE = REPO_ROOT / "results" / "predictions"
RAW_BASE = Path(os.environ.get("nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw")))
GT_DS001 = RAW_BASE / "Dataset001_MSLesSeg" / "labelsTs"
GT_DS002 = RAW_BASE / "Dataset002_WMH" / "labelsTs"

CONFIGS = {
    "DS001": {
        "gt_dir": GT_DS001,
        "models": [
            # ── Single 3D models ──
            ("ResEncL-3D (noTTA)",    PRED_BASE / "DS001_ResEncL_3D"),
            ("ResEncL-3D TTA",        PRED_BASE / "DS001_final"),
            ("CNN-3D TTA", SoftmaxConfig(
                softmax_dirs=[SOFTMAX_BASE / "DS001_DS003_CNN_3D_TTA"],
                threshold=0.50,
            )),
            ("ResEncL-3D DS003 TTA", SoftmaxConfig(
                softmax_dirs=[SOFTMAX_BASE / "DS001_DS003_ResEncL_3D_TTA"],
                threshold=0.50,
            )),
            # ── Single 2.5D chfix models ──
            ("ResEncL-2.5D chfix", SoftmaxConfig(
                softmax_dirs=[SOFTMAX_BASE / "DS001_DS003_ResEncL_25D_TTA_chfix"],
                threshold=0.50,
            )),
            ("CNN-2.5D chfix", SoftmaxConfig(
                softmax_dirs=[SOFTMAX_BASE / "DS001_DS003_CNN_25D_TTA_chfix"],
                threshold=0.50,
            )),
            # ── Best 3D-only ensemble (old best) ──
            ("Ens 3D (old best)", SoftmaxConfig(
                softmax_dirs=[
                    SOFTMAX_BASE / "DS001_DS003_CNN_3D_TTA",
                    SOFTMAX_BASE / "DS001_DS003_ResEncL_3D_TTA",
                ],
                threshold=0.50,
            )),
            # ── New best: 3D + 2.5D ensemble ──
            ("Ens 3D+2.5D (NEW)", SoftmaxConfig(
                softmax_dirs=[
                    SOFTMAX_BASE / "DS001_DS003_CNN_3D_TTA",
                    SOFTMAX_BASE / "DS001_DS003_ResEncL_25D_TTA_chfix",
                    SOFTMAX_BASE / "DS001_DS003_ResEncL_3D_TTA",
                ],
                threshold=0.40,
            )),
        ],
    },
    "DS002": {
        "gt_dir": GT_DS002,
        "models": [
            # ── Single 3D models ──
            ("ResEncL-3D (noTTA)",   PRED_BASE / "DS002_ResEncL_3D"),
            ("ResEncL-3D TTA",       PRED_BASE / "DS002_final"),
            ("CNN-3D TTA", SoftmaxConfig(
                softmax_dirs=[SOFTMAX_BASE / "DS002_DS003_CNN_3D_TTA"],
                threshold=0.50,
            )),
            ("ResEncL-3D DS003 TTA", SoftmaxConfig(
                softmax_dirs=[SOFTMAX_BASE / "DS002_DS003_ResEncL_3D_TTA"],
                threshold=0.50,
            )),
            # ── Single 2.5D chfix models ──
            ("ResEncL-2.5D chfix", SoftmaxConfig(
                softmax_dirs=[SOFTMAX_BASE / "DS002_DS003_ResEncL_25D_TTA_chfix"],
                threshold=0.50,
            )),
            ("CNN-2.5D chfix", SoftmaxConfig(
                softmax_dirs=[SOFTMAX_BASE / "DS002_DS003_CNN_25D_TTA_chfix"],
                threshold=0.50,
            )),
            # ── Best 3D-only ensemble (old best) ──
            ("Ens 3D (old best)", SoftmaxConfig(
                softmax_dirs=[
                    SOFTMAX_BASE / "DS002_DS003_CNN_3D_TTA",
                    SOFTMAX_BASE / "DS002_DS003_ResEncL_3D_TTA",
                    SOFTMAX_BASE / "DS002_ResEncL_3D_TTA",
                ],
                threshold=0.50,
            )),
            # ── New best: 5-model 3D + 2.5D ensemble ──
            ("Ens 3D+2.5D (NEW)", SoftmaxConfig(
                softmax_dirs=[
                    SOFTMAX_BASE / "DS002_DS002_ResEncL_3D_TTA",
                    SOFTMAX_BASE / "DS002_DS003_CNN_25D_TTA_chfix",
                    SOFTMAX_BASE / "DS002_DS003_CNN_3D_TTA",
                    SOFTMAX_BASE / "DS002_DS003_ResEncL_25D_TTA_chfix",
                    SOFTMAX_BASE / "DS002_DS003_ResEncL_3D_TTA",
                ],
                threshold=0.50,
            )),
        ],
    },
}


def run_lesion_analysis(pred_source: Union[Path, SoftmaxConfig], gt_dir: Path) -> dict:
    """Run per-subject lesion analysis, return aggregated results.

    pred_source can be:
      - Path: directory of hard NIfTI prediction files
      - SoftmaxConfig: average softmax NPZs on-the-fly and threshold
    """
    gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".nii.gz"))
    all_gt_lesions = []
    all_fp_lesions = []
    total_tp = total_fn = total_fp = 0
    total_match_counts = defaultdict(int)

    for f in gt_files:
        gf = gt_dir / f
        gnii = nib.load(str(gf))
        g = (np.asarray(gnii.dataobj) > 0).astype(np.uint8)
        pixdim = [float(v) for v in gnii.header.get_zooms()[:3]]

        if isinstance(pred_source, SoftmaxConfig):
            stem = f[:-len(".nii.gz")]
            mean_prob = _load_softmax_mean(
                pred_source.softmax_dirs, stem,
                pred_source.softmax_channel, pred_source.axis_perm)
            if mean_prob is None:
                continue
            p = (mean_prob >= pred_source.threshold).astype(np.uint8)
        else:
            pf = pred_source / f
            if not pf.exists():
                continue
            pnii = nib.load(str(pf))
            p = (np.asarray(pnii.dataobj) > 0).astype(np.uint8)

        result = analyze_lesions(g, p, pixdim)
        all_gt_lesions.extend(result["gt_lesions"])
        all_fp_lesions.extend(result["fp_lesions"])
        total_tp += result["tp"]
        total_fn += result["fn"]
        total_fp += result["fp"]
        for k, v in result["match_counts"].items():
            total_match_counts[k] += v

    # Aggregate by size bin
    bin_stats = {}
    for bin_name, _, _ in SIZE_BINS:
        gt_in = [l for l in all_gt_lesions if l["bin"] == bin_name]
        tp_in = sum(1 for l in gt_in if l["detected"])
        fn_in = sum(1 for l in gt_in if not l["detected"])
        fp_in = [l for l in all_fp_lesions if l["bin"] == bin_name]

        det_rate = tp_in / max(1, len(gt_in))
        dices = [l["lesion_dice"] for l in gt_in if l["detected"] and l["lesion_dice"] > 0]
        mean_dice = np.mean(dices) if dices else 0.0

        # Per-bin match topology
        bin_match = {}
        for mt in ["one_to_one", "split", "merge", "complex", "fn"]:
            bin_match[mt] = sum(1 for l in gt_in if l["match_type"] == mt)

        bin_stats[bin_name] = {
            "n_gt": len(gt_in),
            "tp": tp_in,
            "fn": fn_in,
            "fp": len(fp_in),
            "det_rate": det_rate,
            "mean_dice": mean_dice,
            "match_topology": bin_match,
        }

    # Overall
    l_sens = total_tp / max(1, total_tp + total_fn)
    l_ppv = total_tp / max(1, total_tp + total_fp)
    l_f1 = 2 * l_sens * l_ppv / max(1e-9, l_sens + l_ppv)

    return {
        "bins": bin_stats,
        "total_gt": len(all_gt_lesions),
        "total_tp": total_tp,
        "total_fn": total_fn,
        "total_fp": len(all_fp_lesions),
        "sensitivity": l_sens,
        "ppv": l_ppv,
        "f1": l_f1,
        "match_counts": dict(total_match_counts),
    }


def print_comparison(dataset_name: str, gt_dir: Path, models: list):
    """Print side-by-side comparison of lesion detection by size bin."""
    print(f"\n{'='*120}")
    print(f"  LESION DETECTION BY SIZE — {dataset_name}")
    print(f"{'='*120}")

    # Run analysis for each model
    results = {}
    for model_name, pred_source in models:
        t0 = time.time()
        results[model_name] = run_lesion_analysis(pred_source, gt_dir)
        elapsed = time.time() - t0
        r = results[model_name]
        print(f"  [{model_name}] {r['total_gt']} GT lesions, "
              f"Sens={r['sensitivity']:.4f} PPV={r['ppv']:.4f} F1={r['f1']:.4f} "
              f"({elapsed:.0f}s)")

    model_names = [m[0] for m in models]

    # ── Detection Rate by Size Bin ──
    print(f"\n  {'DETECTION RATE (Sensitivity) by Size':}")
    header = f"  {'Size Bin':<26} {'#GT':>5}"
    for mn in model_names:
        header += f"  {mn:>22}"
    print(header)
    print(f"  {'-'*26} {'-'*5}" + f"  {'-'*22}" * len(model_names))

    for bin_name, _, _ in SIZE_BINS:
        n_gt = results[model_names[0]]["bins"][bin_name]["n_gt"]
        row = f"  {bin_name:<26} {n_gt:>5}"
        for mn in model_names:
            dr = results[mn]["bins"][bin_name]["det_rate"]
            tp = results[mn]["bins"][bin_name]["tp"]
            row += f"  {dr:>7.1%} ({tp:>4}/{n_gt:<4})"
        print(row)

    # Overall
    n_gt_total = results[model_names[0]]["total_gt"]
    row = f"  {'OVERALL':<26} {n_gt_total:>5}"
    for mn in model_names:
        s = results[mn]["sensitivity"]
        tp = results[mn]["total_tp"]
        row += f"  {s:>7.1%} ({tp:>4}/{n_gt_total:<4})"
    print(row)

    # ── Mean Lesion Dice by Size Bin ──
    print(f"\n  {'MEAN LESION DICE (detected lesions only) by Size':}")
    header = f"  {'Size Bin':<26}"
    for mn in model_names:
        header += f"  {mn:>22}"
    print(header)
    print(f"  {'-'*26}" + f"  {'-'*22}" * len(model_names))

    for bin_name, _, _ in SIZE_BINS:
        row = f"  {bin_name:<26}"
        for mn in model_names:
            md = results[mn]["bins"][bin_name]["mean_dice"]
            row += f"  {md:>22.4f}"
        print(row)

    # ── False Positives by Size ──
    print(f"\n  {'FALSE POSITIVES by Size':}")
    header = f"  {'Size Bin':<26}"
    for mn in model_names:
        header += f"  {mn:>22}"
    print(header)
    print(f"  {'-'*26}" + f"  {'-'*22}" * len(model_names))

    for bin_name, _, _ in SIZE_BINS:
        row = f"  {bin_name:<26}"
        for mn in model_names:
            fp = results[mn]["bins"][bin_name]["fp"]
            row += f"  {fp:>22}"
        print(row)

    total_row = f"  {'TOTAL FP':<26}"
    for mn in model_names:
        total_row += f"  {results[mn]['total_fp']:>22}"
    print(total_row)

    # ── Overall Summary ──
    print(f"\n  {'OVERALL LESION METRICS':}")
    header = f"  {'Metric':<26}"
    for mn in model_names:
        header += f"  {mn:>22}"
    print(header)
    print(f"  {'-'*26}" + f"  {'-'*22}" * len(model_names))

    for metric, label in [("sensitivity", "Sensitivity"),
                           ("ppv", "PPV (Precision)"),
                           ("f1", "F1 Score")]:
        row = f"  {label:<26}"
        for mn in model_names:
            row += f"  {results[mn][metric]:>22.4f}"
        print(row)

    # ── Delta analysis: improvement from baseline ──
    baseline = model_names[0]
    print(f"\n  {'DETECTION RATE IMPROVEMENT vs Baseline':}")
    header = f"  {'Size Bin':<26}"
    for mn in model_names[1:]:
        header += f"  {mn:>22}"
    print(header)
    print(f"  {'-'*26}" + f"  {'-'*22}" * (len(model_names) - 1))

    for bin_name, _, _ in SIZE_BINS:
        base_dr = results[baseline]["bins"][bin_name]["det_rate"]
        row = f"  {bin_name:<26}"
        for mn in model_names[1:]:
            dr = results[mn]["bins"][bin_name]["det_rate"]
            delta = dr - base_dr
            sign = "+" if delta >= 0 else ""
            row += f"  {sign}{delta:>7.1%}pp            "
        print(row)

    base_sens = results[baseline]["sensitivity"]
    row = f"  {'OVERALL':<26}"
    for mn in model_names[1:]:
        delta = results[mn]["sensitivity"] - base_sens
        sign = "+" if delta >= 0 else ""
        row += f"  {sign}{delta:>7.1%}pp            "
    print(row)

    # ── Match Topology (split / merge / 1:1) ──
    print(f"\n  {'MATCH TOPOLOGY (how GT lesions were matched to predictions)':}")
    header = f"  {'Match Type':<26}"
    for mn in model_names:
        header += f"  {mn:>22}"
    print(header)
    print(f"  {'-'*26}" + f"  {'-'*22}" * len(model_names))

    for mt, label in [("one_to_one", "1:1 (correct match)"),
                       ("split", "Split (GT→N pred)"),
                       ("merge", "Merge (N GT→1 pred)"),
                       ("complex", "Complex (M:N)"),
                       ("fn", "Missed (FN)")]:
        row = f"  {label:<26}"
        for mn in model_names:
            mc = results[mn]["match_counts"]
            cnt = mc.get(mt, 0)
            total = results[mn]["total_gt"]
            pct = cnt / max(1, total) * 100
            row += f"  {cnt:>6} ({pct:>5.1f}%)        "
        print(row)

    # Per-size-bin topology for last two models (old best vs new best)
    if len(model_names) >= 2:
        for mn in model_names[-2:]:
            print(f"\n    [{mn}] per-size topology:")
            for bin_name, _, _ in SIZE_BINS:
                bm = results[mn]["bins"][bin_name]["match_topology"]
                n = results[mn]["bins"][bin_name]["n_gt"]
                parts = []
                for mt in ["one_to_one", "split", "merge", "complex", "fn"]:
                    if bm[mt] > 0:
                        parts.append(f"{mt}={bm[mt]}")
                print(f"      {bin_name:<26}  n={n:>5}  {', '.join(parts)}")

    return results


def main():
    t0 = time.time()
    all_results = {}
    for ds_name, cfg in CONFIGS.items():
        all_results[ds_name] = print_comparison(
            ds_name, cfg["gt_dir"], cfg["models"])

    print(f"\n{'='*120}")
    print(f"  Total analysis time: {time.time()-t0:.0f}s")
    print(f"{'='*120}")

    return all_results


if __name__ == "__main__":
    main()
