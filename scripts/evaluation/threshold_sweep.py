#!/usr/bin/env python3
"""
threshold_sweep.py
==================
Sweep probability thresholds (0.30 → 0.50) across all TTA configs and report
mean DSC + NSD per threshold, for thesis documentation.

Configs:
  DS001 — Single TTA (DS001 model), DS003 alone, Ensemble average
  DS002 — Single TTA (DS002 model), DS003 alone, Ensemble average

Softmax source: *.npz in each TTA prediction directory.
IMPORTANT: nnU-Net NPZ stores softmax in (z,y,x) order. Always transpose to (x,y,z).

Usage:
    python scripts/evaluation/threshold_sweep.py
"""

from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_test_sets import compute_metrics

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
RESULTS_BASE = Path(os.environ.get("nnUNet_results",
                                   str(REPO_ROOT / "data" / "nnUNet_results")))
RAW_BASE     = Path(os.environ.get("nnUNet_raw",
                                   str(REPO_ROOT / "data" / "nnUNet_raw")))
PRED_BASE    = RESULTS_BASE / "predictions"

DATASETS = {
    "DS001": {
        "tta_dir":     PRED_BASE / "DS001_ResEncL_3D_TTA",
        "ds003_dir":   PRED_BASE / "DS001_DS003_ResEncL_3D_TTA",
        "gt_dir":      RAW_BASE  / "Dataset001_MSLesSeg" / "labelsTs",
        "name":        "DS001 (MSLesSeg-2024, 22 subjects)",
    },
    "DS002": {
        "tta_dir":     PRED_BASE / "DS002_ResEncL_3D_TTA",
        "ds003_dir":   PRED_BASE / "DS002_DS003_ResEncL_3D_TTA",
        "gt_dir":      RAW_BASE  / "Dataset002_WMH"      / "labelsTs",
        "name":        "DS002 (WMH Challenge, 110 subjects)",
    },
}

THRESHOLDS  = [0.30, 0.35, 0.40, 0.45, 0.50]
METRIC_KEYS = ["dice", "nsd"]   # we print all but focus on these two


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def load_softmax(npz_path: Path) -> np.ndarray:
    """Return foreground probability map with shape (x,y,z)."""
    data = np.load(npz_path)
    key  = [k for k in data.files if "softmax" in k.lower() or "probabilities" in k.lower()]
    key  = key[0] if key else data.files[0]
    arr  = data[key]          # may be (C, z, y, x) or (C, x, y, z)
    if arr.ndim == 4:
        arr = arr[1]          # foreground channel
    # nnU-Net stores (z,y,x) → transpose to (x,y,z)
    out = np.transpose(arr, (2, 1, 0))
    return out.astype(np.float32)


def load_gt(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    return (img.get_fdata() > 0.5).astype(np.uint8), img


def sweep_config(softmax_map_fn, gt_dir: Path, thresholds: list[float]) -> dict:
    """
    softmax_map_fn: callable(stem) → np.ndarray (x,y,z) foreground probability
    Returns {thr: {mean_dsc, std_dsc, mean_nsd, std_nsd, mean_hd95, n}}
    """
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    results   = {t: {"dice": [], "nsd": [], "hd95": [], "sens": [], "ppv": []}
                  for t in thresholds}

    for gt_path in gt_files:
        stem     = gt_path.name.replace(".nii.gz", "")
        gt_arr, gt_img = load_gt(gt_path)
        spacing  = gt_img.header.get_zooms()[:3]

        try:
            softmax = softmax_map_fn(stem)
        except FileNotFoundError:
            print(f"  [SKIP] {stem} — softmax not found")
            continue

        for thr in thresholds:
            pred = (softmax >= thr).astype(np.uint8)
            try:
                m = compute_metrics(pred, gt_arr, list(spacing))
                results[thr]["dice"].append(m.get("dice",        np.nan))
                results[thr]["nsd"].append(m.get("nsd",          np.nan))
                results[thr]["hd95"].append(m.get("hd95",        np.nan))
                results[thr]["sens"].append(m.get("sensitivity", np.nan))
                results[thr]["ppv"].append(m.get("ppv",          np.nan))
            except Exception as e:
                print(f"  [WARN] {stem} thr={thr}: {e}", flush=True)
                for k in ("dice", "nsd", "hd95", "sens", "ppv"):
                    results[thr][k].append(np.nan)

    out = {}
    for thr, vals in results.items():
        n = len([v for v in vals["dice"] if not np.isnan(v)])
        out[thr] = {
            k: float(np.nanmean(vals[k])) for k in ("dice", "nsd", "hd95", "sens", "ppv")
        }
        out[thr]["n"] = n
    return out


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for ds_key, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  {cfg['name']}")
        print(f"{'='*60}")

        tta_dir   = cfg["tta_dir"]
        ds003_dir = cfg["ds003_dir"]
        gt_dir    = cfg["gt_dir"]

        # ── Helper softmax loaders ─────────────────────────────────
        def make_loader(src_dir):
            def _load(stem):
                p = src_dir / f"{stem}.npz"
                if not p.exists():
                    raise FileNotFoundError(p)
                return load_softmax(p)
            return _load

        def ensemble_loader(stem):
            s1 = make_loader(tta_dir)(stem)
            s2 = make_loader(ds003_dir)(stem)
            return (s1 + s2) / 2.0

        configs = {
            "Single TTA":   make_loader(tta_dir),
            "DS003 alone":  make_loader(ds003_dir),
            "Ensemble avg": ensemble_loader,
        }

        ds_results = {}
        for cfg_name, loader in configs.items():
            print(f"\n  [{cfg_name}]")
            r = sweep_config(loader, gt_dir, THRESHOLDS)
            ds_results[cfg_name] = r

            # Print table
            hdr = f"  {'Thr':>5} | {'DSC':>7} | {'NSD':>7} | {'HD95':>7} | {'Sens':>7} | {'PPV':>7}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for thr in THRESHOLDS:
                v = r[thr]
                print(f"  {thr:>5.2f} | {v['dice']:>7.4f} | {v['nsd']:>7.4f} | "
                      f"{v['hd95']:>7.2f} | {v['sens']:>7.4f} | {v['ppv']:>7.4f}")

        all_results[ds_key] = ds_results

    # ── Markdown table dump ────────────────────────────────────────
    print("\n\n" + "="*80)
    print("MARKDOWN TABLES FOR PROJECT_STATE.md")
    print("="*80)

    for ds_key, ds_results in all_results.items():
        print(f"\n### {ds_key} — Threshold Sweep (TTA softmax, mean over all test subjects)\n")
        print("| Threshold | Config | DSC | NSD | HD95 | Sensitivity | PPV |")
        print("|-----------|--------|-----|-----|------|-------------|-----|")
        for thr in THRESHOLDS:
            for cfg_name, r in ds_results.items():
                v = r[thr]
                marker = " ← **optimal**" if (
                    (ds_key == "DS001" and thr == 0.30 and cfg_name == "Single TTA") or
                    (ds_key == "DS001" and thr == 0.50 and cfg_name == "DS003 alone") or
                    (ds_key == "DS001" and thr == 0.45 and cfg_name == "Ensemble avg") or
                    (ds_key == "DS002" and thr == 0.30 and cfg_name == "Single TTA") or
                    (ds_key == "DS002" and thr == 0.45 and cfg_name == "Ensemble avg")
                ) else ""
                print(f"| {thr:.2f} | {cfg_name}{marker} | {v['dice']:.4f} | "
                      f"{v['nsd']:.4f} | {v['hd95']:.2f} | {v['sens']:.4f} | {v['ppv']:.4f} |")

    print("\nDone.")


if __name__ == "__main__":
    main()

    sys.stdout.flush()
