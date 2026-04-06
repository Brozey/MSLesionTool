#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_test_sets import compute_metrics


THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]


def fast_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))

    denom_dice = (2.0 * tp + fp + fn)
    dice = 2.0 * tp / denom_dice if denom_dice > 0 else 1.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return {"dice": dice, "sensitivity": sens, "ppv": ppv}


def load_prob(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    key = "softmax" if "softmax" in d.files else "probabilities"
    arr = d[key][1].astype(np.float32)
    return np.transpose(arr, (2, 1, 0))


def find_models(pred_base: Path, dataset_key: str, expected_count: int) -> list[tuple[str, Path]]:
    """Find per-fold prediction directories for a given dataset.

    For DS001: dirs like 25D_fold0_TTA, CNN_3D_fold0_TTA, ResEncL_3D_fold0_TTA (no DS002 suffix)
    For DS002: dirs like 25D_fold0_TTA_DS002, CNN_3D_fold0_TTA_DS002, ResEncL_3D_fold0_TTA_DS002
    """
    import re
    models = []
    fold_pattern = re.compile(r"fold\d")
    for d in sorted(pred_base.iterdir()):
        if not d.is_dir():
            continue
        # Must contain "fold<digit>" but NOT "5fold" (those are ensembles)
        if not fold_pattern.search(d.name) or "5fold" in d.name:
            continue
        # DS001 → dirs without DS002 suffix; DS002 → dirs with DS002 suffix
        has_ds002_suffix = d.name.endswith("_DS002")
        if dataset_key == "DS001" and has_ds002_suffix:
            continue
        if dataset_key == "DS002" and not has_ds002_suffix:
            continue
        n = len(list(d.glob("*.npz")))
        if n == expected_count:
            models.append((d.name, d))
    return models


def evaluate_dataset(
    dataset_key: str,
    gt_dir: Path,
    models: list[tuple[str, Path]],
    out_dir: Path,
    max_combo: int | None,
    top_full_metrics: int,
):
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise RuntimeError(f"No GT files in {gt_dir}")

    model_names = [m[0] for m in models]
    model_dirs = {m[0]: m[1] for m in models}

    # Preload all GT masks and softmax predictions into RAM
    stems = [f.name.replace(".nii.gz", "") for f in gt_files]
    print(f"[{dataset_key}] Preloading GT masks...", flush=True)
    gt_cache = {}
    for gt_path in gt_files:
        stem = gt_path.name.replace(".nii.gz", "")
        gt_cache[stem] = (nib.load(str(gt_path)).get_fdata() > 0.5).astype(np.uint8)

    print(f"[{dataset_key}] Preloading {len(model_names)} x {len(stems)} softmax predictions...", flush=True)
    prob_cache = {}  # (model_name, stem) -> np.float32 array
    for mn in model_names:
        for stem in stems:
            # nnU-Net names npz as either "{stem}.npz" or "{stem}.nii.gz.npz"
            npz_path = model_dirs[mn] / f"{stem}.npz"
            if not npz_path.exists():
                npz_path = model_dirs[mn] / f"{stem}.nii.gz.npz"
            if npz_path.exists():
                prob_cache[(mn, stem)] = load_prob(npz_path)
    print(f"[{dataset_key}] Preloaded {len(prob_cache)} softmax arrays", flush=True)

    combo_rows = []
    per_case_rows = []

    max_k = len(model_names) if max_combo is None else min(max_combo, len(model_names))

    combos = []
    for k in range(1, max_k + 1):
        combos.extend(itertools.combinations(range(len(model_names)), k))

    print(f"[{dataset_key}] Models={len(model_names)} combos={len(combos)} thresholds={len(THRESHOLDS)}", flush=True)

    # Pre-stack all softmax into (N_models, N_voxels) per case, flatten GT
    # OPTIMIZATION: only keep voxels where GT > 0 OR any model softmax > 0.05
    # This reduces 8M+ voxels to ~50-200K, giving ~50-100x speedup
    stacked = {}  # stem -> (N_models, N_relevant_voxels) float32
    gt_flat = {}  # stem -> (N_relevant_voxels,) bool
    gt_total_vox = {}  # stem -> total GT positive voxels (for correct FN computation)
    for stem in stems:
        arrs = []
        for mn in model_names:
            key = (mn, stem)
            if key in prob_cache:
                arrs.append(prob_cache[key].ravel())
            else:
                arrs.append(None)
        if any(a is None for a in arrs):
            continue
        full_stack = np.stack(arrs, axis=0)  # (15, N_voxels)
        gt_bool = gt_cache[stem].ravel().astype(bool)
        # Mask: GT positive OR any model has softmax >= lowest threshold
        relevant = gt_bool | (full_stack.max(axis=0) >= min(THRESHOLDS) - 0.01)
        stacked[stem] = full_stack[:, relevant]
        gt_flat[stem] = gt_bool[relevant]
        print(f"  {stem}: {relevant.sum()}/{len(gt_bool)} relevant voxels ({100*relevant.sum()/len(gt_bool):.1f}%)", flush=True)

    # Free the separate prob_cache — stacked has all data now
    del prob_cache
    import gc; gc.collect()

    valid_stems = list(stacked.keys())
    n_cases = len(valid_stems)
    print(f"[{dataset_key}] Pre-stacked {n_cases} cases", flush=True)

    thresholds_arr = np.array(THRESHOLDS, dtype=np.float32)

    t_start = time.time()
    for ci, combo_idx in enumerate(combos):
        combo_name = " + ".join(model_names[i] for i in combo_idx)
        if ci % 1000 == 0:
            elapsed = time.time() - t_start
            rate = ci / elapsed if elapsed > 0 else 0
            print(f"  [{dataset_key}] combo {ci+1}/{len(combos)} ({elapsed:.1f}s, {rate:.0f}/s) — {combo_name}", flush=True)

        idx = list(combo_idx)
        # Per-threshold aggregation
        dice_sums = np.zeros(len(THRESHOLDS))
        sens_sums = np.zeros(len(THRESHOLDS))
        ppv_sums = np.zeros(len(THRESHOLDS))

        for stem in valid_stems:
            avg = stacked[stem][idx].mean(axis=0)  # (N_voxels,)
            gt_b = gt_flat[stem]

            for ti, thr in enumerate(THRESHOLDS):
                pred_b = avg >= thr
                tp = float(np.sum(pred_b & gt_b))
                fp = float(np.sum(pred_b & ~gt_b))
                fn = float(np.sum(~pred_b & gt_b))
                denom = 2.0 * tp + fp + fn
                dice_sums[ti] += (2.0 * tp / denom) if denom > 0 else 1.0
                sens_sums[ti] += (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
                ppv_sums[ti] += (tp / (tp + fp)) if (tp + fp) > 0 else 1.0

        best_thr_idx = -1
        best_dice = -1.0
        for ti, thr in enumerate(THRESHOLDS):
            dsc = dice_sums[ti] / n_cases
            sens = sens_sums[ti] / n_cases
            ppv = ppv_sums[ti] / n_cases
            combo_rows.append({
                "dataset": dataset_key,
                "combo_size": len(combo_idx),
                "combo": combo_name,
                "threshold": thr,
                "dice": dsc,
                "sensitivity": sens,
                "ppv": ppv,
            })
            if dsc > best_dice:
                best_dice = dsc
                best_thr_idx = ti

        per_case_rows.append({
            "dataset": dataset_key,
            "combo_size": len(combo_idx),
            "combo": combo_name,
            "best_threshold": THRESHOLDS[best_thr_idx],
            "best_dice": best_dice,
        })

    elapsed = time.time() - t_start
    print(f"  [{dataset_key}] fast metrics done: {len(combo_rows)} rows in {elapsed:.1f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{dataset_key}_all_combo_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(combo_rows[0].keys()) if combo_rows else ["dataset","combo_size","combo","threshold","dice","sensitivity","ppv"])
        w.writeheader()
        for r in combo_rows:
            w.writerow(r)

    with (out_dir / f"{dataset_key}_best_per_combo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()) if per_case_rows else ["dataset","combo_size","combo","best_threshold","best_dice"])
        w.writeheader()
        for r in sorted(per_case_rows, key=lambda x: x["best_dice"], reverse=True):
            w.writerow(r)

    top_rows = sorted(per_case_rows, key=lambda x: x["best_dice"], reverse=True)[:top_full_metrics]
    print(f"  [{dataset_key}] computing full metrics for top {len(top_rows)} combos...", flush=True)

    # Free stacked arrays before full metrics (reload from disk for top-N)
    del stacked, gt_flat
    gc.collect()

    full_rows = []
    for ri, row in enumerate(top_rows):
        print(f"    [{ri+1}/{len(top_rows)}] {row['combo']} (fast DSC={row['best_dice']:.4f})", flush=True)
        combo = tuple(row["combo"].split(" + "))
        thr = float(row["best_threshold"])
        vals = {"dice": [], "nsd": [], "hd95": [], "sensitivity": [], "ppv": []}
        for gt_path in gt_files:
            stem = gt_path.name.replace(".nii.gz", "")
            gt_img = nib.load(str(gt_path))
            gt = gt_cache[stem]
            pixdim = list(gt_img.header.get_zooms()[:3])

            probs = []
            missing = False
            for m in combo:
                npz_path = model_dirs[m] / f"{stem}.npz"
                if not npz_path.exists():
                    npz_path = model_dirs[m] / f"{stem}.nii.gz.npz"
                if not npz_path.exists():
                    missing = True
                    break
                probs.append(load_prob(npz_path))
            if missing:
                continue

            pred = (np.mean(np.stack(probs, axis=0), axis=0) >= thr).astype(np.uint8)
            mm = compute_metrics(pred, gt, pixdim)
            for k in vals.keys():
                vals[k].append(float(mm.get(k, np.nan)))

        full_rows.append(
            {
                "dataset": dataset_key,
                "combo": row["combo"],
                "threshold": thr,
                "dice": float(np.nanmean(vals["dice"])),
                "nsd": float(np.nanmean(vals["nsd"])),
                "hd95": float(np.nanmean(vals["hd95"])),
                "sensitivity": float(np.nanmean(vals["sensitivity"])),
                "ppv": float(np.nanmean(vals["ppv"])),
            }
        )

    with (out_dir / f"{dataset_key}_top{top_full_metrics}_full_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "combo", "threshold", "dice", "nsd", "hd95", "sensitivity", "ppv"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in full_rows:
            w.writerow(r)

    total_time = time.time() - t_start
    print(f"[{dataset_key}] wrote CSVs to {out_dir} (total: {total_time:.1f}s)", flush=True)


def main():
    RAW_BASE = Path(os.environ.get("nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw")))

    parser = argparse.ArgumentParser(description="Exhaustive all-combination ensemble analysis from softmax dirs")
    parser.add_argument("--pred-base", type=Path, default=REPO_ROOT / "results" / "predictions")
    parser.add_argument("--raw-base", type=Path, default=RAW_BASE)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "evaluation" / "exhaustive_ensembles")
    parser.add_argument("--max-combo", type=int, default=None, help="Optional cap on combo size")
    parser.add_argument("--top-full-metrics", type=int, default=15, help="Compute heavy metrics only for top N combos")
    parser.add_argument("--dataset", type=str, default=None, help="Run only this dataset (DS001 or DS002)")
    args = parser.parse_args()

    ds_cfg = {
        "DS001": args.raw_base / "Dataset001_MSLesSeg" / "labelsTs",
        "DS002": args.raw_base / "Dataset002_WMH" / "labelsTs",
    }

    for ds_key, gt_dir in ds_cfg.items():
        if args.dataset and ds_key != args.dataset:
            continue
        exp = len(list(gt_dir.glob("*.nii.gz")))
        models = find_models(args.pred_base, ds_key, exp)
        print(f"[{ds_key}] complete softmax dirs: {len(models)}")
        for m, _ in models:
            print(f"  - {m}")
        evaluate_dataset(ds_key, gt_dir, models, args.out_dir, args.max_combo, args.top_full_metrics)


if __name__ == "__main__":
    main()
