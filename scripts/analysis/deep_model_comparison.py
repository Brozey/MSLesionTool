#!/usr/bin/env python3
"""
deep_model_comparison.py — Comprehensive multi-metric comparison of all important models.

Computes per-case: Dice, HD95, AVD, Recall, Precision, F1, plus lesion-level detection rates
by size category. Outputs summary tables and per-case CSV.

Usage:
    python deep_model_comparison.py
"""
import os
import glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label, generate_binary_structure
from collections import defaultdict
import csv
import sys

# ==== CONFIGURATION ====
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = str(REPO_ROOT)
PRED_BASE = os.path.join(BASE, "results", "predictions")
GT_MSL = os.path.join(BASE, "data", "nnUNet_raw", "Dataset001_MSLesSeg", "labelsTs")
GT_WMH = os.path.join(BASE, "data", "nnUNet_raw", "Dataset002_WMH", "labelsTs")

# Models to compare on MSLesSeg-22 test set
MSL_MODELS = {
    # (display_name, pred_dir_name, description)
    "DS001_ResEncL_3D":           ("DS001_ResEncL_3D",           "DS001 ResEncL fold0 (MSL-only, no TTA)"),
    "DS001_ResEncL_3D_TTA":       ("DS001_ResEncL_3D_TTA",       "DS001 ResEncL fold0 (MSL-only, TTA)"),
    "DS001_DS002_ResEncL_3D_TTA": ("DS001_DS002_ResEncL_3D_TTA", "DS002 ResEncL fold0 on MSL (WMH-trained, TTA)"),
    "DS001_DS003_CNN_3D_TTA":     ("DS001_DS003_CNN_3D_TTA",     "DS003 CNN fold0 (Combined, TTA)"),
    "DS001_DS003_ResEncL_3D_TTA": ("DS001_DS003_ResEncL_3D_TTA", "DS003 ResEncL fold0 (Combined, TTA)"),
    "DS001_DS003_ResEncL_25D_TTA_chfix": ("DS001_DS003_ResEncL_25D_TTA_chfix", "DS003 ResEncL 2.5D fold0 (Combined, TTA)"),
    "DS003_TopK_ResEncL_fold0_TTA": ("DS003_TopK_ResEncL_fold0_TTA", "DS003 TopK ResEncL fold0 (Combined+TopK, TTA)"),
    "DS007_5fold_TTA_binary":     ("DS007_5fold_TTA_binary",     "DS007 5-fold ensemble (MultiSize→binary, TTA)"),
    "DS007_5fold_binary":         ("DS007_5fold_binary",         "DS007 5-fold ensemble (MultiSize→binary, no TTA)"),
    "DS001_DS003_final":          ("DS001_DS003_final",          "Best ensemble (CNN+ResEncL+ResEncL25D, thr=0.40)"),
    "DS001_final":                ("DS001_final",                "Single-model final (thr=0.30)"),
}

# Models to compare on WMH-110 test set
WMH_MODELS = {
    "DS002_ResEncL_3D":           ("DS002_ResEncL_3D",           "DS002 ResEncL fold0 (WMH-only, no TTA)"),
    "DS002_ResEncL_3D_TTA":       ("DS002_ResEncL_3D_TTA",       "DS002 ResEncL fold0 (WMH-only, TTA)"),
    "DS002_DS003_CNN_3D_TTA":     ("DS002_DS003_CNN_3D_TTA",     "DS003 CNN fold0 (Combined, TTA)"),
    "DS002_DS003_ResEncL_3D_TTA": ("DS002_DS003_ResEncL_3D_TTA", "DS003 ResEncL fold0 (Combined, TTA)"),
    "DS002_DS003_ResEncL_25D_TTA_chfix": ("DS002_DS003_ResEncL_25D_TTA_chfix", "DS003 ResEncL 2.5D fold0 (Combined, TTA)"),
    "DS002_DS003_CNN_25D_TTA_chfix": ("DS002_DS003_CNN_25D_TTA_chfix", "DS003 CNN 2.5D fold0 (Combined, TTA)"),
    "DS002_final":                ("DS002_final",                "Best WMH ensemble (5-model, thr=0.50)"),
}

# Size bins (voxels at 1mm iso = mm³)
SIZE_BINS = [
    ("tiny",   0,     10),
    ("small",  10,    100),
    ("medium", 100,   1000),
    ("large",  1000,  10000),
    ("huge",   10000, float('inf')),
]


def compute_metrics(pred_arr, gt_arr, spacing=(1.0, 1.0, 1.0)):
    """Compute comprehensive metrics between binary prediction and GT."""
    pred = pred_arr > 0
    gt = gt_arr > 0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    # Dice
    if gt.sum() == 0 and pred.sum() == 0:
        dice = 1.0
    elif gt.sum() == 0 or pred.sum() == 0:
        dice = 0.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)

    # Recall (sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if pred.sum() == 0 else 0.0)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt.sum() == 0 else 0.0)

    # F1 (same as Dice for binary)
    f1 = dice

    # AVD (absolute volume difference) as fraction
    gt_vol = gt.sum()
    pred_vol = pred.sum()
    avd = abs(pred_vol - gt_vol) / gt_vol if gt_vol > 0 else (0.0 if pred_vol == 0 else float('inf'))

    # Volume ratio
    vol_ratio = pred_vol / gt_vol if gt_vol > 0 else float('inf')

    # HD95
    hd95 = compute_hd95(pred, gt, spacing)

    return {
        'dice': dice,
        'hd95': hd95,
        'avd': avd,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'gt_vol': int(gt_vol),
        'pred_vol': int(pred_vol),
        'vol_ratio': vol_ratio,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }


def compute_hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """Compute 95th percentile Hausdorff distance."""
    from scipy.ndimage import distance_transform_edt

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float('inf')

    # Surface voxels (boundary)
    from scipy.ndimage import binary_erosion
    pred_border = np.logical_xor(pred, binary_erosion(pred))
    gt_border = np.logical_xor(gt, binary_erosion(gt))

    if pred_border.sum() == 0:
        pred_border = pred
    if gt_border.sum() == 0:
        gt_border = gt

    # Distance transforms
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)

    # Distances from GT surface to pred surface
    d_gt_to_pred = dt_pred[gt_border]
    d_pred_to_gt = dt_gt[pred_border]

    all_distances = np.concatenate([d_gt_to_pred, d_pred_to_gt])
    return np.percentile(all_distances, 95)


def analyze_lesion_detection(pred, gt, size_bins):
    """Per-lesion detection analysis by size."""
    struct = generate_binary_structure(3, 2)  # 18-connectivity
    gt_cc, n_gt = cc_label(gt, structure=struct)

    results = {b[0]: {'total': 0, 'detected': 0, 'sizes': []} for b in size_bins}

    for lid in range(1, n_gt + 1):
        lesion = gt_cc == lid
        size = int(lesion.sum())

        bin_name = None
        for bname, bmin, bmax in size_bins:
            if bmin <= size < bmax:
                bin_name = bname
                break
        if bin_name is None:
            continue

        results[bin_name]['total'] += 1
        results[bin_name]['sizes'].append(size)

        overlap = np.logical_and(lesion, pred).sum()
        if overlap / size >= 0.1:  # 10% overlap = detected
            results[bin_name]['detected'] += 1

    # False positive lesions
    pred_cc, n_pred = cc_label(pred, structure=struct)
    fp_lesions = 0
    for lid in range(1, n_pred + 1):
        pred_lesion = pred_cc == lid
        overlap = np.logical_and(pred_lesion, gt).sum()
        if overlap / pred_lesion.sum() < 0.1:
            fp_lesions += 1

    results['_fp_lesions'] = fp_lesions
    results['_n_gt_lesions'] = n_gt
    results['_n_pred_lesions'] = n_pred

    return results


def run_comparison(model_dict, gt_dir, subset_name):
    """Run full comparison for a set of models against a GT directory."""
    print(f"\n{'='*90}")
    print(f"  DEEP MODEL COMPARISON — {subset_name} ({len(os.listdir(gt_dir))} test cases)")
    print(f"{'='*90}")

    all_results = {}

    for model_key, (pred_dir_name, description) in model_dict.items():
        pred_dir = os.path.join(PRED_BASE, pred_dir_name)
        if not os.path.isdir(pred_dir):
            print(f"  SKIP {model_key}: directory not found")
            continue

        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz")))
        if not gt_files:
            print(f"  SKIP {subset_name}: no GT files")
            continue

        case_metrics = []
        lesion_stats_agg = {b[0]: {'total': 0, 'detected': 0} for b in SIZE_BINS}
        total_fp_lesions = 0
        total_gt_lesions = 0
        total_pred_lesions = 0

        for gf in gt_files:
            name = os.path.basename(gf)
            pf = os.path.join(pred_dir, name)
            if not os.path.exists(pf):
                continue

            gt_img = sitk.ReadImage(gf)
            pred_img = sitk.ReadImage(pf)
            spacing = gt_img.GetSpacing()[::-1]  # sitk is x,y,z; numpy is z,y,x

            gt_arr = sitk.GetArrayFromImage(gt_img)
            pred_arr = sitk.GetArrayFromImage(pred_img)

            # Binarize (handle multi-label)
            gt_bin = (gt_arr > 0).astype(np.uint8)
            pred_bin = (pred_arr > 0).astype(np.uint8)

            m = compute_metrics(pred_bin, gt_bin, spacing)
            m['case'] = name.replace('.nii.gz', '')
            case_metrics.append(m)

            # Lesion-level analysis
            ls = analyze_lesion_detection(pred_bin, gt_bin, SIZE_BINS)
            for bname in [b[0] for b in SIZE_BINS]:
                lesion_stats_agg[bname]['total'] += ls[bname]['total']
                lesion_stats_agg[bname]['detected'] += ls[bname]['detected']
            total_fp_lesions += ls['_fp_lesions']
            total_gt_lesions += ls['_n_gt_lesions']
            total_pred_lesions += ls['_n_pred_lesions']

        if not case_metrics:
            print(f"  SKIP {model_key}: no matching predictions")
            continue

        all_results[model_key] = {
            'description': description,
            'cases': case_metrics,
            'lesion_stats': lesion_stats_agg,
            'fp_lesions': total_fp_lesions,
            'gt_lesions': total_gt_lesions,
            'pred_lesions': total_pred_lesions,
        }

        n = len(case_metrics)
        sys.stdout.write(f"  Computed {model_key} ({n} cases)\n")
        sys.stdout.flush()

    return all_results


def print_summary_table(all_results, subset_name):
    """Print formatted summary comparison table."""
    if not all_results:
        print("  No results to display.")
        return

    print(f"\n{'='*120}")
    print(f"  SUMMARY TABLE — {subset_name}")
    print(f"{'='*120}")

    # Header
    header = f"{'Model':<45} {'N':>3} {'Dice':>7} {'HD95':>8} {'AVD':>7} {'Recall':>7} {'Prec':>7} {'F1':>7} {'VolR':>6}"
    print(header)
    print("-" * 120)

    rows = []
    for model_key, data in all_results.items():
        cases = data['cases']
        n = len(cases)
        dice = np.mean([c['dice'] for c in cases])
        # For HD95, handle inf
        hd95_vals = [c['hd95'] for c in cases if c['hd95'] != float('inf')]
        hd95 = np.mean(hd95_vals) if hd95_vals else float('inf')
        avd = np.mean([c['avd'] for c in cases if c['avd'] != float('inf')])
        recall = np.mean([c['recall'] for c in cases])
        precision = np.mean([c['precision'] for c in cases])
        f1 = np.mean([c['f1'] for c in cases])
        vol_ratio = np.mean([c['vol_ratio'] for c in cases if c['vol_ratio'] != float('inf')])

        hd95_str = f"{hd95:>8.2f}" if hd95 != float('inf') else "     inf"
        rows.append((dice, model_key, n, hd95, avd, recall, precision, f1, vol_ratio, data['description']))

    # Sort by Dice descending
    rows.sort(key=lambda x: -x[0])

    for dice, model_key, n, hd95, avd, recall, precision, f1, vol_ratio, desc in rows:
        hd95_str = f"{hd95:>8.2f}" if hd95 != float('inf') else "     inf"
        print(f"{model_key:<45} {n:>3} {dice:>7.4f} {hd95_str} {avd:>7.3f} {recall:>7.4f} {precision:>7.4f} {f1:>7.4f} {vol_ratio:>6.2f}")

    print()

    # Lesion detection table
    print(f"\n{'='*120}")
    print(f"  LESION DETECTION BY SIZE — {subset_name}")
    print(f"{'='*120}")

    # Header
    size_header = f"{'Model':<40}"
    for bname, _, _ in SIZE_BINS:
        size_header += f" {bname:>12}"
    size_header += f" {'FP_les':>8} {'GT_les':>8}"
    print(size_header)
    print("-" * 120)

    for dice, model_key, n, hd95, avd, recall, precision, f1, vol_ratio, desc in rows:
        data = all_results[model_key]
        ls = data['lesion_stats']
        line = f"{model_key:<40}"
        for bname, _, _ in SIZE_BINS:
            total = ls[bname]['total']
            det = ls[bname]['detected']
            if total > 0:
                pct = 100 * det / total
                line += f" {det:>4}/{total:<4} {pct:>4.0f}%"
            else:
                line += f" {'---':>12}"
        line += f" {data['fp_lesions']:>8} {data['gt_lesions']:>8}"
        print(line)

    print()

    # Detailed per-metric ranking
    print(f"\n{'='*120}")
    print(f"  PER-METRIC RANKINGS — {subset_name}")
    print(f"{'='*120}")

    metrics_to_rank = [
        ('dice', 'Dice (higher=better)', True),
        ('hd95', 'HD95 mm (lower=better)', False),
        ('recall', 'Recall (higher=better)', True),
        ('precision', 'Precision (higher=better)', True),
        ('avd', 'AVD (lower=better)', False),
    ]

    for metric, label, higher_better in metrics_to_rank:
        print(f"\n  {label}:")
        metric_vals = []
        for model_key, data in all_results.items():
            vals = [c[metric] for c in data['cases'] if c[metric] != float('inf')]
            if vals:
                metric_vals.append((np.mean(vals), np.std(vals), model_key))

        metric_vals.sort(key=lambda x: -x[0] if higher_better else x[0])
        for rank, (mean, std, model_key) in enumerate(metric_vals, 1):
            marker = " *** BEST" if rank == 1 else ""
            print(f"    #{rank}: {model_key:<45} {mean:.4f} ± {std:.4f}{marker}")


def save_per_case_csv(all_results, subset_name, output_dir):
    """Save per-case metrics to CSV for external analysis."""
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"deep_comparison_{subset_name}.csv")

    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'case', 'dice', 'hd95', 'avd', 'recall', 'precision',
                         'f1', 'gt_vol', 'pred_vol', 'vol_ratio', 'tp', 'fp', 'fn'])
        for model_key, data in all_results.items():
            for c in data['cases']:
                writer.writerow([
                    model_key, c['case'], f"{c['dice']:.6f}",
                    f"{c['hd95']:.4f}" if c['hd95'] != float('inf') else 'inf',
                    f"{c['avd']:.6f}" if c['avd'] != float('inf') else 'inf',
                    f"{c['recall']:.6f}", f"{c['precision']:.6f}", f"{c['f1']:.6f}",
                    c['gt_vol'], c['pred_vol'],
                    f"{c['vol_ratio']:.6f}" if c['vol_ratio'] != float('inf') else 'inf',
                    c['tp'], c['fp'], c['fn'],
                ])
    print(f"\n  Per-case CSV saved: {fname}")
    return fname


if __name__ == '__main__':
    output_dir = os.path.join(BASE, "results", "analysis")

    # === MSLesSeg-22 comparison ===
    print("\n" + "#"*90)
    print("  ANALYZING MSLesSeg-22 TEST SET")
    print("#"*90)
    msl_results = run_comparison(MSL_MODELS, GT_MSL, "MSLesSeg-22")
    print_summary_table(msl_results, "MSLesSeg-22")
    save_per_case_csv(msl_results, "MSLesSeg22", output_dir)

    # === WMH-110 comparison ===
    print("\n" + "#"*90)
    print("  ANALYZING WMH-110 TEST SET")
    print("#"*90)
    wmh_results = run_comparison(WMH_MODELS, GT_WMH, "WMH-110")
    print_summary_table(wmh_results, "WMH-110")
    save_per_case_csv(wmh_results, "WMH110", output_dir)

    print("\n" + "="*90)
    print("  ANALYSIS COMPLETE")
    print("="*90)
