#!/usr/bin/env python3
"""
lst_ai_comparison.py — Compare LST-AI vs our models on MSLesSeg-22 test set.

Produces a per-case CSV and prints a summary table.

Usage:
    python scripts/evaluation/lst_ai_comparison.py
"""
import os
import glob
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label, generate_binary_structure
from scipy.ndimage import distance_transform_edt, binary_erosion
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = str(REPO_ROOT)
PRED_BASE = os.path.join(BASE, "results", "predictions")
GT_DIR = os.path.join(BASE, "data", "nnUNet_raw", "Dataset001_MSLesSeg", "labelsTs")
OUT_CSV = os.path.join(BASE, "results", "analysis", "lst_ai_comparison.csv")
OUT_SUMMARY = os.path.join(BASE, "results", "analysis", "lst_ai_comparison_summary.csv")

# Models to compare
MODELS = {
    "LST_AI":                      "LST-AI (TF ensemble, skull-stripped)",
    "DS001_DS003_CNN_3D_TTA":      "CNN 3D fold0 TTA (our best MSL)",
    "DS001_DS003_ResEncL_3D_TTA":  "ResEncL 3D fold0 TTA",
    "DS001_DS003_ResEncL_25D_TTA_chfix": "ResEncL 2.5D fold0 TTA",
    "DS007_5fold_TTA_binary":      "DS007 5-fold TTA (3ch multi-size)",
    "DS007_5fold_binary":          "DS007 5-fold (3ch multi-size)",
    "DS001_DS003_final":           "Best ensemble (CNN+RE3D+RE25D, thr=0.40)",
}


def compute_metrics(pred_arr, gt_arr, spacing=(1.0, 1.0, 1.0)):
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

    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if pred.sum() == 0 else 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt.sum() == 0 else 0.0)

    gt_vol = gt.sum()
    pred_vol = pred.sum()
    avd = abs(pred_vol - gt_vol) / gt_vol if gt_vol > 0 else (0.0 if pred_vol == 0 else float('inf'))

    hd95 = compute_hd95(pred, gt, spacing)

    return {
        'dice': dice, 'hd95': hd95, 'avd': avd,
        'recall': recall, 'precision': precision,
        'gt_vol': int(gt_vol), 'pred_vol': int(pred_vol),
    }


def compute_hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float('inf')

    pred_border = np.logical_xor(pred, binary_erosion(pred))
    gt_border = np.logical_xor(gt, binary_erosion(gt))

    if pred_border.sum() == 0:
        pred_border = pred
    if gt_border.sum() == 0:
        gt_border = gt

    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return np.percentile(all_dists, 95)


def count_lesions_detected(pred_arr, gt_arr, overlap_thr=0.1):
    """Count GT lesions detected (>overlap_thr overlap with pred)."""
    gt = gt_arr > 0
    pred = pred_arr > 0
    s = generate_binary_structure(3, 2)
    gt_labels, n_gt = cc_label(gt, structure=s)

    detected = 0
    for i in range(1, n_gt + 1):
        lesion = gt_labels == i
        overlap = np.logical_and(lesion, pred).sum()
        if overlap / lesion.sum() >= overlap_thr:
            detected += 1

    return detected, n_gt


def main():
    # Find all GT files
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "MSL_*.nii.gz")))
    subjects = [os.path.basename(f).replace('.nii.gz', '') for f in gt_files]
    print(f"Found {len(subjects)} test subjects")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    all_results = {}
    rows = []

    for model_dir, desc in MODELS.items():
        pred_dir = os.path.join(PRED_BASE, model_dir)
        if not os.path.isdir(pred_dir):
            print(f"  [SKIP] {model_dir} - directory not found")
            continue

        metrics_list = []
        total_detected = 0
        total_lesions = 0

        for subj in subjects:
            gt_path = os.path.join(GT_DIR, f"{subj}.nii.gz")
            pred_path = os.path.join(pred_dir, f"{subj}.nii.gz")

            if not os.path.exists(pred_path):
                print(f"  [WARN] {model_dir}/{subj} - prediction not found")
                continue

            gt_img = sitk.ReadImage(gt_path)
            pred_img = sitk.ReadImage(pred_path)
            gt_arr = sitk.GetArrayFromImage(gt_img)
            pred_arr = sitk.GetArrayFromImage(pred_img)
            spacing = gt_img.GetSpacing()

            m = compute_metrics(pred_arr, gt_arr, spacing)
            det, n_gt = count_lesions_detected(pred_arr, gt_arr)
            m['detected'] = det
            m['total_lesions'] = n_gt
            m['subject'] = subj
            m['model'] = model_dir
            metrics_list.append(m)
            total_detected += det
            total_lesions += n_gt
            rows.append(m)

        if metrics_list:
            dices = [m['dice'] for m in metrics_list]
            recalls = [m['recall'] for m in metrics_list]
            precisions = [m['precision'] for m in metrics_list]
            hd95s = [m['hd95'] for m in metrics_list if m['hd95'] != float('inf')]
            det_rate = total_detected / total_lesions if total_lesions > 0 else 0

            all_results[model_dir] = {
                'desc': desc,
                'dice_mean': np.mean(dices),
                'dice_std': np.std(dices),
                'recall_mean': np.mean(recalls),
                'precision_mean': np.mean(precisions),
                'hd95_mean': np.mean(hd95s) if hd95s else float('inf'),
                'detection_rate': det_rate,
                'n_cases': len(metrics_list),
                'total_detected': total_detected,
                'total_lesions': total_lesions,
            }

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Model':<45} {'Dice':>8} {'±std':>7} {'Recall':>8} {'Prec':>8} {'HD95':>8} {'Det%':>7} {'Det/Tot':>10}")
    print("-" * 120)
    for model_dir in MODELS:
        if model_dir not in all_results:
            continue
        r = all_results[model_dir]
        print(f"{r['desc']:<45} {r['dice_mean']:>8.4f} {r['dice_std']:>7.4f} "
              f"{r['recall_mean']:>8.4f} {r['precision_mean']:>8.4f} "
              f"{r['hd95_mean']:>8.2f} {r['detection_rate']:>6.1%} "
              f"{r['total_detected']:>4}/{r['total_lesions']:<4}")
    print("=" * 120)

    # Write per-case CSV
    fieldnames = ['model', 'subject', 'dice', 'recall', 'precision', 'hd95', 'avd',
                  'gt_vol', 'pred_vol', 'detected', 'total_lesions']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-case CSV: {OUT_CSV}")

    # Write summary CSV
    summary_fields = ['model', 'description', 'dice_mean', 'dice_std', 'recall_mean',
                      'precision_mean', 'hd95_mean', 'detection_rate', 'n_cases']
    with open(OUT_SUMMARY, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for model_dir in MODELS:
            if model_dir not in all_results:
                continue
            r = all_results[model_dir]
            w.writerow({
                'model': model_dir,
                'description': r['desc'],
                'dice_mean': f"{r['dice_mean']:.4f}",
                'dice_std': f"{r['dice_std']:.4f}",
                'recall_mean': f"{r['recall_mean']:.4f}",
                'precision_mean': f"{r['precision_mean']:.4f}",
                'hd95_mean': f"{r['hd95_mean']:.2f}",
                'detection_rate': f"{r['detection_rate']:.4f}",
                'n_cases': r['n_cases'],
            })
    print(f"Summary CSV:  {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
