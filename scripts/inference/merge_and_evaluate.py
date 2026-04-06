#!/usr/bin/env python3
"""Merge multi-size lesion predictions to binary and evaluate against GT."""
import argparse
import glob
import os

import nibabel as nib
import numpy as np


def merge_to_binary(input_dir, output_dir):
    """Merge labels 1,2,3 -> 1 (lesion) for all NIfTI files."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    print(f"Merging {len(files)} files to binary in {output_dir}")
    for f in files:
        img = nib.load(f)
        data = img.get_fdata().astype(np.uint8)
        binary = (data > 0).astype(np.uint8)
        out_img = nib.Nifti1Image(binary, img.affine, img.header)
        nib.save(out_img, os.path.join(output_dir, os.path.basename(f)))
    return output_dir


def evaluate(pred_dir, gt_dir):
    """Compute Dice, Precision, Recall for binary predictions vs GT."""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    print(f"\nEvaluating {len(pred_files)} predictions against GT in {gt_dir}")
    print()

    dice_scores = []
    precision_scores = []
    recall_scores = []
    case_ids = []

    for pf in pred_files:
        case_id = os.path.basename(pf)
        gf = os.path.join(gt_dir, case_id)
        if not os.path.exists(gf):
            print(f"WARNING: No GT for {case_id}, skipping")
            continue

        pred = nib.load(pf).get_fdata().astype(bool)
        gt = nib.load(gf).get_fdata().astype(bool)

        tp = float(np.sum(pred & gt))
        fp = float(np.sum(pred & ~gt))
        fn = float(np.sum(~pred & gt))

        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        dice_scores.append(dice)
        precision_scores.append(precision)
        recall_scores.append(recall)
        case_ids.append(case_id)

        print(f"{case_id}: Dice={dice:.4f}  Prec={precision:.4f}  Recall={recall:.4f}  "
              f"TP={int(tp)} FP={int(fp)} FN={int(fn)}")

    print()
    print("===== SUMMARY =====")
    print(f"Cases evaluated: {len(dice_scores)}")
    print(f"Mean Dice:      {np.mean(dice_scores):.4f} +/- {np.std(dice_scores):.4f}")
    print(f"Median Dice:    {np.median(dice_scores):.4f}")
    print(f"Mean Precision: {np.mean(precision_scores):.4f} +/- {np.std(precision_scores):.4f}")
    print(f"Mean Recall:    {np.mean(recall_scores):.4f} +/- {np.std(recall_scores):.4f}")
    print(f"Min Dice:       {np.min(dice_scores):.4f}")
    print(f"Max Dice:       {np.max(dice_scores):.4f}")

    # Per-dataset breakdown
    msl_idx = [i for i, c in enumerate(case_ids) if "MSL" in c]
    wmh_idx = [i for i, c in enumerate(case_ids) if "WMH" in c]
    if msl_idx:
        msl_dice = [dice_scores[i] for i in msl_idx]
        print(f"\nMSL subset ({len(msl_dice)} cases): "
              f"Mean Dice={np.mean(msl_dice):.4f} +/- {np.std(msl_dice):.4f}")
    if wmh_idx:
        wmh_dice = [dice_scores[i] for i in wmh_idx]
        print(f"WMH subset ({len(wmh_dice)} cases): "
              f"Mean Dice={np.mean(wmh_dice):.4f} +/- {np.std(wmh_dice):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Multi-size prediction dir")
    parser.add_argument("--gt_dir", required=True, help="Multi-size GT dir")
    args = parser.parse_args()

    base = os.path.dirname(args.pred_dir)
    pred_binary_dir = os.path.join(base, "DS004_5fold_binary")
    gt_binary_dir = os.path.join(base, "DS004_gt_binary")

    print("========================================")
    print("  Merging predictions to binary")
    print("========================================")
    merge_to_binary(args.pred_dir, pred_binary_dir)

    print("\n========================================")
    print("  Merging GT to binary")
    print("========================================")
    merge_to_binary(args.gt_dir, gt_binary_dir)

    print("\n========================================")
    print("  Evaluation: Binary Dice")
    print("========================================")
    evaluate(pred_binary_dir, gt_binary_dir)


if __name__ == "__main__":
    main()
