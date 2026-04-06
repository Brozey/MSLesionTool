#!/usr/bin/env python3
"""
Analyze prediction performance by lesion size.
Uses connected components to identify individual lesions in GT,
then measures detection rate and Dice per size category.
"""
import os
import sys
import glob
import numpy as np
import SimpleITK as sitk
from collections import defaultdict

def analyze_by_lesion_size(pred_dir, gt_dir, subset_prefix=None):
    """Analyze predictions broken down by individual lesion size."""
    
    preds = sorted(glob.glob(os.path.join(pred_dir, '*.nii.gz')))
    if not preds:
        print(f"No predictions found in {pred_dir}")
        return
    
    # Size bins (in voxels, at 1mm isotropic = mm³)
    bins = [
        ('tiny',    0,      10),
        ('small',   10,     100),
        ('medium',  100,    1000),
        ('large',   1000,   10000),
        ('huge',    10000,  float('inf')),
    ]
    
    # Per-bin stats
    stats = {b[0]: {'total': 0, 'detected': 0, 'dice_sum': 0.0, 'sizes': []} for b in bins}
    
    # Per-case stats
    case_results = []
    
    for pf in preds:
        name = os.path.basename(pf)
        case_id = name.replace('.nii.gz', '')
        
        if subset_prefix and not case_id.startswith(subset_prefix):
            continue
            
        gf = os.path.join(gt_dir, name)
        if not os.path.exists(gf):
            continue
        
        pred_img = sitk.ReadImage(pf)
        gt_img = sitk.ReadImage(gf)
        
        pred = sitk.GetArrayFromImage(pred_img) > 0
        gt = sitk.GetArrayFromImage(gt_img) > 0
        
        # Connected components on GT
        gt_sitk = sitk.GetImageFromArray(gt.astype(np.uint8))
        cc = sitk.ConnectedComponent(gt_sitk)
        cc_arr = sitk.GetArrayFromImage(cc)
        
        n_lesions = cc_arr.max()
        
        case_info = {
            'case': case_id,
            'n_lesions': n_lesions,
            'total_gt_vol': int(gt.sum()),
            'total_pred_vol': int(pred.sum()),
        }
        
        for lid in range(1, n_lesions + 1):
            lesion_mask = cc_arr == lid
            lesion_size = int(lesion_mask.sum())
            
            # Find which bin
            bin_name = None
            for bname, bmin, bmax in bins:
                if bmin <= lesion_size < bmax:
                    bin_name = bname
                    break
            
            if bin_name is None:
                continue
            
            stats[bin_name]['total'] += 1
            stats[bin_name]['sizes'].append(lesion_size)
            
            # Check if detected: >50% overlap with any prediction
            overlap = np.logical_and(lesion_mask, pred)
            overlap_ratio = overlap.sum() / lesion_size
            detected = overlap_ratio > 0.1  # at least 10% overlap = detected
            
            if detected:
                stats[bin_name]['detected'] += 1
            
            # Lesion-level Dice
            pred_in_region = pred[lesion_mask]
            # Expand region slightly for pred-side measurement
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(lesion_mask, iterations=2)
            pred_near = np.logical_and(pred, dilated)
            gt_near = np.logical_and(gt, dilated)
            
            intersection = np.logical_and(pred_near, gt_near).sum()
            if gt_near.sum() + pred_near.sum() > 0:
                local_dice = 2 * intersection / (gt_near.sum() + pred_near.sum())
            else:
                local_dice = 1.0 if gt_near.sum() == 0 else 0.0
            
            stats[bin_name]['dice_sum'] += local_dice
        
        case_results.append(case_info)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Lesion Size Analysis: {pred_dir}")
    if subset_prefix:
        print(f"Subset: {subset_prefix}*")
    print(f"{'='*70}")
    print(f"{'Size':>10} {'Range (vox)':>15} {'Total':>8} {'Detected':>10} {'Det%':>8} {'Avg Dice':>10} {'Avg Vol':>10}")
    print(f"{'-'*10} {'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    
    total_all = 0
    detected_all = 0
    
    for bname, bmin, bmax in bins:
        s = stats[bname]
        total_all += s['total']
        detected_all += s['detected']
        
        if s['total'] == 0:
            print(f"{bname:>10} {bmin:>6}-{('inf' if bmax==float('inf') else str(int(bmax))):>7} {0:>8} {'-':>10} {'-':>8} {'-':>10} {'-':>10}")
            continue
        
        det_pct = 100 * s['detected'] / s['total']
        avg_dice = s['dice_sum'] / s['total']
        avg_vol = np.mean(s['sizes'])
        
        bmax_str = 'inf' if bmax == float('inf') else str(int(bmax))
        print(f"{bname:>10} {bmin:>6}-{bmax_str:>7} {s['total']:>8} {s['detected']:>10} {det_pct:>7.1f}% {avg_dice:>10.4f} {avg_vol:>10.1f}")
    
    print(f"{'-'*10} {'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    if total_all > 0:
        print(f"{'TOTAL':>10} {'':>15} {total_all:>8} {detected_all:>10} {100*detected_all/total_all:>7.1f}%")
    
    # Case-level summary
    print(f"\nCases analyzed: {len(case_results)}")
    vols = [c['total_gt_vol'] for c in case_results]
    n_les = [c['n_lesions'] for c in case_results]
    print(f"Lesions per case: {np.mean(n_les):.1f} ± {np.std(n_les):.1f} (range {np.min(n_les)}-{np.max(n_les)})")
    print(f"GT volume per case: {np.mean(vols):.0f} ± {np.std(vols):.0f} voxels")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--subset', default=None, help='Filter cases by prefix (e.g. MSL, WMH)')
    args = parser.parse_args()
    
    analyze_by_lesion_size(args.pred_dir, args.gt_dir, args.subset)
