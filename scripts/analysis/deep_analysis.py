"""Deep per-class, per-case analysis of all models on MSLesSeg test set."""
import os, glob
from pathlib import Path
import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

gt_binary_dir = str(REPO_ROOT / 'data' / 'nnUNet_raw' / 'Dataset001_MSLesSeg' / 'labelsTs')
gt_multisize_dir = str(REPO_ROOT / 'data' / 'nnUNet_raw' / 'Dataset005_MSLesSeg_MultiSize' / 'labelsTs')
pred_base = str(REPO_ROOT / 'results' / 'predictions')

def dice(p, g):
    tp = float(np.sum(p & g)); fp = float(np.sum(p & ~g)); fn = float(np.sum(~p & g))
    return 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 1.0

def precision_fn(p, g):
    tp = float(np.sum(p & g)); fp = float(np.sum(p & ~g))
    return tp / (tp + fp) if (tp + fp) > 0 else 1.0

def recall_fn(p, g):
    tp = float(np.sum(p & g)); fn = float(np.sum(~p & g))
    return tp / (tp + fn) if (tp + fn) > 0 else 1.0

models = {
    'DS007_5fold':      ('DS007_5fold_multisize', True),
    'DS007_5fold_TTA':  ('DS007_5fold_TTA_multisize', True),
    'CNN_3D_TTA':       ('DS001_DS003_CNN_3D_TTA', False),
    'ResEncL_3D_TTA':   ('DS001_DS003_ResEncL_3D_TTA', False),
    'Ens_best':         ('DS001_ensemble', False),
    'ResEncL25D_TTA':   ('DS001_DS003_ResEncL_25D_TTA_chfix', False),
}

gt_files = sorted(glob.glob(os.path.join(gt_binary_dir, '*.nii.gz')))
results = {}

for case_path in gt_files:
    case = os.path.basename(case_path)
    gt_bin = nib.load(case_path).get_fdata().astype(bool)
    gt_ms = nib.load(os.path.join(gt_multisize_dir, case)).get_fdata().astype(np.uint8)
    
    n_small = int(np.sum(gt_ms == 1))
    n_medium = int(np.sum(gt_ms == 2))
    n_large = int(np.sum(gt_ms == 3))
    n_total = int(np.sum(gt_bin))
    
    cr = {'n_small': n_small, 'n_medium': n_medium, 'n_large': n_large, 'n_total': n_total}
    
    for mname, (dirname, is_multisize) in models.items():
        pf = os.path.join(pred_base, dirname, case)
        if not os.path.exists(pf):
            cr[mname + '_dice'] = -1
            continue
        pred = nib.load(pf).get_fdata()
        pred_bin = (pred > 0).astype(bool) if is_multisize else pred.astype(bool)
        
        cr[mname + '_dice'] = dice(pred_bin, gt_bin)
        cr[mname + '_prec'] = precision_fn(pred_bin, gt_bin)
        cr[mname + '_rec'] = recall_fn(pred_bin, gt_bin)
        
        # Per-class recall (binary prediction vs multi-size GT)
        for c, label in [(1, 'sm'), (2, 'md'), (3, 'lg')]:
            gt_c = (gt_ms == c)
            ng = float(np.sum(gt_c))
            if ng == 0:
                cr[f'{mname}_{label}_rec'] = -1
            else:
                cr[f'{mname}_{label}_rec'] = float(np.sum(pred_bin & gt_c)) / ng
    
    results[case] = cr

# ========================================================================
# PART A: Per-case comparison table
# ========================================================================
print('=' * 130)
print('PER-CASE DICE COMPARISON (sorted by DS007 Dice, ascending — worst first)')
print('=' * 130)
header = f"{'Case':>25s}  {'Total':>6s} {'Small':>5s} {'Med':>5s} {'Large':>5s} | {'DS007':>7s} {'CNN3D':>7s} {'Ens':>7s} {'R25D':>7s} | {'D-CNN':>6s} {'D-Ens':>6s}"
print(header)
print('-' * 130)

sorted_cases = sorted(results.keys(), key=lambda x: results[x].get('DS007_5fold_dice', 0))
for case in sorted_cases:
    r = results[case]
    ds007 = r.get('DS007_5fold_dice', -1)
    cnn = r.get('CNN_3D_TTA_dice', -1)
    ens = r.get('Ens_best_dice', -1)
    r25d = r.get('ResEncL25D_TTA_dice', -1)
    d_cnn = ds007 - cnn if (ds007 >= 0 and cnn >= 0) else 0
    d_ens = ds007 - ens if (ds007 >= 0 and ens >= 0) else 0
    
    print(f"{case:>25s}  {r['n_total']:>6d} {r['n_small']:>5d} {r['n_medium']:>5d} {r['n_large']:>5d} | "
          f"{ds007:>7.4f} {cnn:>7.4f} {ens:>7.4f} {r25d:>7.4f} | "
          f"{d_cnn:>+6.3f} {d_ens:>+6.3f}")

# ========================================================================
# PART B: Per-class recall comparison
# ========================================================================
print()
print('=' * 130)
print('PER-CLASS RECALL COMPARISON (averaged across cases)')
print('What fraction of ground truth voxels (by size class) does each model detect?')
print('=' * 130)
print()

for mname in ['DS007_5fold', 'DS007_5fold_TTA', 'CNN_3D_TTA', 'ResEncL_3D_TTA', 'Ens_best', 'ResEncL25D_TTA']:
    sm_list, md_list, lg_list = [], [], []
    dice_list = []
    for case in results:
        r = results[case]
        d = r.get(f'{mname}_dice', -1)
        if d >= 0: dice_list.append(d)
        for c, label, lst in [(1, 'sm', sm_list), (2, 'md', md_list), (3, 'lg', lg_list)]:
            v = r.get(f'{mname}_{label}_rec', -1)
            if v >= 0: lst.append(v)
    
    if not dice_list:
        continue
    
    print(f"  {mname:20s}  Dice={np.mean(dice_list):.4f}  "
          f"Small_Recall={np.mean(sm_list):.4f}  Med_Recall={np.mean(md_list):.4f}  Large_Recall={np.mean(lg_list):.4f}")

# ========================================================================
# PART C: Confusion analysis — what does DS007 predict for GT voxels?
# ========================================================================
print()
print('=' * 130)
print('CONFUSION ANALYSIS: What does DS007_5fold predict for each GT size class?')
print('For GT voxels of class C, what fraction lands in each predicted class?')
print('=' * 130)
print()

conf_totals = np.zeros((4, 4))  # gt_class x pred_class

for case in sorted(results.keys()):
    pf = os.path.join(pred_base, 'DS007_5fold_multisize', case)
    if not os.path.exists(pf):
        continue
    pred = nib.load(pf).get_fdata().astype(np.uint8)
    gt_ms = nib.load(os.path.join(gt_multisize_dir, case)).get_fdata().astype(np.uint8)
    
    for gt_c in range(4):
        mask = (gt_ms == gt_c)
        if np.sum(mask) == 0:
            continue
        for pred_c in range(4):
            conf_totals[gt_c, pred_c] += float(np.sum(mask & (pred == pred_c)))

# Normalize rows
gt_pred_label = 'GT \\ Pred'
print(f"{gt_pred_label:>12s}  {'BG':>8s} {'Small':>8s} {'Medium':>8s} {'Large':>8s}  |  {'Total vx':>10s}")
print('-' * 70)
gt_labels = ['BG', 'Small', 'Medium', 'Large']
for gt_c in range(4):
    row_total = conf_totals[gt_c].sum()
    if row_total == 0:
        continue
    row_pct = conf_totals[gt_c] / row_total * 100
    print(f"{gt_labels[gt_c]:>12s}  {row_pct[0]:>7.2f}% {row_pct[1]:>7.2f}% {row_pct[2]:>7.2f}% {row_pct[3]:>7.2f}%  |  {row_total:>10.0f}")

# ========================================================================
# PART D: Win/loss summary
# ========================================================================
print()
print('=' * 130)
print('WIN/LOSS SUMMARY')
print('=' * 130)

ds007_arr = np.array([results[c]['DS007_5fold_dice'] for c in sorted_cases])
cnn_arr = np.array([results[c]['CNN_3D_TTA_dice'] for c in sorted_cases])
ens_arr = np.array([results[c]['Ens_best_dice'] for c in sorted_cases])
r3d_arr = np.array([results[c]['ResEncL_3D_TTA_dice'] for c in sorted_cases])

for comp_name, comp_arr in [('CNN_3D_TTA', cnn_arr), ('Ens_best', ens_arr), ('ResEncL_3D_TTA', r3d_arr)]:
    wins = int(np.sum(ds007_arr > comp_arr))
    ties = int(np.sum(ds007_arr == comp_arr))
    losses = int(np.sum(ds007_arr < comp_arr))
    avg_delta = np.mean(ds007_arr - comp_arr)
    print(f"  DS007_5fold vs {comp_name:20s}: W={wins} L={losses} T={ties}  Avg delta={avg_delta:+.4f}")

# ========================================================================
# PART E: Lesion-level detection (connected components)
# ========================================================================
print()
print('=' * 130)
print('LESION-LEVEL DETECTION RATE (by size class)')
print('A lesion counts as detected if >50% of its voxels are predicted as foreground')
print('=' * 130)
print()

from scipy.ndimage import label as cc_label

for mname, (dirname, is_multisize) in [
    ('DS007_5fold', ('DS007_5fold_multisize', True)),
    ('CNN_3D_TTA', ('DS001_DS003_CNN_3D_TTA', False)),
    ('Ens_best', ('DS001_ensemble', False)),
]:
    detected = {1: 0, 2: 0, 3: 0}
    total_lesions = {1: 0, 2: 0, 3: 0}
    
    for case in sorted(results.keys()):
        pf = os.path.join(pred_base, dirname, case)
        if not os.path.exists(pf):
            continue
        pred_raw = nib.load(pf).get_fdata()
        pred_bin = (pred_raw > 0).astype(bool) if is_multisize else pred_raw.astype(bool)
        gt_bin = nib.load(os.path.join(gt_binary_dir, case)).get_fdata().astype(bool)
        gt_ms = nib.load(os.path.join(gt_multisize_dir, case)).get_fdata().astype(np.uint8)
        
        # Get connected components from binary GT
        labeled, n_components = cc_label(gt_bin)
        
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            comp_voxels = np.sum(comp_mask)
            
            # Determine size class from multi-size GT (majority label in component)
            ms_vals = gt_ms[comp_mask]
            size_class = int(np.median(ms_vals))  # Should be 1, 2, or 3
            if size_class not in [1, 2, 3]:
                continue
            
            total_lesions[size_class] += 1
            
            # Check if >50% of component is predicted as foreground
            overlap = np.sum(pred_bin & comp_mask)
            if overlap / comp_voxels > 0.5:
                detected[size_class] += 1
    
    print(f"  {mname:20s}:")
    for c, label in [(1, 'Small'), (2, 'Medium'), (3, 'Large')]:
        det_rate = detected[c] / total_lesions[c] * 100 if total_lesions[c] > 0 else 0
        print(f"    {label:8s}: {detected[c]:>4d} / {total_lesions[c]:>4d} detected ({det_rate:.1f}%)")
    total_det = sum(detected.values())
    total_les = sum(total_lesions.values())
    print(f"    {'Total':8s}: {total_det:>4d} / {total_les:>4d} detected ({total_det/total_les*100:.1f}%)")
    print()

# ========================================================================
# PART F: False positive analysis
# ========================================================================
print('=' * 130)
print('FALSE POSITIVE ANALYSIS (model predicts lesion where GT has no lesion)')
print('=' * 130)
print()

for mname, (dirname, is_multisize) in [
    ('DS007_5fold', ('DS007_5fold_multisize', True)),
    ('CNN_3D_TTA', ('DS001_DS003_CNN_3D_TTA', False)),
    ('Ens_best', ('DS001_ensemble', False)),
]:
    total_fp = 0
    total_tp = 0
    fp_components_total = 0
    
    for case in sorted(results.keys()):
        pf = os.path.join(pred_base, dirname, case)
        if not os.path.exists(pf):
            continue
        pred_raw = nib.load(pf).get_fdata()
        pred_bin = (pred_raw > 0).astype(bool) if is_multisize else pred_raw.astype(bool)
        gt_bin = nib.load(os.path.join(gt_binary_dir, case)).get_fdata().astype(bool)
        
        fp = pred_bin & ~gt_bin
        tp = pred_bin & gt_bin
        total_fp += int(np.sum(fp))
        total_tp += int(np.sum(tp))
        
        # Count FP connected components
        if np.sum(fp) > 0:
            _, n_fp = cc_label(fp)
            fp_components_total += n_fp
    
    fp_ratio = total_fp / (total_fp + total_tp) * 100 if (total_fp + total_tp) > 0 else 0
    print(f"  {mname:20s}: FP voxels={total_fp:>8d}  TP voxels={total_tp:>8d}  "
          f"FP ratio={fp_ratio:.1f}%  FP components={fp_components_total}")
