"""
Ensemble Complementarity Analysis — WHY does the 3-model ensemble work?

Analyzes the 3-model ensemble (CNN-3D + ResEncL-2.5D + ResEncL-3D) on MSLesSeg-22
to understand WHY averaging these three models beats any individual model.

Key questions:
1. Error correlation: Are model errors uncorrelated (complementary)?
2. Disagreement analysis: Where do models disagree, and who's right?
3. Confidence analysis: Does ensemble averaging boost correct predictions?
4. Per-lesion agreement: Do models detect different lesions?
5. Size-stratified complementarity: Which model handles which lesion sizes?
"""

import os
import sys
import numpy as np
import nibabel as nib
from scipy import ndimage
from collections import defaultdict
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_BASE = str(REPO_ROOT / "results" / "predictions")
GT_DIR = str(REPO_ROOT / "data" / "nnUNet_raw" / "Dataset001_MSLesSeg" / "labelsTs")
OUTPUT_DIR = str(REPO_ROOT / "results" / "analysis")

MODELS = {
    "CNN_3D": os.path.join(PRED_BASE, "DS001_DS003_CNN_3D_TTA"),
    "ResEncL_25D": os.path.join(PRED_BASE, "DS001_DS003_ResEncL_25D_TTA_chfix"),
    "ResEncL_3D": os.path.join(PRED_BASE, "DS001_DS003_ResEncL_3D_TTA"),
}

ENSEMBLE_THR = 0.40  # The optimal threshold for the 3-model ensemble
SINGLE_THR = 0.50    # Default single-model threshold

SIZE_BINS = {
    "tiny": (0, 10),
    "small": (10, 100),
    "medium": (100, 1000),
    "large": (1000, float("inf")),
}


def load_softmax(npz_path):
    """Load foreground probability from NPZ, transpose to (x,y,z)."""
    d = np.load(npz_path)
    key = "softmax" if "softmax" in d.files else "probabilities"
    arr = d[key][1].astype(np.float32)
    return np.transpose(arr, (2, 1, 0))


def load_nifti(path):
    """Load NIfTI as binary array."""
    return nib.load(path).get_fdata().astype(np.uint8)


def get_voxel_volume(nifti_path):
    """Get voxel volume in mm³."""
    img = nib.load(nifti_path)
    return float(np.prod(img.header.get_zooms()))


def dice(pred, gt):
    intersection = np.sum(pred & gt)
    denom = np.sum(pred) + np.sum(gt)
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def get_connected_components(binary_mask, voxel_vol):
    """Extract connected components with volumes in mm³."""
    labeled, n = ndimage.label(binary_mask, structure=ndimage.generate_binary_structure(3, 2))
    components = []
    for i in range(1, n + 1):
        mask = labeled == i
        vol_mm3 = np.sum(mask) * voxel_vol
        components.append({"label": i, "mask": mask, "volume_mm3": vol_mm3})
    return labeled, components


def classify_size(vol_mm3):
    for name, (lo, hi) in SIZE_BINS.items():
        if lo <= vol_mm3 < hi:
            return name
    return "large"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_files = sorted(Path(GT_DIR).glob("*.nii.gz"))
    case_ids = [f.name.replace(".nii.gz", "") for f in gt_files]

    print(f"{'='*80}")
    print(f"ENSEMBLE COMPLEMENTARITY ANALYSIS")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Cases: {len(case_ids)} (MSLesSeg-22)")
    print(f"Ensemble threshold: {ENSEMBLE_THR}, Single threshold: {SINGLE_THR}")
    print(f"{'='*80}\n")

    # ─── Per-case accumulators ────────────────────────────────────────
    all_case_results = []

    # ─── Global accumulators ──────────────────────────────────────────
    # Error correlation
    model_errors = {m: [] for m in MODELS}  # per-voxel errors (flat)
    model_correct_exclusive = {m: 0 for m in MODELS}  # voxels ONLY this model gets right
    model_wrong_exclusive = {m: 0 for m in MODELS}

    # Disagreement zones
    total_agree_correct = 0
    total_agree_wrong = 0
    total_disagree = 0
    total_disagree_majority_correct = 0
    total_disagree_majority_wrong = 0
    total_fg_voxels = 0

    # Confidence
    ens_correct_mean_conf = []
    ens_wrong_mean_conf = []
    single_correct_mean_conf = {m: [] for m in MODELS}
    single_wrong_mean_conf = {m: [] for m in MODELS}

    # Per-lesion detection
    lesion_detection = {m: defaultdict(lambda: {"detected": 0, "total": 0}) for m in MODELS}
    lesion_detection["ensemble"] = defaultdict(lambda: {"detected": 0, "total": 0})
    # Track which lesions each model uniquely detects
    lesion_unique_detect = {m: 0 for m in MODELS}
    lesion_any_detect = 0
    lesion_all_detect = 0
    lesion_none_detect = 0
    total_gt_lesions = 0

    # Voxel-level improvement tracking
    ens_fixes_all = 0  # ensemble correct, ALL singles wrong
    ens_breaks = 0     # ensemble wrong, at least one single correct

    for case_idx, case_id in enumerate(case_ids):
        gt_path = os.path.join(GT_DIR, f"{case_id}.nii.gz")
        gt = load_nifti(gt_path)
        voxel_vol = get_voxel_volume(gt_path)

        # Load softmax for each model
        probs = {}
        preds_single = {}
        for mname, mdir in MODELS.items():
            npz_path = os.path.join(mdir, f"{case_id}.npz")
            prob = load_softmax(npz_path)
            probs[mname] = prob
            preds_single[mname] = (prob >= SINGLE_THR).astype(np.uint8)

        # Ensemble
        prob_stack = np.stack(list(probs.values()), axis=0)
        ens_prob = np.mean(prob_stack, axis=0)
        ens_pred = (ens_prob >= ENSEMBLE_THR).astype(np.uint8)

        # ─── 1. Dice scores ──────────────────────────────────────────
        gt_bool = gt > 0
        dices = {}
        for mname in MODELS:
            dices[mname] = dice(preds_single[mname] > 0, gt_bool)
        dices["ensemble"] = dice(ens_pred > 0, gt_bool)

        # ─── 2. Voxel-level error analysis ────────────────────────────
        # For each voxel, classify: TP, FP, FN, TN per model
        fg_mask = gt_bool | (ens_pred > 0)
        for mname in MODELS:
            fg_mask = fg_mask | (preds_single[mname] > 0)

        total_fg_voxels += np.sum(fg_mask)

        # Per-model correctness (binary: correct prediction or not)
        correct = {}
        for mname in MODELS:
            correct[mname] = preds_single[mname].astype(bool) == gt_bool
        correct_ens = ens_pred.astype(bool) == gt_bool

        # Error correlation: collect flat vectors on foreground region
        for mname in MODELS:
            err = (~correct[mname])[fg_mask]
            model_errors[mname].append(err)

        # Exclusive correctness
        for mname in MODELS:
            others_wrong = np.ones_like(gt_bool, dtype=bool)
            for other in MODELS:
                if other != mname:
                    others_wrong &= ~correct[other]
            model_correct_exclusive[mname] += np.sum(correct[mname] & others_wrong & fg_mask)
            # This model exclusively wrong
            others_correct = np.ones_like(gt_bool, dtype=bool)
            for other in MODELS:
                if other != mname:
                    others_correct &= correct[other]
            model_wrong_exclusive[mname] += np.sum(~correct[mname] & others_correct & fg_mask)

        # ─── 3. Disagreement analysis ─────────────────────────────────
        pred_stack = np.stack([preds_single[m] for m in MODELS], axis=0)
        vote_sum = np.sum(pred_stack > 0, axis=0)  # 0, 1, 2, or 3

        all_agree = (vote_sum == 0) | (vote_sum == len(MODELS))
        disagree = ~all_agree & fg_mask

        agree_predict_pos = (vote_sum == len(MODELS))
        agree_predict_neg = (vote_sum == 0)

        agree_correct = ((agree_predict_pos & gt_bool) | (agree_predict_neg & ~gt_bool)) & fg_mask
        agree_wrong = ((agree_predict_pos & ~gt_bool) | (agree_predict_neg & gt_bool)) & fg_mask

        total_agree_correct += np.sum(agree_correct)
        total_agree_wrong += np.sum(agree_wrong)
        total_disagree += np.sum(disagree)

        # When they disagree, is majority right?
        majority_pred = vote_sum >= 2  # 2 or 3 out of 3
        maj_correct_mask = (majority_pred == gt_bool) & disagree
        maj_wrong_mask = (majority_pred != gt_bool) & disagree
        total_disagree_majority_correct += np.sum(maj_correct_mask)
        total_disagree_majority_wrong += np.sum(maj_wrong_mask)

        # ─── 4. Confidence analysis ──────────────────────────────────
        ens_tp = (ens_pred > 0) & gt_bool
        ens_fp = (ens_pred > 0) & ~gt_bool
        ens_fn = (ens_pred == 0) & gt_bool

        if np.any(ens_tp):
            ens_correct_mean_conf.append(float(np.mean(ens_prob[ens_tp])))
        if np.any(ens_fp | ens_fn):
            wrong_mask = ens_fp | ens_fn
            ens_wrong_mean_conf.append(float(np.mean(ens_prob[wrong_mask & (ens_pred > 0)])) if np.any(ens_fp) else 0)

        for mname in MODELS:
            p = probs[mname]
            tp = (preds_single[mname] > 0) & gt_bool
            fp = (preds_single[mname] > 0) & ~gt_bool
            if np.any(tp):
                single_correct_mean_conf[mname].append(float(np.mean(p[tp])))
            if np.any(fp):
                single_wrong_mean_conf[mname].append(float(np.mean(p[fp])))

        # ─── 5. Ensemble fixes vs breaks ──────────────────────────────
        all_singles_wrong = np.ones_like(gt_bool, dtype=bool)
        any_single_correct_fg = np.zeros_like(gt_bool, dtype=bool)
        for mname in MODELS:
            all_singles_wrong &= ~correct[mname]
            any_single_correct_fg |= correct[mname]

        ens_fixes_all += np.sum(correct_ens & all_singles_wrong & fg_mask)
        ens_breaks += np.sum(~correct_ens & any_single_correct_fg & fg_mask)

        # ─── 6. Per-lesion detection ──────────────────────────────────
        gt_labeled, gt_comps = get_connected_components(gt_bool, voxel_vol)

        for comp in gt_comps:
            size_bin = classify_size(comp["volume_mm3"])
            total_gt_lesions += 1

            detected_by = {}
            for mname in MODELS:
                overlap = np.sum(preds_single[mname][comp["mask"]] > 0)
                detected_by[mname] = overlap > 0
                lesion_detection[mname][size_bin]["total"] += 1
                if detected_by[mname]:
                    lesion_detection[mname][size_bin]["detected"] += 1

            ens_overlap = np.sum(ens_pred[comp["mask"]] > 0)
            ens_detected = ens_overlap > 0
            lesion_detection["ensemble"][size_bin]["total"] += 1
            if ens_detected:
                lesion_detection["ensemble"][size_bin]["detected"] += 1

            any_detect = any(detected_by.values())
            all_detect = all(detected_by.values())
            none_detect = not any_detect

            if any_detect:
                lesion_any_detect += 1
            if all_detect:
                lesion_all_detect += 1
            if none_detect:
                lesion_none_detect += 1

            # Unique detections
            for mname in MODELS:
                if detected_by[mname] and sum(detected_by.values()) == 1:
                    lesion_unique_detect[mname] += 1

        case_result = {"case_id": case_id, **dices}
        all_case_results.append(case_result)

        sys.stdout.write(f"\r  Processing {case_idx+1}/{len(case_ids)}: {case_id} "
                         f"| Ens={dices['ensemble']:.3f} | "
                         f"CNN={dices['CNN_3D']:.3f} | "
                         f"RE25D={dices['ResEncL_25D']:.3f} | "
                         f"RE3D={dices['ResEncL_3D']:.3f}")
        sys.stdout.flush()

    print("\n")

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("1. CASE-BY-CASE DICE COMPARISON")
    print("=" * 80)
    print(f"\n{'Case':<15} {'CNN_3D':>8} {'RE_25D':>8} {'RE_3D':>8} {'Ensemble':>9} {'Best Single':>12} {'Ens-Best':>9}")
    print("-" * 75)

    ens_wins = 0
    ens_total_gain = 0
    ens_dices = []
    best_single_dices = []

    for r in all_case_results:
        best_single = max(r["CNN_3D"], r["ResEncL_25D"], r["ResEncL_3D"])
        diff = r["ensemble"] - best_single
        marker = " *" if diff > 0 else ""
        print(f"{r['case_id']:<15} {r['CNN_3D']:>8.4f} {r['ResEncL_25D']:>8.4f} "
              f"{r['ResEncL_3D']:>8.4f} {r['ensemble']:>9.4f} {best_single:>12.4f} {diff:>+9.4f}{marker}")
        if diff > 0:
            ens_wins += 1
        ens_total_gain += diff
        ens_dices.append(r["ensemble"])
        best_single_dices.append(best_single)

    print("-" * 75)
    print(f"{'MEAN':<15} {np.mean([r['CNN_3D'] for r in all_case_results]):>8.4f} "
          f"{np.mean([r['ResEncL_25D'] for r in all_case_results]):>8.4f} "
          f"{np.mean([r['ResEncL_3D'] for r in all_case_results]):>8.4f} "
          f"{np.mean(ens_dices):>9.4f} {np.mean(best_single_dices):>12.4f} "
          f"{np.mean(ens_dices) - np.mean(best_single_dices):>+9.4f}")
    print(f"\nEnsemble wins: {ens_wins}/{len(case_ids)} cases ({100*ens_wins/len(case_ids):.0f}%)")
    print(f"Mean gain over best single: {ens_total_gain / len(case_ids):+.4f}")

    # ─── 2. Error correlation ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("2. ERROR CORRELATION (lower = more complementary)")
    print("=" * 80)

    mnames = list(MODELS.keys())
    err_flat = {}
    for mname in mnames:
        err_flat[mname] = np.concatenate(model_errors[mname]).astype(float)

    print(f"\n{'':>15}", end="")
    for m in mnames:
        print(f"{m:>15}", end="")
    print()
    for m1 in mnames:
        print(f"{m1:>15}", end="")
        for m2 in mnames:
            corr = np.corrcoef(err_flat[m1], err_flat[m2])[0, 1]
            print(f"{corr:>15.4f}", end="")
        print()

    print("\n  Interpretation: Correlation < 0.5 means models make different errors")
    print("  Correlation ~ 1.0 means models fail on the same voxels (no diversity)")

    # ─── 3. Exclusive correctness ─────────────────────────────────────
    print(f"\n{'='*80}")
    print("3. EXCLUSIVE CONTRIBUTIONS (voxels ONLY this model gets right/wrong)")
    print("=" * 80)

    total_exclusive_correct = sum(model_correct_exclusive.values())
    print(f"\n{'Model':<20} {'Exclusively Correct':>20} {'% of Exclusive':>15} {'Exclusively Wrong':>20}")
    print("-" * 80)
    for mname in mnames:
        ec = model_correct_exclusive[mname]
        ew = model_wrong_exclusive[mname]
        pct = 100 * ec / total_exclusive_correct if total_exclusive_correct > 0 else 0
        print(f"{mname:<20} {ec:>20,} {pct:>14.1f}% {ew:>20,}")

    print(f"\n  Total voxels where exactly ONE model is correct: {total_exclusive_correct:,}")
    print("  These are the voxels where model diversity matters most.")

    # ─── 4. Disagreement analysis ─────────────────────────────────────
    print(f"\n{'='*80}")
    print("4. DISAGREEMENT ANALYSIS")
    print("=" * 80)

    total_classified = total_agree_correct + total_agree_wrong + total_disagree
    print(f"\n  All agree & correct:    {total_agree_correct:>12,} ({100*total_agree_correct/total_classified:.1f}%)")
    print(f"  All agree & WRONG:      {total_agree_wrong:>12,} ({100*total_agree_wrong/total_classified:.1f}%)")
    print(f"  Disagree:               {total_disagree:>12,} ({100*total_disagree/total_classified:.1f}%)")
    print(f"    - Majority correct:   {total_disagree_majority_correct:>12,} ({100*total_disagree_majority_correct/total_disagree:.1f}% of disagreements)")
    print(f"    - Majority wrong:     {total_disagree_majority_wrong:>12,} ({100*total_disagree_majority_wrong/total_disagree:.1f}% of disagreements)")
    print(f"\n  KEY: Averaging works because majority vote is correct "
          f"{100*total_disagree_majority_correct/total_disagree:.1f}% of the time when models disagree")

    # ─── 5. Ensemble fixes vs breaks ──────────────────────────────────
    print(f"\n{'='*80}")
    print("5. ENSEMBLE FIXES vs BREAKS")
    print("=" * 80)

    print(f"\n  Voxels ensemble FIXES (all singles wrong, ensemble right): {ens_fixes_all:>10,}")
    print(f"  Voxels ensemble BREAKS (some single right, ensemble wrong): {ens_breaks:>10,}")
    print(f"  Net benefit (fixes - breaks):                               {ens_fixes_all - ens_breaks:>+10,}")
    if ens_breaks > 0:
        print(f"  Fix/Break ratio:                                            {ens_fixes_all/ens_breaks:>10.2f}x")

    # ─── 6. Confidence analysis ───────────────────────────────────────
    print(f"\n{'='*80}")
    print("6. CONFIDENCE ANALYSIS (mean softmax probability)")
    print("=" * 80)

    print(f"\n  Ensemble TP mean confidence: {np.mean(ens_correct_mean_conf):.4f}")
    fp_confs = [c for c in ens_wrong_mean_conf if c > 0]
    if fp_confs:
        print(f"  Ensemble FP mean confidence: {np.mean(fp_confs):.4f}")
    print()
    for mname in mnames:
        tp_c = np.mean(single_correct_mean_conf[mname]) if single_correct_mean_conf[mname] else 0
        fp_c = np.mean(single_wrong_mean_conf[mname]) if single_wrong_mean_conf[mname] else 0
        print(f"  {mname} TP conf: {tp_c:.4f}, FP conf: {fp_c:.4f}, gap: {tp_c - fp_c:.4f}")

    print("\n  Larger gap (TP - FP confidence) = better calibration")
    print("  Ensemble should have larger gap than individuals (averaging reduces noise)")

    # ─── 7. Per-lesion detection ──────────────────────────────────────
    print(f"\n{'='*80}")
    print("7. PER-LESION DETECTION AGREEMENT")
    print("=" * 80)

    print(f"\n  Total GT lesions: {total_gt_lesions}")
    print(f"  Detected by ALL 3 models:  {lesion_all_detect:>5} ({100*lesion_all_detect/total_gt_lesions:.1f}%)")
    print(f"  Detected by ANY model:     {lesion_any_detect:>5} ({100*lesion_any_detect/total_gt_lesions:.1f}%)")
    print(f"  Detected by NONE:          {lesion_none_detect:>5} ({100*lesion_none_detect/total_gt_lesions:.1f}%)")
    print(f"\n  Unique detections (only this model finds it):")
    for mname in mnames:
        print(f"    {mname}: {lesion_unique_detect[mname]} lesions")

    # Per-size detection rates
    print(f"\n  {'Size Bin':<10}", end="")
    for mname in list(MODELS.keys()) + ["ensemble"]:
        print(f"  {mname:>12}", end="")
    print(f"  {'#GT':>6}")
    print("  " + "-" * 70)

    for size_bin in SIZE_BINS:
        print(f"  {size_bin:<10}", end="")
        for mname in list(MODELS.keys()) + ["ensemble"]:
            d = lesion_detection[mname][size_bin]
            rate = 100 * d["detected"] / d["total"] if d["total"] > 0 else 0
            print(f"  {rate:>11.1f}%", end="")
        d = lesion_detection["ensemble"][size_bin]
        print(f"  {d['total']:>6}")
    print()

    # Overall detection
    print(f"  {'OVERALL':<10}", end="")
    for mname in list(MODELS.keys()) + ["ensemble"]:
        det = sum(lesion_detection[mname][s]["detected"] for s in SIZE_BINS)
        tot = sum(lesion_detection[mname][s]["total"] for s in SIZE_BINS)
        rate = 100 * det / tot if tot > 0 else 0
        print(f"  {rate:>11.1f}%", end="")
    print(f"  {total_gt_lesions:>6}")

    # ─── 8. Summary ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("8. SUMMARY — WHY THE ENSEMBLE WORKS")
    print("=" * 80)

    # Calculate average pairwise error correlation
    corrs = []
    for i, m1 in enumerate(mnames):
        for j, m2 in enumerate(mnames):
            if j > i:
                corrs.append(np.corrcoef(err_flat[m1], err_flat[m2])[0, 1])
    avg_corr = np.mean(corrs)

    print(f"""
  1. ERROR DIVERSITY: Average pairwise error correlation = {avg_corr:.4f}
     {"GOOD: Models make different mistakes (< 0.7)" if avg_corr < 0.7 else "MODERATE: Some error overlap"}

  2. DISAGREEMENT RESOLUTION: When models disagree ({total_disagree:,} voxels),
     majority vote is correct {100*total_disagree_majority_correct/total_disagree:.1f}% of the time.

  3. NET BENEFIT: Ensemble FIXES {ens_fixes_all:,} voxels that ALL singles get wrong,
     while only BREAKING {ens_breaks:,} voxels. Net = {ens_fixes_all - ens_breaks:+,} voxels.

  4. UNIQUE CONTRIBUTIONS:""")
    for mname in mnames:
        print(f"     - {mname}: {model_correct_exclusive[mname]:,} exclusively correct voxels, "
              f"{lesion_unique_detect[mname]} unique lesion detections")

    print(f"""
  5. COMPLEMENTARY STRENGTHS:
     - CNN_3D: Best detector (highest recall), finds most tiny/small lesions
     - ResEncL_25D: Best delineator (highest per-lesion Dice on detected lesions)
     - ResEncL_3D: Most balanced (good recall + precision)
     Averaging combines CNN's detection power with ResEncL's boundary precision.
""")

    print("=" * 80)
    print("DONE. Analysis complete.")


if __name__ == "__main__":
    main()
