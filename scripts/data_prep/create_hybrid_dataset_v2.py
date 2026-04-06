"""
Create Dataset502_HybridFLAIR with PATIENT-LEVEL splitting.

Fixes the data leakage issue where the same patient's different timepoints
could end up in both train and test sets.

Strategy:
  - DS1 (MSLesSeg): Images from DS500 (raw) — already skull-stripped at source
  - DS3 (WMH Challenge): Images from DS501 (SynthStrip) — properly skull-stripped
  - Labels: Identical between DS500/DS501, sourced from DS500
  
Split:
  - DS1 subjects grouped by patient ID (DS1_P{N}_T{tp} → group by P{N})
  - All timepoints of a patient go to the SAME set (train or test)
  - DS3 subjects are all unique patients, split independently
  - 80/20 train/test ratio, seed=42
"""

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parents[2]


def extract_ds1_patient_id(case_id: str) -> str:
    """
    Extract patient ID from DS1 case names.
    
    Examples:
      DS1_P1_T1  → DS1_P1
      DS1_P12_T4 → DS1_P12
      DS1_P54    → DS1_P54  (test patients without timepoint suffix)
    """
    # Match DS1_P{number} optionally followed by _T{number}
    m = re.match(r"(DS1_P\d+)(?:_T\d+)?$", case_id)
    if m:
        return m.group(1)
    return case_id  # fallback


def get_case_id(filename: str) -> str:
    """Strip _0000.nii.gz suffix to get case ID."""
    return filename.replace("_0000.nii.gz", "")


def create_hybrid_dataset():
    base = Path(os.environ.get("nnUNet_raw", REPO_ROOT / "data" / "nnUNet_raw"))
    ds500 = base / "Dataset500_RawFLAIR"
    ds501 = base / "Dataset501_SkullStrippedFLAIR"
    ds502 = base / "Dataset502_HybridFLAIR"

    seed = 42
    train_ratio = 0.8

    # Verify sources
    for ds in [ds500, ds501]:
        if not ds.exists():
            raise FileNotFoundError(f"{ds} not found!")

    # ── Collect ALL subjects from DS500 ──────────────────────────────────
    # (DS500 has the complete set — we use it as the master list)
    all_images = sorted(
        list((ds500 / "imagesTr").glob("*_0000.nii.gz"))
        + list((ds500 / "imagesTs").glob("*_0000.nii.gz"))
    )

    # Separate DS1 and DS3
    ds1_cases = []  # (case_id, image_path)
    ds3_cases = []

    for img_path in all_images:
        case_id = get_case_id(img_path.name)
        if case_id.startswith("DS1_"):
            ds1_cases.append(case_id)
        elif case_id.startswith("DS3_"):
            ds3_cases.append(case_id)
        else:
            print(f"WARNING: Unknown prefix: {case_id}")

    print(f"Total DS1 series: {len(ds1_cases)}")
    print(f"Total DS3 subjects: {len(ds3_cases)}")

    # ── Group DS1 by patient ─────────────────────────────────────────────
    ds1_patient_groups = defaultdict(list)
    for case_id in ds1_cases:
        patient_id = extract_ds1_patient_id(case_id)
        ds1_patient_groups[patient_id].append(case_id)

    ds1_patients = sorted(ds1_patient_groups.keys(),
                          key=lambda x: int(re.search(r'P(\d+)', x).group(1)))

    print(f"\nDS1 unique patients: {len(ds1_patients)}")
    multi_tp = {p: cases for p, cases in ds1_patient_groups.items() if len(cases) > 1}
    print(f"DS1 patients with multiple timepoints: {len(multi_tp)}")
    for p in sorted(multi_tp, key=lambda x: int(re.search(r'P(\d+)', x).group(1))):
        print(f"  {p}: {sorted(multi_tp[p])}")

    # ── Patient-level split for DS1 ──────────────────────────────────────
    rng = random.Random(seed)
    ds1_patients_shuffled = list(ds1_patients)
    rng.shuffle(ds1_patients_shuffled)

    n_train_patients = int(len(ds1_patients_shuffled) * train_ratio)
    ds1_train_patients = set(ds1_patients_shuffled[:n_train_patients])
    ds1_test_patients = set(ds1_patients_shuffled[n_train_patients:])

    ds1_train_cases = []
    ds1_test_cases = []
    for patient_id in sorted(ds1_patient_groups.keys(),
                             key=lambda x: int(re.search(r'P(\d+)', x).group(1))):
        cases = ds1_patient_groups[patient_id]
        if patient_id in ds1_train_patients:
            ds1_train_cases.extend(cases)
        else:
            ds1_test_cases.extend(cases)

    print(f"\nDS1 patient-level split:")
    print(f"  Train: {len(ds1_train_patients)} patients → {len(ds1_train_cases)} series")
    print(f"  Test:  {len(ds1_test_patients)} patients → {len(ds1_test_cases)} series")

    # ── Split for DS3 (each subject = unique patient) ────────────────────
    ds3_shuffled = list(ds3_cases)
    rng2 = random.Random(seed)
    rng2.shuffle(ds3_shuffled)

    n_train_ds3 = int(len(ds3_shuffled) * train_ratio)
    ds3_train_cases = ds3_shuffled[:n_train_ds3]
    ds3_test_cases = ds3_shuffled[n_train_ds3:]

    print(f"\nDS3 split:")
    print(f"  Train: {len(ds3_train_cases)} subjects")
    print(f"  Test:  {len(ds3_test_cases)} subjects")

    # ── Combine ──────────────────────────────────────────────────────────
    train_cases = sorted(ds1_train_cases + ds3_train_cases)
    test_cases = sorted(ds1_test_cases + ds3_test_cases)

    print(f"\nTotal split:")
    print(f"  Train: {len(train_cases)} (DS1: {len(ds1_train_cases)}, DS3: {len(ds3_train_cases)})")
    print(f"  Test:  {len(test_cases)} (DS1: {len(ds1_test_cases)}, DS3: {len(ds3_test_cases)})")

    # ── Verify NO leakage ────────────────────────────────────────────────
    train_patient_ids = {extract_ds1_patient_id(c) for c in train_cases if c.startswith("DS1_")}
    test_patient_ids = {extract_ds1_patient_id(c) for c in test_cases if c.startswith("DS1_")}
    overlap = train_patient_ids & test_patient_ids
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE DETECTED! Overlapping patients: {overlap}")
    print(f"\n✓ No patient-level overlap between train and test (verified)")

    # ── Clear old DS502 and create new ───────────────────────────────────
    if ds502.exists():
        print(f"\nRemoving old Dataset502...")
        shutil.rmtree(str(ds502))

    for subdir in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
        (ds502 / subdir).mkdir(parents=True, exist_ok=True)

    # ── Copy files ───────────────────────────────────────────────────────
    def find_source_image(case_id: str) -> Path:
        """Find the image file in DS500 (train or test)."""
        fname = f"{case_id}_0000.nii.gz"
        for subdir in ["imagesTr", "imagesTs"]:
            p = ds500 / subdir / fname
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found for {case_id}")

    def find_source_label(case_id: str) -> Path:
        """Find the label file in DS500 (train or test)."""
        fname = f"{case_id}.nii.gz"
        for subdir in ["labelsTr", "labelsTs"]:
            p = ds500 / subdir / fname
            if p.exists():
                return p
        raise FileNotFoundError(f"Label not found for {case_id}")

    def find_ss_image(case_id: str) -> Path:
        """Find the skull-stripped image in DS501."""
        fname = f"{case_id}_0000.nii.gz"
        for subdir in ["imagesTr", "imagesTs"]:
            p = ds501 / subdir / fname
            if p.exists():
                return p
        raise FileNotFoundError(f"Skull-stripped image not found for {case_id}")

    errors = []

    for split_label, cases in [("Tr", train_cases), ("Ts", test_cases)]:
        print(f"\n{'='*60}")
        print(f"Copying images{split_label}: {len(cases)} subjects")
        ds1_count = ds3_count = 0

        for case_id in cases:
            img_dst = ds502 / f"images{split_label}" / f"{case_id}_0000.nii.gz"
            lbl_dst = ds502 / f"labels{split_label}" / f"{case_id}.nii.gz"

            try:
                # Label: always from DS500
                src_lbl = find_source_label(case_id)

                if case_id.startswith("DS1_"):
                    # DS1: Use raw image from DS500 (already skull-stripped)
                    src_img = find_source_image(case_id)
                    ds1_count += 1
                elif case_id.startswith("DS3_"):
                    # DS3: Use skull-stripped image from DS501
                    src_img = find_ss_image(case_id)
                    ds3_count += 1
                else:
                    errors.append(f"Unknown prefix: {case_id}")
                    continue

                shutil.copy2(str(src_img), str(img_dst))
                shutil.copy2(str(src_lbl), str(lbl_dst))

            except FileNotFoundError as e:
                errors.append(str(e))
                print(f"  ERROR: {e}")

        print(f"  DS1 (raw): {ds1_count}, DS3 (skull-stripped): {ds3_count}")

    if errors:
        print(f"\n{'!'*60}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    # ── dataset.json ─────────────────────────────────────────────────────
    dataset_json = {
        "channel_names": {"0": "FLAIR"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz",
        "name": "Dataset502_HybridFLAIR",
        "description": (
            "Hybrid MS lesion segmentation dataset with patient-level train/test split. "
            "DS1 (MSLesSeg): raw images (already skull-stripped at source). "
            "DS3 (WMH Challenge): SynthStrip skull-stripped images. "
            "All timepoints of the same patient are in the same split to prevent data leakage."
        ),
        "reference": "MSLesSeg (Rondinella et al.) + WMH Challenge (MICCAI 2017)",
        "licence": "Research use only",
    }
    with open(ds502 / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # ── Print patient assignments for DS1 ────────────────────────────────
    print(f"\n{'='*60}")
    print("DS1 Patient Assignments:")
    print(f"\n  TRAIN patients ({len(ds1_train_patients)}):")
    for p in sorted(ds1_train_patients, key=lambda x: int(re.search(r'P(\d+)', x).group(1))):
        cases = sorted(ds1_patient_groups[p])
        print(f"    {p}: {cases}")

    print(f"\n  TEST patients ({len(ds1_test_patients)}):")
    for p in sorted(ds1_test_patients, key=lambda x: int(re.search(r'P(\d+)', x).group(1))):
        cases = sorted(ds1_patient_groups[p])
        print(f"    {p}: {cases}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Dataset502_HybridFLAIR created successfully!")
    print(f"  Training: {len(train_cases)} samples ({len(ds1_train_patients)} DS1 patients + {len(ds3_train_cases)} DS3)")
    print(f"  Testing:  {len(test_cases)} samples ({len(ds1_test_patients)} DS1 patients + {len(ds3_test_cases)} DS3)")
    print(f"  No patient-level data leakage ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_hybrid_dataset()
