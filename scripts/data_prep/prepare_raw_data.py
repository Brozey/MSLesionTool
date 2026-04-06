#!/usr/bin/env python3
"""
prepare_raw_data.py
===================
Extract and flatten the raw source datasets from their original zip/folder
structures into the flat ``images/`` + ``labels/`` layout expected by
``harmonize_data.py``.

Handles:
  - DS1 (MSLesSeg Dataset): nested P{N}/T{tp}/ structure, multi-timepoint,
    .nii.gz files  →  flat images/labels dirs.
  - DS2 (Patient Archive): nested Patient-{N}/ structure, .nii files
    →  flat images/labels dirs (compressed to .nii.gz).
  - DS3 (WMH Challenge): multi-site dataset (Utrecht, Singapore, Amsterdam)
    with pre-processed FLAIR + wmh labels  →  flat images/labels dirs.

Usage:
    python prepare_raw_data.py          # run all
    python prepare_raw_data.py ds2 ds3  # run only DS2 and DS3
"""

from __future__ import annotations

import gzip
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw_data"

# Dataset 1 — MSLesSeg
DS1_ZIP = RAW_DATA / "_mslesseg_outer" / "MSLesSeg Dataset.zip"
DS1_EXTRACT = RAW_DATA / "_mslesseg_extracted"
DS1_OUT_IMAGES = RAW_DATA / "dataset1" / "images"
DS1_OUT_LABELS = RAW_DATA / "dataset1" / "labels"

# Dataset 2 — Patient archive (already extracted)
DS2_EXTRACTED = RAW_DATA / "dataset2"
DS2_OUT_IMAGES = RAW_DATA / "dataset2_flat" / "images"
DS2_OUT_LABELS = RAW_DATA / "dataset2_flat" / "labels"

# Dataset 3 — WMH Challenge (dataverse_files.zip)
DS3_ZIP = PROJECT_ROOT / "dataverse_files.zip"
DS3_OUT_IMAGES = RAW_DATA / "dataset3" / "images"
DS3_OUT_LABELS = RAW_DATA / "dataset3" / "labels"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def nii_to_nii_gz(src: Path, dst: Path) -> None:
    """Compress a .nii file to .nii.gz."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ──────────────────────────────────────────────────────────────────────────────
# DS1 — MSLesSeg Dataset
# ──────────────────────────────────────────────────────────────────────────────
def prepare_ds1() -> int:
    """
    Extract MSLesSeg Dataset.zip and flatten into images/ + labels/.

    Each patient-timepoint becomes a unique case:
        P1_T1_FLAIR.nii.gz  →  images/P1_T1.nii.gz
        P1_T1_MASK.nii.gz   →  labels/P1_T1.nii.gz
    """
    print("\n" + "=" * 60)
    print("  DS1: MSLesSeg Dataset")
    print("=" * 60)

    # Extract zip if not already done
    if not DS1_EXTRACT.exists():
        print(f"  Extracting {DS1_ZIP.name} ...")
        with zipfile.ZipFile(DS1_ZIP, "r") as zf:
            zf.extractall(DS1_EXTRACT)
        print("  Extraction complete.")
    else:
        print(f"  Already extracted at {DS1_EXTRACT}")

    DS1_OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    DS1_OUT_LABELS.mkdir(parents=True, exist_ok=True)

    # Walk both train/ and test/ splits
    base = DS1_EXTRACT / "MSLesSeg Dataset"
    count = 0

    for split in ("train", "test"):
        split_dir = base / split
        if not split_dir.is_dir():
            print(f"  [WARN] {split_dir} does not exist, skipping.")
            continue

        for patient_dir in sorted(split_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name  # e.g. "P1"

            # Timepoint sub-folders (T1, T2, ...) — or files directly at patient level
            timepoint_dirs = sorted(
                [d for d in patient_dir.iterdir() if d.is_dir()]
            )

            if timepoint_dirs:
                for tp_dir in timepoint_dirs:
                    tp_id = tp_dir.name  # e.g. "T1"
                    case_id = f"{patient_id}_{tp_id}"  # e.g. "P1_T1"
                    flair = tp_dir / f"{patient_id}_{tp_id}_FLAIR.nii.gz"
                    mask = tp_dir / f"{patient_id}_{tp_id}_MASK.nii.gz"

                    if flair.exists() and mask.exists():
                        copy_file(flair, DS1_OUT_IMAGES / f"{case_id}.nii.gz")
                        copy_file(mask, DS1_OUT_LABELS / f"{case_id}.nii.gz")
                        count += 1
                    else:
                        print(f"  [WARN] Missing FLAIR or MASK for {case_id}")
            else:
                # Files directly in patient dir (test set may have this)
                flair = patient_dir / f"{patient_id}_FLAIR.nii.gz"
                mask = patient_dir / f"{patient_id}_MASK.nii.gz"
                if flair.exists() and mask.exists():
                    copy_file(flair, DS1_OUT_IMAGES / f"{patient_id}.nii.gz")
                    copy_file(mask, DS1_OUT_LABELS / f"{patient_id}.nii.gz")
                    count += 1

    print(f"  DS1: {count} FLAIR/MASK pairs flattened → {DS1_OUT_IMAGES}")
    return count


# ──────────────────────────────────────────────────────────────────────────────
# DS2 — Patient archive
# ──────────────────────────────────────────────────────────────────────────────
def prepare_ds2() -> int:
    """
    Flatten Patient-{N}/ directories into images/ + labels/.

    Original:   Patient-1/1-Flair.nii  +  Patient-1/1-LesionSeg-Flair.nii
    Output:     images/Patient-1.nii.gz  +  labels/Patient-1.nii.gz
    
    Also compresses .nii → .nii.gz.  Skips cases that already exist on disk.
    """
    print("\n" + "=" * 60)
    print("  DS2: Patient Archive")
    print("=" * 60)

    DS2_OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    DS2_OUT_LABELS.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    for patient_dir in sorted(DS2_EXTRACTED.iterdir()):
        if not patient_dir.is_dir() or not patient_dir.name.startswith("Patient-"):
            continue

        patient_name = patient_dir.name  # e.g. "Patient-1"
        patient_num = patient_name.split("-")[1]

        out_img = DS2_OUT_IMAGES / f"{patient_name}.nii.gz"
        out_lbl = DS2_OUT_LABELS / f"{patient_name}.nii.gz"

        # Skip if both outputs already exist and are > 0 bytes
        if out_img.exists() and out_lbl.exists() and out_img.stat().st_size > 0 and out_lbl.stat().st_size > 0:
            skipped += 1
            count += 1
            continue

        # Find FLAIR and lesion segmentation
        flair = patient_dir / f"{patient_num}-Flair.nii"
        lesion = patient_dir / f"{patient_num}-LesionSeg-Flair.nii"

        # Some datasets may use slightly different naming
        if not flair.exists():
            candidates = list(patient_dir.glob("*[Ff]lair.nii"))
            flair = candidates[0] if candidates else flair
        if not lesion.exists():
            candidates = list(patient_dir.glob("*LesionSeg*Flair.nii"))
            lesion = candidates[0] if candidates else lesion

        if flair.exists() and lesion.exists():
            print(f"  Compressing {patient_name} ...")
            nii_to_nii_gz(flair, out_img)
            nii_to_nii_gz(lesion, out_lbl)
            count += 1
        else:
            missing = []
            if not flair.exists():
                missing.append(f"FLAIR ({flair.name})")
            if not lesion.exists():
                missing.append(f"Label ({lesion.name})")
            print(f"  [WARN] {patient_name}: missing {', '.join(missing)}")

    print(f"  DS2: {count} FLAIR/lesion pairs ({skipped} skipped, "
          f"{count - skipped} new) → {DS2_OUT_IMAGES}")
    return count


# ──────────────────────────────────────────────────────────────────────────────
# DS3 — WMH Challenge (dataverse_files.zip)
# ──────────────────────────────────────────────────────────────────────────────
def _sanitize(name: str) -> str:
    """Replace spaces/dots/special chars with underscores for safe filenames."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_")


def prepare_ds3() -> int:
    """
    Extract WMH Challenge dataset from dataverse_files.zip and flatten.

    Structure inside zip:
        {training|test}/{Site}[/{Scanner}]/{id}/pre/FLAIR.nii.gz   ← image
        {training|test}/{Site}[/{Scanner}]/{id}/wmh.nii.gz         ← label

    Amsterdam has a scanner sub-folder (GE3T, GE1T5, Philips_VU .PETMR_01.);
    Utrecht and Singapore do not.

    Output:
        images/{Site}_{Scanner}_{id}.nii.gz  or  images/{Site}_{id}.nii.gz
        labels/{Site}_{Scanner}_{id}.nii.gz  or  labels/{Site}_{id}.nii.gz
    """
    print("\n" + "=" * 60)
    print("  DS3: WMH Challenge")
    print("=" * 60)

    if not DS3_ZIP.exists():
        print(f"  [ERROR] Zip not found: {DS3_ZIP}")
        return 0

    DS3_OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    DS3_OUT_LABELS.mkdir(parents=True, exist_ok=True)

    count = 0
    with zipfile.ZipFile(DS3_ZIP, "r") as zf:
        all_names = zf.namelist()

        # Find all pre/FLAIR.nii.gz entries (skip additional_annotations)
        flair_entries = [
            n for n in all_names
            if n.endswith("/pre/FLAIR.nii.gz")
            and not n.startswith("additional_annotations")
        ]

        for flair_path in sorted(flair_entries):
            # Derive label path: replace /pre/FLAIR.nii.gz with /wmh.nii.gz
            subject_dir = flair_path.rsplit("/pre/FLAIR.nii.gz", 1)[0]
            wmh_path = subject_dir + "/wmh.nii.gz"

            if wmh_path not in all_names:
                print(f"  [WARN] No wmh.nii.gz for {subject_dir}")
                continue

            # Parse path to build case_id
            parts = subject_dir.split("/")
            # parts[0] = training|test, parts[1] = Site, ...
            split = parts[0]          # training or test
            site = parts[1]           # Utrecht, Singapore, Amsterdam

            if site == "Amsterdam":
                # Amsterdam has scanner sub-dir: Amsterdam/GE3T/100
                scanner = parts[2]
                subj_id = parts[3]
                case_id = f"{_sanitize(site)}_{_sanitize(scanner)}_{subj_id}"
            else:
                # Utrecht/0, Singapore/50
                subj_id = parts[2]
                case_id = f"{_sanitize(site)}_{subj_id}"

            out_img = DS3_OUT_IMAGES / f"{case_id}.nii.gz"
            out_lbl = DS3_OUT_LABELS / f"{case_id}.nii.gz"

            # Skip if both exist
            if (out_img.exists() and out_lbl.exists()
                    and out_img.stat().st_size > 0
                    and out_lbl.stat().st_size > 0):
                count += 1
                continue

            # Extract FLAIR
            with zf.open(flair_path) as src, open(out_img, "wb") as dst:
                shutil.copyfileobj(src, dst)

            # Extract wmh label
            with zf.open(wmh_path) as src, open(out_lbl, "wb") as dst:
                shutil.copyfileobj(src, dst)

            count += 1

    print(f"  DS3: {count} FLAIR/WMH pairs extracted → {DS3_OUT_IMAGES}")
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Preparing raw datasets for nnU-Net harmonisation pipeline")
    print(f"Project root: {PROJECT_ROOT}")

    # Allow selective runs: python prepare_raw_data.py ds2 ds3
    requested = {a.lower() for a in sys.argv[1:]}
    run_all = len(requested) == 0

    n1, n2, n3 = 0, 0, 0

    if run_all or "ds1" in requested:
        n1 = prepare_ds1()
    if run_all or "ds2" in requested:
        n2 = prepare_ds2()
    if run_all or "ds3" in requested:
        n3 = prepare_ds3()

    print("\n" + "=" * 60)
    print(f"  SUMMARY")
    if run_all or "ds1" in requested:
        print(f"  DS1 (MSLesSeg)       : {n1} cases → {DS1_OUT_IMAGES}")
    if run_all or "ds2" in requested:
        print(f"  DS2 (Patient Archive): {n2} cases → {DS2_OUT_IMAGES}")
    if run_all or "ds3" in requested:
        print(f"  DS3 (WMH Challenge)  : {n3} cases → {DS3_OUT_IMAGES}")
    total = n1 + n2 + n3
    print(f"  Total                : {total} cases")
    print("=" * 60)
    print("\nNext: update config/dataset_config.yaml paths, then run harmonize_data.py")


if __name__ == "__main__":
    main()
