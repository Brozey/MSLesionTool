#!/usr/bin/env python3
"""
analyze_lesion_extents.py
=========================
Measures per-lesion spatial extent (in slices) along each anatomical axis
for all connected-component lesions in DS003 training labels.

Purpose: Determine optimal context window K for 3 view-specific 2.5D models
(axial, coronal, sagittal) by analyzing how many slices lesions span in each
direction.

For each lesion (connected component), measures:
  - Axial extent   (axis 2 = z, inferior-superior)
  - Coronal extent (axis 1 = y, anterior-posterior)
  - Sagittal extent (axis 0 = x, left-right)

Reports percentile statistics to inform K selection:
  - Median: typical lesion span
  - 75th percentile: captures most lesions fully
  - 90th percentile: captures nearly all lesions
  - 95th percentile: conservative upper bound

Also accounts for voxel spacing (anisotropic data → different mm extent).

Output: Console summary + CSV at data/nnUNet_results/evaluation/lesion_extents.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.ndimage import label as connected_components, find_objects

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset003_Combined" / "labelsTr"
OUTPUT_DIR = REPO_ROOT / "results" / "evaluation"


def analyze_single_label(label_path: Path) -> list[dict]:
    """Analyze all lesions in a single label file."""
    img = nib.load(str(label_path))
    data = img.get_fdata().astype(np.uint8)
    spacing = img.header.get_zooms()[:3]  # (sx, sy, sz) in mm

    # Binarize (any label > 0 is lesion)
    binary = data > 0
    if binary.sum() == 0:
        return []

    labeled, n_lesions = connected_components(binary)
    slices = find_objects(labeled)

    results = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue

        lesion_mask = labeled[sl] == (i + 1)
        vol_voxels = int(lesion_mask.sum())
        if vol_voxels == 0:
            continue

        # Extent in slices along each axis (bounding box size)
        extent_x = sl[0].stop - sl[0].start  # sagittal
        extent_y = sl[1].stop - sl[1].start  # coronal
        extent_z = sl[2].stop - sl[2].start  # axial

        # Extent in mm
        extent_x_mm = extent_x * spacing[0]
        extent_y_mm = extent_y * spacing[1]
        extent_z_mm = extent_z * spacing[2]

        # Volume in mm³
        voxel_vol = float(np.prod(spacing))
        vol_mm3 = vol_voxels * voxel_vol

        results.append({
            "subject": label_path.stem.replace(".nii", ""),
            "lesion_id": i + 1,
            "vol_voxels": vol_voxels,
            "vol_mm3": round(vol_mm3, 2),
            "extent_x_slices": extent_x,  # sagittal
            "extent_y_slices": extent_y,  # coronal
            "extent_z_slices": extent_z,  # axial
            "extent_x_mm": round(extent_x_mm, 2),
            "extent_y_mm": round(extent_y_mm, 2),
            "extent_z_mm": round(extent_z_mm, 2),
            "spacing_x": round(spacing[0], 3),
            "spacing_y": round(spacing[1], 3),
            "spacing_z": round(spacing[2], 3),
        })

    return results


def main():
    label_files = sorted(LABELS_DIR.glob("*.nii.gz"))
    print(f"Found {len(label_files)} label files in {LABELS_DIR}")

    all_lesions = []
    for i, lf in enumerate(label_files):
        lesions = analyze_single_label(lf)
        all_lesions.extend(lesions)
        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{len(label_files)} subjects ({len(all_lesions)} lesions so far)")

    print(f"\nTotal lesions found: {len(all_lesions)}")

    # Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "lesion_extents.csv"
    if all_lesions:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_lesions[0].keys())
            writer.writeheader()
            writer.writerows(all_lesions)
        print(f"Saved: {csv_path}")

    # ── Statistics ────────────────────────────────────────────────────────
    ext_x = np.array([l["extent_x_slices"] for l in all_lesions])
    ext_y = np.array([l["extent_y_slices"] for l in all_lesions])
    ext_z = np.array([l["extent_z_slices"] for l in all_lesions])
    ext_x_mm = np.array([l["extent_x_mm"] for l in all_lesions])
    ext_y_mm = np.array([l["extent_y_mm"] for l in all_lesions])
    ext_z_mm = np.array([l["extent_z_mm"] for l in all_lesions])
    vols = np.array([l["vol_mm3"] for l in all_lesions])

    # Size categories
    size_bins = {
        "tiny (<10 mm³)":    (0, 10),
        "small (10-100)":    (10, 100),
        "medium (100-1000)": (100, 1000),
        "large (>1000)":     (1000, np.inf),
    }

    print("\n" + "=" * 80)
    print("LESION EXTENT ANALYSIS — ALL LESIONS")
    print("=" * 80)

    percentiles = [25, 50, 75, 90, 95, 99]

    print(f"\n{'':>20} {'Sagittal (X)':>14} {'Coronal (Y)':>14} {'Axial (Z)':>14}")
    print("-" * 65)
    print(f"{'Mean (slices)':>20} {ext_x.mean():>14.1f} {ext_y.mean():>14.1f} {ext_z.mean():>14.1f}")
    for p in percentiles:
        px = np.percentile(ext_x, p)
        py = np.percentile(ext_y, p)
        pz = np.percentile(ext_z, p)
        print(f"{'P' + str(p) + ' (slices)':>20} {px:>14.1f} {py:>14.1f} {pz:>14.1f}")
    print(f"{'Max (slices)':>20} {ext_x.max():>14d} {ext_y.max():>14d} {ext_z.max():>14d}")

    print(f"\n{'':>20} {'Sagittal (X)':>14} {'Coronal (Y)':>14} {'Axial (Z)':>14}")
    print("-" * 65)
    print(f"{'Mean (mm)':>20} {ext_x_mm.mean():>14.1f} {ext_y_mm.mean():>14.1f} {ext_z_mm.mean():>14.1f}")
    for p in percentiles:
        px = np.percentile(ext_x_mm, p)
        py = np.percentile(ext_y_mm, p)
        pz = np.percentile(ext_z_mm, p)
        print(f"{'P' + str(p) + ' (mm)':>20} {px:>14.1f} {py:>14.1f} {pz:>14.1f}")

    # ── Per size category ─────────────────────────────────────────────────
    for cat_name, (lo, hi) in size_bins.items():
        mask = (vols >= lo) & (vols < hi)
        n = mask.sum()
        if n == 0:
            continue
        print(f"\n{'=' * 80}")
        print(f"LESION EXTENT — {cat_name}  (n={n})")
        print(f"{'=' * 80}")

        cx, cy, cz = ext_x[mask], ext_y[mask], ext_z[mask]
        print(f"\n{'':>20} {'Sagittal (X)':>14} {'Coronal (Y)':>14} {'Axial (Z)':>14}")
        print("-" * 65)
        print(f"{'Mean (slices)':>20} {cx.mean():>14.1f} {cy.mean():>14.1f} {cz.mean():>14.1f}")
        for p in percentiles:
            print(f"{'P' + str(p) + ' (slices)':>20} {np.percentile(cx, p):>14.1f} {np.percentile(cy, p):>14.1f} {np.percentile(cz, p):>14.1f}")
        print(f"{'Max (slices)':>20} {cx.max():>14d} {cy.max():>14d} {cz.max():>14d}")

    # ── K recommendations ─────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("K RECOMMENDATIONS FOR 2.5D MODELS")
    print(f"{'=' * 80}")
    print("\nK = context window = K slices total (center + (K-1)/2 on each side)")
    print("K must be odd. Chosen to capture P90 of medium+large lesions.\n")

    # Focus on medium + large lesions (they contribute most to DSC)
    med_large = vols >= 100
    if med_large.sum() > 0:
        for axis_name, ext_arr in [("Sagittal (X)", ext_x), ("Coronal (Y)", ext_y), ("Axial (Z)", ext_z)]:
            vals = ext_arr[med_large]
            p75 = np.percentile(vals, 75)
            p90 = np.percentile(vals, 90)
            p95 = np.percentile(vals, 95)
            # K should be odd and >= p90 extent
            k_p75 = int(np.ceil(p75)) | 1  # round up to odd
            k_p90 = int(np.ceil(p90)) | 1
            k_p95 = int(np.ceil(p95)) | 1
            print(f"  {axis_name}:")
            print(f"    P75={p75:.0f} → K={k_p75}  |  P90={p90:.0f} → K={k_p90}  |  P95={p95:.0f} → K={k_p95}")

    # Also show for ALL lesions
    print(f"\n  For ALL lesions (including tiny):")
    for axis_name, ext_arr in [("Sagittal (X)", ext_x), ("Coronal (Y)", ext_y), ("Axial (Z)", ext_z)]:
        p75 = np.percentile(ext_arr, 75)
        p90 = np.percentile(ext_arr, 90)
        p95 = np.percentile(ext_arr, 95)
        k_p75 = int(np.ceil(p75)) | 1
        k_p90 = int(np.ceil(p90)) | 1
        k_p95 = int(np.ceil(p95)) | 1
        print(f"  {axis_name}:")
        print(f"    P75={p75:.0f} → K={k_p75}  |  P90={p90:.0f} → K={k_p90}  |  P95={p95:.0f} → K={k_p95}")

    # ── Spacing distribution ──────────────────────────────────────────────
    spacings = set()
    for l in all_lesions:
        spacings.add((l["spacing_x"], l["spacing_y"], l["spacing_z"]))
    print(f"\n  Unique voxel spacings found: {len(spacings)}")
    for sp in sorted(spacings):
        n_with = sum(1 for l in all_lesions if (l["spacing_x"], l["spacing_y"], l["spacing_z"]) == sp)
        print(f"    {sp} — {n_with} lesions")


if __name__ == "__main__":
    main()
