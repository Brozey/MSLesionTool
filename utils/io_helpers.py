"""
utils/io_helpers.py
Shared I/O helpers for the FLAIR lesion segmentation nnU-Net pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import nibabel as nib
except ImportError:
    nib = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────────────
# Module-level logger
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("nnunet_pipeline")


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """
    Configure the pipeline-wide logger.

    Parameters
    ----------
    level    : Logging level (default: INFO).
    log_file : If provided, also write log messages to this file.
    """
    fmt = "[%(asctime)s] %(levelname)-8s %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(level)

    # Optional file handler (avoids duplicates if called multiple times)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(str(log_file), encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str | Path = None) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "dataset_config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# nnU-Net path helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_nnunet_env_paths() -> Tuple[Path, Path, Path]:
    """Return (nnUNet_raw, nnUNet_preprocessed, nnUNet_results) from env."""
    keys = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
    paths: List[Path] = []
    for k in keys:
        val = os.environ.get(k)
        if val is None:
            raise EnvironmentError(
                f"Environment variable '{k}' is not set. "
                "Run setup_env.sh / setup_env.ps1 first."
            )
        paths.append(Path(val))
    return tuple(paths)  # type: ignore[return-value]


def nnunet_dataset_dir(dataset_id: int, dataset_name: str) -> str:
    """Return the canonical nnU-Net folder name, e.g. 'Dataset500_RawFLAIR'."""
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def ensure_nnunet_dataset_dirs(raw_base: Path, dataset_id: int,
                                dataset_name: str) -> Dict[str, Path]:
    """Create imagesTr, imagesTs, labelsTr, labelsTs under the dataset folder."""
    ds_dir = raw_base / nnunet_dataset_dir(dataset_id, dataset_name)
    subdirs = {}
    for sub in ("imagesTr", "imagesTs", "labelsTr", "labelsTs"):
        p = ds_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        subdirs[sub] = p
    return subdirs


# ──────────────────────────────────────────────────────────────────────────────
# Filename helpers (nnU-Net v2 conventions)
# ──────────────────────────────────────────────────────────────────────────────
def to_nnunet_image_name(base_name: str) -> str:
    """Return '<base_name>_0000.nii.gz' (single-channel FLAIR)."""
    return f"{base_name}_0000.nii.gz"


def to_nnunet_label_name(base_name: str) -> str:
    """Return '<base_name>.nii.gz'."""
    return f"{base_name}.nii.gz"


def strip_nifti_ext(filename: str) -> str:
    """Strip .nii.gz or .nii extension and return the stem."""
    name = filename
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def prefixed_base(prefix: str, original_stem: str) -> str:
    """Return prefix + '_' + original_stem, e.g. 'DS1_sub-001'."""
    return f"{prefix}_{original_stem}"


# ──────────────────────────────────────────────────────────────────────────────
# File copy with nnU-Net rename
# ──────────────────────────────────────────────────────────────────────────────
def copy_as_nnunet(
    src_path: Path,
    dst_dir: Path,
    new_base_name: str,
    is_label: bool = False,
) -> Path:
    """
    Copy *src_path* into *dst_dir* with the correct nnU-Net naming.
    - Images  -> <new_base_name>_0000.nii.gz
    - Labels  -> <new_base_name>.nii.gz
    Returns the destination path.
    """
    if is_label:
        dst_name = to_nnunet_label_name(new_base_name)
    else:
        dst_name = to_nnunet_image_name(new_base_name)
    dst_path = dst_dir / dst_name
    shutil.copy2(src_path, dst_path)
    return dst_path


# ──────────────────────────────────────────────────────────────────────────────
# JSON writer
# ──────────────────────────────────────────────────────────────────────────────
def write_json(data: Dict[str, Any], path: Path) -> None:
    """Write a dict to a JSON file with human-readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=4, sort_keys=False)
    print(f"  [IO] Wrote {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Glob helpers
# ──────────────────────────────────────────────────────────────────────────────
def glob_nifti(directory: Path, suffix: str = ".nii.gz") -> List[Path]:
    """Return sorted list of NIfTI files in *directory* matching *suffix*."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        print(f"  [WARN] No files matching '*{suffix}' in {directory}")
    return files


def pair_images_labels(
    images: List[Path],
    labels: List[Path],
    img_suffix: str = ".nii.gz",
    lbl_suffix: str = ".nii.gz",
) -> List[Tuple[Path, Path]]:
    """
    Match image files to label files by their stem (after stripping suffix).
    Returns a list of (image_path, label_path) tuples.
    Raises ValueError if any image has no matching label.
    """
    label_map: Dict[str, Path] = {}
    for lbl in labels:
        stem = strip_nifti_ext(lbl.name)
        label_map[stem] = lbl

    paired: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for img in images:
        stem = strip_nifti_ext(img.name)
        if stem in label_map:
            paired.append((img, label_map[stem]))
        else:
            missing.append(img.name)

    if missing:
        raise ValueError(
            f"The following images have no matching labels: {missing}"
        )
    return paired


# ──────────────────────────────────────────────────────────────────────────────
# NIfTI geometry validation
# ──────────────────────────────────────────────────────────────────────────────
def validate_nifti_pair(
    image_path: Path,
    label_path: Path,
    shape_tolerance: int = 0,
) -> List[str]:
    """
    Compare spatial properties of an image/label NIfTI pair.

    Returns a list of warning strings (empty = all OK).
    Checks performed:
      1. Matching voxel-grid shape (within *shape_tolerance*).
      2. Matching affine matrices (up to floating-point tolerance).
      3. Label contains at least two unique values (not blank).
    """
    if nib is None:
        logger.warning("nibabel not installed — skipping geometry validation.")
        return []

    warnings: List[str] = []
    img_nii = nib.load(str(image_path))
    lbl_nii = nib.load(str(label_path))

    # Shape check
    img_shape = img_nii.shape[:3]  # ignore 4th dim if present
    lbl_shape = lbl_nii.shape[:3]
    diff = tuple(abs(a - b) for a, b in zip(img_shape, lbl_shape))
    if any(d > shape_tolerance for d in diff):
        warnings.append(
            f"Shape mismatch: image={img_shape} vs label={lbl_shape}  "
            f"(diff={diff}, tolerance={shape_tolerance})"
        )

    # Affine check
    if not np.allclose(img_nii.affine, lbl_nii.affine, atol=1e-3):
        warnings.append("Affine matrices differ between image and label.")

    # Blank-label check
    lbl_data = np.asarray(lbl_nii.dataobj)
    unique_vals = np.unique(lbl_data)
    if len(unique_vals) < 2:
        warnings.append(
            f"Label has only {len(unique_vals)} unique value(s): {unique_vals}. "
            "May be a blank mask."
        )

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Label binarization
# ──────────────────────────────────────────────────────────────────────────────
def binarize_label(label_path: Path, output_path: Path) -> Path:
    """
    Binarize a NIfTI label mask: voxel > 0 → 1, else 0.
    Writes the result to *output_path* preserving header/affine.
    Returns *output_path*.
    """
    if nib is None:
        raise ImportError("nibabel is required for label binarization.")

    nii = nib.load(str(label_path))
    data = np.asarray(nii.dataobj)
    binary = (data > 0).astype(np.uint8)

    out_nii = nib.Nifti1Image(binary, affine=nii.affine, header=nii.header)
    out_nii.set_data_dtype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nii, str(output_path))
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Stratified train/test split
# ──────────────────────────────────────────────────────────────────────────────
def stratified_split(
    records: list,
    train_ratio: float,
    seed: int,
) -> Tuple[list, list]:
    """
    Split records into train/test while preserving the proportion of each
    source dataset (identified by the prefix before the first '_').

    Each record is expected to be (prefixed_base_name, ...).
    Falls back to a simple random split if grouping fails.
    """
    import random as _random

    rng = _random.Random(seed)

    # Group by prefix (dataset origin)
    groups: Dict[str, list] = {}
    for rec in records:
        prefix = rec[0].split("_", 1)[0]
        groups.setdefault(prefix, []).append(rec)

    train_set: list = []
    test_set: list = []
    for prefix, group in sorted(groups.items()):
        rng.shuffle(group)
        n_train = max(1, int(len(group) * train_ratio))
        train_set.extend(group[:n_train])
        test_set.extend(group[n_train:])
        logger.info(
            "  [stratified] %-6s  total=%3d  train=%3d  test=%3d",
            prefix, len(group), n_train, len(group) - n_train,
        )

    # Final shuffle so training isn't grouped by source
    rng.shuffle(train_set)
    rng.shuffle(test_set)
    return train_set, test_set
