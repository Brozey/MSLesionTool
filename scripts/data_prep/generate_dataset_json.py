#!/usr/bin/env python3
"""
generate_dataset_json.py
========================
Dynamically build and save ``dataset.json`` for any nnU-Net v2 dataset by
scanning the files already present in ``imagesTr``.

Can be called:
  1. As a library   –  ``from generate_dataset_json import generate_dataset_json``
  2. From the CLI   –  ``python generate_dataset_json.py --id 500 --name RawFLAIR``

The generated JSON follows the nnU-Net v2 spec:
  - ``channel_names``: ``{"0": "FLAIR"}``
  - ``labels``: ``{"background": 0, "lesion": 1}``
  - ``numTraining``: auto-counted from imagesTr
  - ``file_ending``: ``.nii.gz``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_helpers import (
    get_nnunet_env_paths,
    load_config,
    nnunet_dataset_dir,
    write_json,
)


def generate_dataset_json(
    dataset_id: int,
    dataset_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Scan imagesTr inside the dataset folder and create ``dataset.json``.

    Parameters
    ----------
    dataset_id   : nnU-Net numeric ID (e.g. 500).
    dataset_name : Descriptive name without the ID prefix (e.g. "RawFLAIR").
    config       : Loaded YAML config (used for label map).  If None, defaults
                   are applied.

    Returns
    -------
    Path to the written dataset.json.
    """
    raw_base, _, _ = get_nnunet_env_paths()
    ds_folder = raw_base / nnunet_dataset_dir(dataset_id, dataset_name)

    images_tr = ds_folder / "imagesTr"
    if not images_tr.is_dir():
        raise FileNotFoundError(f"imagesTr not found at {images_tr}")

    # Count training cases (each image ends with _0000.nii.gz)
    train_images = sorted(images_tr.glob("*_0000.nii.gz"))
    num_training = len(train_images)

    # Resolve label map from config or use defaults
    if config and "labels" in config:
        raw_labels = config["labels"]
        # nnU-Net v2 expects string keys
        labels = {str(k): int(v) for k, v in raw_labels.items()}
    else:
        labels = {"background": 0, "lesion": 1}

    dataset_json: Dict[str, Any] = {
        "channel_names": {
            "0": "FLAIR",
        },
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        # Optional but recommended metadata
        "name": f"Dataset{dataset_id:03d}_{dataset_name}",
        "description": (
            f"Merged multi-source FLAIR lesion segmentation dataset "
            f"({dataset_name})."
        ),
        "reference": "",
        "licence": "",
        "release": "1.0",
    }

    out_path = ds_folder / "dataset.json"
    write_json(dataset_json, out_path)
    print(f"[json] Dataset {dataset_id}: {num_training} training cases registered.")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset.json for an nnU-Net v2 dataset."
    )
    parser.add_argument("--id", type=int, required=True, help="Dataset ID (e.g. 500)")
    parser.add_argument("--name", type=str, required=True, help="Dataset name (e.g. RawFLAIR)")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dataset_config.yaml (for label map). Optional.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    generate_dataset_json(args.id, args.name, cfg)


if __name__ == "__main__":
    main()
