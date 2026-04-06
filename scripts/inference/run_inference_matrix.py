#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
import sys

# ── Environment check ─────────────────────────────────────────────────────────
for _var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    if _var not in os.environ:
        sys.exit(f"ERROR: Set {_var} environment variable (standard nnUNet setup)")

REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.nnUNetTrainer_25D import nnUNetPredictor25D


@dataclass
class ModelSpec:
    key: str
    folder: Path
    checkpoint: str = "checkpoint_best.pth"


def expected_cases(images_ts: Path) -> int:
    return len(list(images_ts.glob("*_0000.nii.gz")))


def output_name(target_key: str, model_key: str) -> str:
    if model_key == "DS001_ResEncL_3D":
        return f"{target_key}_ResEncL_3D_TTA"
    if model_key == "DS002_ResEncL_3D":
        return f"{target_key}_DS002_ResEncL_3D_TTA"
    if model_key == "DS003_ResEncL_3D":
        return f"{target_key}_DS003_ResEncL_3D_TTA"
    if model_key == "DS003_CNN_3D":
        return f"{target_key}_DS003_CNN_3D_TTA"
    if model_key == "DS003_ResEncL_25D":
        return f"{target_key}_DS003_ResEncL_25D_TTA"
    if model_key == "DS003_CNN_25D":
        return f"{target_key}_DS003_CNN_25D_TTA"
    if model_key == "DS001_CNN_25D":
        return f"{target_key}_DS001_CNN_25D_TTA"
    return f"{target_key}_{model_key}_TTA"


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset nnU-Net inference matrix with TTA + softmax")
    parser.add_argument("--npp", type=int, default=8, help="num_processes_preprocessing")
    parser.add_argument("--nps", type=int, default=8, help="num_processes_segmentation_export")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = Path(os.environ["nnUNet_results"])
    pred_base = REPO_ROOT / "results" / "predictions"

    raw_base = Path(os.environ["nnUNet_raw"])
    targets = {
        "DS001": raw_base / "Dataset001_MSLesSeg" / "imagesTs",
        "DS002": raw_base / "Dataset002_WMH" / "imagesTs",
    }

    models = [
        ModelSpec("DS001_ResEncL_3D", base / "Dataset001_MSLesSeg" / "nnUNetTrainer_WandB__nnUNetResEncUNetLPlans__3d_fullres"),
        ModelSpec("DS001_CNN_25D", base / "Dataset001_MSLesSeg" / "nnUNetTrainer_25D__nnUNetPlans__3d_fullres"),
        ModelSpec("DS002_ResEncL_3D", base / "Dataset002_WMH" / "nnUNetTrainer_WandB__nnUNetResEncUNetLPlans__3d_fullres"),
        ModelSpec("DS003_ResEncL_3D", base / "Dataset003_Combined" / "nnUNetTrainer_WandB__nnUNetResEncUNetLPlans__3d_fullres"),
        ModelSpec("DS003_CNN_3D", base / "Dataset003_Combined" / "nnUNetTrainer_WandB__nnUNetPlans__3d_fullres"),
        ModelSpec("DS003_ResEncL_25D", base / "Dataset003_Combined" / "nnUNetTrainer_25D__nnUNetResEncUNetLPlans__3d_fullres"),
        ModelSpec("DS003_CNN_25D", base / "Dataset003_Combined" / "nnUNetTrainer_25D__nnUNetPlans__3d_fullres"),
    ]

    device = torch.device(args.device)

    for model in models:
        if not model.folder.exists():
            print(f"[SKIP] Missing model folder: {model.folder}")
            continue

        print("=" * 100)
        print(f"MODEL: {model.key}")
        print(f"FOLDER: {model.folder}")

        predictor_cls = nnUNetPredictor25D if "25D" in model.key else nnUNetPredictor
        predictor = predictor_cls(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        try:
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=str(model.folder),
                use_folds=(0,),
                checkpoint_name=model.checkpoint,
            )
        except Exception as e:
            print(f"[SKIP] failed to initialize {model.key}: {e}")
            continue

        for target_key, input_dir in targets.items():
            if not input_dir.exists():
                print(f"  [SKIP] Missing input dir: {input_dir}")
                continue

            out_dir = pred_base / output_name(target_key, model.key)
            out_dir.mkdir(parents=True, exist_ok=True)

            exp_n = expected_cases(input_dir)
            have_npz = len(list(out_dir.glob("*.npz")))
            have_nii = len(list(out_dir.glob("*.nii.gz")))
            if not args.overwrite and have_npz == exp_n and have_nii >= exp_n:
                print(f"  [OK] {target_key} already complete: {out_dir.name} ({have_npz} npz)")
                continue

            print(f"  [RUN] {target_key} -> {out_dir.name}")
            t0 = time.time()
            try:
                predictor.predict_from_files(
                    list_of_lists_or_source_folder=str(input_dir),
                    output_folder_or_list_of_truncated_output_files=str(out_dir),
                    save_probabilities=True,
                    overwrite=args.overwrite,
                    num_processes_preprocessing=args.npp,
                    num_processes_segmentation_export=args.nps,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0,
                )
            except Exception as e:
                print(f"  [SKIP] {target_key} with {model.key} failed: {e}")
                continue
            dt = time.time() - t0
            have_npz = len(list(out_dir.glob("*.npz")))
            print(f"  [DONE] {out_dir.name}: {have_npz}/{exp_n} npz in {dt/60:.1f} min")

    print("\nInference matrix complete.")


if __name__ == "__main__":
    main()
