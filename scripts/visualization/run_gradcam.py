#!/usr/bin/env python3
"""
run_gradcam.py
==============
Generate GradCAM heatmaps for every trained nnU-Net model on the held-out
test set.  Produces NIfTI files (for medical viewers) and optional PNG
slice panels for quick visual review.

For each experiment in the ablation matrix (2 datasets x 2 architectures),
this script:
  1. Loads the trained nnU-Net checkpoint (fold 0 by default, or best).
  2. Auto-discovers encoder layers.
  3. Runs multi-layer GradCAM on every test subject.
  4. Saves heatmaps as NIfTI + optional PNGs.

Usage:
    python run_gradcam.py                                 # default config
    python run_gradcam.py --fold 0                        # specific fold
    python run_gradcam.py --config config/dataset_config.yaml
    python run_gradcam.py --dataset-id 500 --plans nnUNetPlans  # single model
    python run_gradcam.py --no-png                        # skip PNG export
    python run_gradcam.py --num-layers 3                  # top-3 encoder stages
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_helpers import (
    get_nnunet_env_paths,
    glob_nifti,
    load_config,
    logger,
    setup_logging,
    strip_nifti_ext,
)
from utils.gradcam import (
    compute_gradcam,
    compute_multilayer_gradcam,
    discover_encoder_layers,
    save_gradcam_nifti,
    save_gradcam_slices_png,
)

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required:  pip install nibabel")


# ──────────────────────────────────────────────────────────────────────────────
# nnU-Net model loader
# ──────────────────────────────────────────────────────────────────────────────
def load_nnunet_model(
    dataset_id: int,
    plans_name: str,
    configuration: str,
    trainer: str,
    fold: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a trained nnU-Net v2 model from its checkpoint directory.

    Uses the nnU-Net v2 API to rebuild the network from the stored plans
    and load the trained weights.
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    _, _, results_base = get_nnunet_env_paths()

    # nnU-Net stores models at:
    #   nnUNet_results/DatasetXXX_Name/<trainer>__<plans>__<config>/fold_N/
    # The predictor handles the resolution of these paths.
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  # disable TTA for GradCAM — we want clean gradients
        device=device,
        verbose=False,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(
            _find_model_dir(results_base, dataset_id, trainer, plans_name, configuration)
        ),
        use_folds=(fold,),
        checkpoint_name="checkpoint_final.pth",
    )

    # The actual PyTorch network lives inside the predictor
    network = predictor.network
    network.eval()
    return network


def _find_model_dir(
    results_base: Path,
    dataset_id: int,
    trainer: str,
    plans_name: str,
    configuration: str,
) -> Path:
    """
    Locate the nnU-Net model directory matching the given parameters.

    Convention:  nnUNet_results/DatasetXXX_<Name>/<Trainer>__<Plans>__<Config>/
    """
    # Find the dataset folder (we don't know the name suffix for sure)
    ds_prefix = f"Dataset{dataset_id:03d}_"
    candidates = [d for d in results_base.iterdir() if d.is_dir() and d.name.startswith(ds_prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"No results directory found for dataset {dataset_id} in {results_base}"
        )
    ds_dir = candidates[0]

    # Trainer__Plans__Config
    model_dir_name = f"{trainer}__{plans_name}__{configuration}"
    model_dir = ds_dir / model_dir_name
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Available: {[d.name for d in ds_dir.iterdir() if d.is_dir()]}"
        )
    return model_dir


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing helper (minimal — resamples to model's expected spacing)
# ──────────────────────────────────────────────────────────────────────────────
def load_and_preprocess_volume(
    nifti_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load a NIfTI, normalise to zero-mean/unit-var, and return as a
    (1, 1, D, H, W) tensor plus the raw numpy data (for PNG overlays).

    Note: For exact reproduction of nnU-Net's preprocessing (resampling,
    clipping, normalisation), you would use the nnU-Net preprocessor.
    This simplified version is sufficient for GradCAM visualisation
    because the heatmap is relative, not absolute.
    """
    nii = nib.load(str(nifti_path))
    data = np.asarray(nii.dataobj, dtype=np.float32)

    raw_data = data.copy()

    # Z-score normalisation (nnU-Net default for MRI)
    mask = data > 0
    if mask.any():
        mean_val = data[mask].mean()
        std_val = data[mask].std()
        if std_val > 1e-8:
            data = (data - mean_val) / std_val

    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    return tensor, raw_data


def _make_brain_mask(flair: np.ndarray, dilate_voxels: int = 3) -> np.ndarray:
    """Create a binary brain foreground mask from a FLAIR volume."""
    from scipy import ndimage as _ndi
    nonzero = flair[flair > 0]
    if nonzero.size == 0:
        return np.ones(flair.shape, dtype=bool)
    thr = max(0, nonzero.mean() - 2.0 * nonzero.std())
    mask = flair > thr
    if dilate_voxels > 0:
        struct = _ndi.generate_binary_structure(mask.ndim, 1)
        mask = _ndi.binary_dilation(mask, struct, iterations=dilate_voxels)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────────
def run_gradcam_for_model(
    dataset_id: int,
    dataset_name: str,
    plans_name: str,
    plan_tag: str,  # e.g. "CNN" or "ResEncL"
    configuration: str,
    trainer: str,
    fold: int,
    device: torch.device,
    output_base: Path,
    export_png: bool = True,
    num_slices_png: int = 5,
    num_layers: int = 3,
) -> int:
    """
    Generate GradCAM heatmaps for all test subjects of one model.

    Returns the number of subjects processed.
    """
    raw_base, _, _ = get_nnunet_env_paths()

    # Locate test images
    ds_folder = None
    ds_prefix = f"Dataset{dataset_id:03d}_"
    for d in raw_base.iterdir():
        if d.is_dir() and d.name.startswith(ds_prefix):
            ds_folder = d
            break
    if ds_folder is None:
        logger.warning("Dataset folder for %d not found — skipping.", dataset_id)
        return 0

    test_dir = ds_folder / "imagesTs"
    if not test_dir.is_dir() or not list(test_dir.glob("*.nii.gz")):
        logger.warning("No test images in %s — skipping.", test_dir)
        return 0

    test_files = glob_nifti(test_dir)
    logger.info(
        "  DS%d %s: %d test subjects, fold=%d, layers=%d",
        dataset_id, plan_tag, len(test_files), fold, num_layers,
    )

    # Load model
    logger.info("  Loading model checkpoint...")
    try:
        model = load_nnunet_model(
            dataset_id, plans_name, configuration, trainer, fold, device
        )
    except (FileNotFoundError, Exception) as e:
        logger.error("  Could not load model: %s", e)
        return 0

    # Discover target layers
    all_layers = discover_encoder_layers(model)
    logger.info("  Discovered %d encoder stages", len(all_layers))
    target_layers = [mod for _, mod in all_layers[:num_layers]]

    if not target_layers:
        logger.error("  No encoder layers found — cannot compute GradCAM.")
        return 0

    # Output directories
    exp_tag = f"DS{dataset_id}_{plan_tag}"
    nifti_out = output_base / exp_tag / "nifti"
    png_out = output_base / exp_tag / "png"
    nifti_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for tf in test_files:
        subject = strip_nifti_ext(tf.name).replace("_0000", "")
        logger.info("    [%d/%d] %s", count + 1, len(test_files), subject)

        # Load volume
        input_tensor, raw_data = load_and_preprocess_volume(tf, device)

        # Compute GradCAM
        with torch.enable_grad():
            if len(target_layers) == 1:
                cam = compute_gradcam(model, input_tensor, target_layers[0], target_class=1)
            else:
                cam = compute_multilayer_gradcam(
                    model, input_tensor, target_layers, target_class=1
                )

        # Apply brain mask to remove non-brain padding artifacts
        brain_mask = _make_brain_mask(raw_data).astype(np.float32)
        cam *= brain_mask

        # Save NIfTI
        save_gradcam_nifti(cam, tf, nifti_out / f"{subject}_gradcam.nii.gz")

        # Save PNG slices
        if export_png:
            save_gradcam_slices_png(
                cam, raw_data, png_out / subject, subject, num_slices=num_slices_png
            )

        count += 1
        # Free GPU memory
        del input_tensor
        torch.cuda.empty_cache() if device.type == "cuda" else None

    return count


def main(config_path: Optional[str] = None) -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="GradCAM XAI for nnU-Net models.")
    parser.add_argument("--config", type=str, default=config_path)
    parser.add_argument("--fold", type=int, default=0, help="Fold checkpoint to load (default: 0)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--dataset-id", type=int, default=None, help="Run for single dataset only")
    parser.add_argument("--plans", type=str, default=None, help="Run for single plans only")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG slice export")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of encoder stages for multi-layer GradCAM")
    parser.add_argument("--num-slices", type=int, default=5, help="Number of PNG slices per subject")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, _, results_base = get_nnunet_env_paths()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Using device: %s", device)

    xai_cfg = cfg.get("xai", {}).get("gradcam", {})
    num_layers = args.num_layers or xai_cfg.get("num_encoder_layers", 3)
    num_slices = args.num_slices or xai_cfg.get("num_slices_png", 5)
    export_png = not args.no_png and xai_cfg.get("export_png", True)
    trainer = "nnUNetTrainer"
    configuration = "3d_fullres"

    output_base = results_base / "gradcam"
    output_base.mkdir(parents=True, exist_ok=True)

    # Define the experiment matrix
    experiments = [
        (cfg["nnunet_dataset_ids"]["raw_flair"],           cfg["nnunet_dataset_names"]["raw_flair"],           "nnUNetPlans",            "CNN"),
        (cfg["nnunet_dataset_ids"]["raw_flair"],           cfg["nnunet_dataset_names"]["raw_flair"],           "nnUNetResEncUNetLPlans", "ResEncL"),
        (cfg["nnunet_dataset_ids"]["skull_stripped_flair"], cfg["nnunet_dataset_names"]["skull_stripped_flair"], "nnUNetPlans",            "CNN"),
        (cfg["nnunet_dataset_ids"]["skull_stripped_flair"], cfg["nnunet_dataset_names"]["skull_stripped_flair"], "nnUNetResEncUNetLPlans", "ResEncL"),
    ]

    # Filter if user requested a single run
    if args.dataset_id is not None:
        experiments = [e for e in experiments if e[0] == args.dataset_id]
    if args.plans is not None:
        experiments = [e for e in experiments if e[2] == args.plans]

    total_subjects = 0
    for ds_id, ds_name, plans, tag in experiments:
        logger.info("═" * 60)
        logger.info("  GradCAM: Dataset %d (%s) — %s", ds_id, ds_name, tag)
        logger.info("═" * 60)

        n = run_gradcam_for_model(
            dataset_id=ds_id,
            dataset_name=ds_name,
            plans_name=plans,
            plan_tag=tag,
            configuration=configuration,
            trainer=trainer,
            fold=args.fold,
            device=device,
            output_base=output_base,
            export_png=export_png,
            num_slices_png=num_slices,
            num_layers=num_layers,
        )
        total_subjects += n

    logger.info("GradCAM complete: %d subjects processed across %d experiments.",
                total_subjects, len(experiments))
    logger.info("Output: %s", output_base)


if __name__ == "__main__":
    main()
