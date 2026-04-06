#!/usr/bin/env python3
"""
run_xai_analysis.py
====================
Comprehensive XAI analysis for the best-performing nnU-Net models.
Implements three complementary explainability methods:

  1. Grad-CAM — gradient-based attribution (fast, architecture-aware)
  2. 3D RISE  — randomized input sampling for explanation (model-agnostic)
  3. Occlusion Sensitivity — systematic region masking (model-agnostic)

Usage:
    python run_xai_analysis.py                     # all methods, DS001 test set
    python run_xai_analysis.py --methods gradcam   # GradCAM only
    python run_xai_analysis.py --methods rise       # RISE only
    python run_xai_analysis.py --methods occlusion  # occlusion only
    python run_xai_analysis.py --max-subjects 5     # first 5 subjects
    python run_xai_analysis.py --models resencl     # ResEncL only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set env vars if not already set (AGENT_LESSONS #1)
os.environ.setdefault("nnUNet_raw", str(PROJECT_ROOT / "data" / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(PROJECT_ROOT / "data" / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(PROJECT_ROOT / "data" / "nnUNet_results"))

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("xai_analysis")


def make_brain_mask(flair: np.ndarray, dilate_voxels: int = 3) -> np.ndarray:
    """Create a binary brain foreground mask from a FLAIR volume.

    Thresholds slightly above zero (mean - 2*std of nonzero voxels),
    then dilates to avoid clipping at the brain edge.
    """
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

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
RAW_BASE = PROJECT_ROOT / "data" / "nnUNet_raw"
RESULTS_BASE = PROJECT_ROOT / "data" / "nnUNet_results"
OUTPUT_BASE = PROJECT_ROOT / "results" / "xai"

# Models to analyze — the best-performing models used in ensembles
MODELS = {
    "resencl": {
        "label": "ResEncL-3D (DS003)",
        "dataset_id": 3,
        "trainer": "nnUNetTrainer_WandB",
        "plans": "nnUNetResEncUNetLPlans",
        "config": "3d_fullres",
        "checkpoint": "checkpoint_best.pth",
    },
    "cnn": {
        "label": "CNN-3D (DS003)",
        "dataset_id": 3,
        "trainer": "nnUNetTrainer_WandB",
        "plans": "nnUNetPlans",
        "config": "3d_fullres",
        "checkpoint": "checkpoint_best.pth",
    },
}

# Evaluate on DS001 test set (22 subjects, 1mm isotropic, skull-stripped)
TEST_DATASET_ID = 1
TEST_DATASET_NAME = "MSLesSeg"


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════
def find_model_dir(dataset_id: int, trainer: str, plans: str, config: str) -> Path:
    """Find the nnU-Net model directory."""
    prefix = f"Dataset{dataset_id:03d}_"
    for d in RESULTS_BASE.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            model_dir = d / f"{trainer}__{plans}__{config}"
            if model_dir.is_dir():
                return model_dir
    raise FileNotFoundError(f"No model dir for DS{dataset_id:03d} {trainer}__{plans}__{config}")


def load_model(model_key: str, device: torch.device) -> torch.nn.Module:
    """Load a trained nnU-Net model using the predictor API."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    cfg = MODELS[model_key]
    model_dir = find_model_dir(cfg["dataset_id"], cfg["trainer"], cfg["plans"], cfg["config"])

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True,
        use_mirroring=False,  # no TTA — we want clean gradients
        device=device, verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_dir),
        use_folds=(0,),
        checkpoint_name=cfg["checkpoint"],
    )
    network = predictor.network
    network = network.to(device)
    network.eval()
    logger.info("  Loaded %s from %s", cfg["label"], model_dir.name)
    return network


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (2-channel: FLAIR + T1)
# ═══════════════════════════════════════════════════════════════════════════════
def get_test_subjects() -> List[Tuple[str, Path, Path, Optional[Path]]]:
    """Get (subject_id, flair_path, t1_path, label_path) for all test subjects."""
    ds_dir = RAW_BASE / f"Dataset{TEST_DATASET_ID:03d}_{TEST_DATASET_NAME}"
    img_dir = ds_dir / "imagesTs"
    lbl_dir = ds_dir / "labelsTs"

    flair_files = sorted(img_dir.glob("*_0000.nii.gz"))
    subjects = []
    for flair_path in flair_files:
        subject_id = flair_path.name.replace("_0000.nii.gz", "")
        t1_path = flair_path.parent / f"{subject_id}_0001.nii.gz"
        lbl_path = lbl_dir / f"{subject_id}.nii.gz"
        if t1_path.exists():
            subjects.append((subject_id, flair_path, t1_path,
                             lbl_path if lbl_path.exists() else None))
    return subjects


def load_volume(flair_path: Path, t1_path: Path, device: torch.device,
                patch_size: Tuple[int, ...] = (160, 192, 160),
                ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Tuple[slice, ...]]:
    """
    Load FLAIR + T1 as 2-channel input tensor, center-cropped to patch_size.
    Returns (tensor [1,2,D,H,W], raw_flair_np, raw_t1_np, crop_slices).
    """
    flair_nii = nib.load(str(flair_path))
    t1_nii = nib.load(str(t1_path))
    flair_data = np.asarray(flair_nii.dataobj, dtype=np.float32)
    t1_data = np.asarray(t1_nii.dataobj, dtype=np.float32)

    raw_flair = flair_data.copy()
    raw_t1 = t1_data.copy()

    # Z-score normalisation per channel (nnU-Net default for MRI)
    for data in [flair_data, t1_data]:
        mask = data > 0
        if mask.any():
            mean_val = data[mask].mean()
            std_val = data[mask].std()
            if std_val > 1e-8:
                data[mask] = (data[mask] - mean_val) / std_val
                data[~mask] = 0.0

    # Center crop/pad to patch_size for model compatibility
    crop_slices = []
    for dim, ps in enumerate(patch_size):
        s = flair_data.shape[dim]
        if s >= ps:
            start = (s - ps) // 2
            crop_slices.append(slice(start, start + ps))
        else:
            crop_slices.append(slice(0, s))
    crop_slices = tuple(crop_slices)

    flair_crop = flair_data[crop_slices]
    t1_crop = t1_data[crop_slices]

    # Pad if any dimension is smaller than patch_size
    pad_widths = []
    for dim, ps in enumerate(patch_size):
        s = flair_crop.shape[dim]
        if s < ps:
            pad_before = (ps - s) // 2
            pad_after = ps - s - pad_before
            pad_widths.append((pad_before, pad_after))
        else:
            pad_widths.append((0, 0))
    if any(pw != (0, 0) for pw in pad_widths):
        flair_crop = np.pad(flair_crop, pad_widths, mode="constant", constant_values=0)
        t1_crop = np.pad(t1_crop, pad_widths, mode="constant", constant_values=0)

    # Stack as (1, 2, D, H, W)
    tensor = torch.from_numpy(
        np.stack([flair_crop, t1_crop], axis=0)[np.newaxis]
    ).to(device)
    return tensor, raw_flair, raw_t1, crop_slices


def load_label(label_path: Optional[Path]) -> Optional[np.ndarray]:
    """Load ground truth label if available."""
    if label_path is None or not label_path.exists():
        return None
    return np.asarray(nib.load(str(label_path)).dataobj, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Grad-CAM (gradient-based, architecture-aware)
# ═══════════════════════════════════════════════════════════════════════════════
class _FeatureHook:
    """Forward/backward hook capturing activations and gradients."""
    def __init__(self):
        self.activations = None
        self.gradients = None

    def forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()


def discover_encoder_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """Find encoder stages (deepest first)."""
    candidates = []
    for name, module in model.named_modules():
        if "encoder" in name and "stages" in name:
            parts = name.split(".")
            if len(parts) == 3 and parts[2].isdigit():
                candidates.append((name, module))
    return list(reversed(candidates))  # deepest first


def compute_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor,
                    target_layer: torch.nn.Module, target_class: int = 1) -> np.ndarray:
    """Compute 3D Grad-CAM heatmap for a single layer. Returns (D,H,W) in [0,1]."""
    hook = _FeatureHook()
    fwd_handle = target_layer.register_forward_hook(hook.forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(hook.backward_hook)

    model.zero_grad()
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    if isinstance(output, (list, tuple)):
        output = output[0]  # full-res head (deep supervision)

    target_score = output[0, target_class].sum()
    target_score.backward(retain_graph=False)

    gradients = hook.gradients      # (1, F, d, h, w)
    activations = hook.activations  # (1, F, d, h, w)

    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="trilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    fwd_handle.remove()
    bwd_handle.remove()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)


def compute_multilayer_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor,
                                target_layers: List[torch.nn.Module],
                                target_class: int = 1, num_layers: int = 3) -> np.ndarray:
    """Compute Grad-CAM averaged across top-K encoder stages."""
    layers = target_layers[:num_layers]
    weights = [1.0 - i / (2 * len(layers)) for i in range(len(layers))]

    combined = None
    for w, layer in zip(weights, layers):
        cam = compute_gradcam(model, input_tensor, layer, target_class)
        if combined is None:
            combined = w * cam
        else:
            combined += w * cam

    cmin, cmax = combined.min(), combined.max()
    if cmax - cmin > 1e-8:
        combined = (combined - cmin) / (cmax - cmin)
    return combined.astype(np.float32)


def run_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor,
                num_layers: int = 3) -> np.ndarray:
    """Run multi-layer Grad-CAM on input. Returns (D,H,W) heatmap in [0,1]."""
    layers = discover_encoder_layers(model)
    target_modules = [mod for _, mod in layers[:num_layers]]
    if not target_modules:
        raise RuntimeError("No encoder layers found for Grad-CAM")

    with torch.enable_grad():
        cam = compute_multilayer_gradcam(model, input_tensor, target_modules,
                                         num_layers=num_layers)
    return cam


# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: 3D RISE (Randomized Input Sampling for Explanation)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_rise_masks(input_shape: Tuple[int, ...], n_masks: int = 500,
                        mask_res: int = 8, p_keep: float = 0.5,
                        device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generate random binary masks for RISE at low resolution, then upsample.

    The masks are generated at (mask_res, mask_res, mask_res) and bilinearly
    upsampled to the input spatial dimensions with random offsets.

    Returns tensor of shape (n_masks, 1, D, H, W) on device.
    """
    D, H, W = input_shape
    masks = []

    # Use slightly larger grid to allow random cropping
    grid_d = mask_res + 1
    grid_h = mask_res + 1
    grid_w = mask_res + 1

    for _ in range(n_masks):
        # Random binary mask at low resolution
        grid = (torch.rand(1, 1, grid_d, grid_h, grid_w) < p_keep).float()

        # Upsample to slightly larger than input (for random crop offset)
        up_d = int(np.ceil(D / mask_res) * (mask_res + 1))
        up_h = int(np.ceil(H / mask_res) * (mask_res + 1))
        up_w = int(np.ceil(W / mask_res) * (mask_res + 1))

        upsampled = F.interpolate(grid, size=(up_d, up_h, up_w),
                                  mode="trilinear", align_corners=False)

        # Random crop to input size
        d_off = np.random.randint(0, max(1, up_d - D))
        h_off = np.random.randint(0, max(1, up_h - H))
        w_off = np.random.randint(0, max(1, up_w - W))

        mask = upsampled[:, :, d_off:d_off+D, h_off:h_off+H, w_off:w_off+W]
        masks.append(mask)

    return torch.cat(masks, dim=0).to(device)  # (n_masks, 1, D, H, W)


def run_rise(model: torch.nn.Module, input_tensor: torch.Tensor,
             n_masks: int = 500, mask_res: int = 8, p_keep: float = 0.5,
             batch_size: int = 4) -> np.ndarray:
    """
    Compute 3D RISE saliency map.

    For each random mask, multiply the input, forward pass, measure lesion
    confidence. The saliency map is the weighted average of masks by confidence.

    Returns (D, H, W) heatmap in [0, 1].
    """
    device = input_tensor.device
    spatial = input_tensor.shape[2:]  # (D, H, W)

    logger.info("    Generating %d RISE masks at resolution %d...", n_masks, mask_res)
    masks = generate_rise_masks(spatial, n_masks, mask_res, p_keep, device)

    saliency = torch.zeros(spatial, device=device)
    total_weight = 0.0

    with torch.no_grad():
        for i in range(0, n_masks, batch_size):
            batch_masks = masks[i:i+batch_size]  # (B, 1, D, H, W)
            # Mask both channels equally
            masked_input = input_tensor * batch_masks  # broadcast over channels

            # Process one at a time to avoid OOM on large 3D volumes
            for j in range(batch_masks.shape[0]):
                single_input = masked_input[j:j+1]  # (1, 2, D, H, W)
                output = model(single_input)
                if isinstance(output, (list, tuple)):
                    output = output[0]

                # Lesion confidence: mean probability of class 1
                prob = torch.softmax(output, dim=1)
                confidence = prob[0, 1].mean().item()  # scalar

                saliency += confidence * batch_masks[j, 0]
                total_weight += confidence

            if (i + batch_size) % 100 == 0:
                logger.info("    RISE progress: %d/%d masks", min(i + batch_size, n_masks), n_masks)

    if total_weight > 0:
        saliency /= total_weight

    cam = saliency.cpu().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Occlusion Sensitivity (model-agnostic, systematic)
# ═══════════════════════════════════════════════════════════════════════════════
def run_occlusion(model: torch.nn.Module, input_tensor: torch.Tensor,
                  patch_size: int = 16, stride: int = 8) -> np.ndarray:
    """
    Compute 3D occlusion sensitivity map.

    Slides a 3D cube across the volume, zeroing out each patch and measuring
    the drop in lesion prediction confidence. High drop = important region.

    Returns (D, H, W) heatmap in [0, 1].
    """
    device = input_tensor.device
    D, H, W = input_tensor.shape[2:]

    # Get baseline prediction confidence
    with torch.no_grad():
        baseline_output = model(input_tensor)
        if isinstance(baseline_output, (list, tuple)):
            baseline_output = baseline_output[0]
        baseline_prob = torch.softmax(baseline_output, dim=1)
        baseline_conf = baseline_prob[0, 1].mean().item()

    sensitivity = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    total_patches = (
        len(range(0, D - patch_size + 1, stride)) *
        len(range(0, H - patch_size + 1, stride)) *
        len(range(0, W - patch_size + 1, stride))
    )
    logger.info("    Occlusion: %d patches (size=%d, stride=%d)", total_patches, patch_size, stride)

    patch_idx = 0
    with torch.no_grad():
        for d in range(0, D - patch_size + 1, stride):
            for h in range(0, H - patch_size + 1, stride):
                for w in range(0, W - patch_size + 1, stride):
                    # Zero out a patch (both channels)
                    occluded = input_tensor.clone()
                    occluded[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size] = 0.0

                    output = model(occluded)
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    prob = torch.softmax(output, dim=1)
                    occluded_conf = prob[0, 1].mean().item()

                    # Drop in confidence = importance of this patch
                    drop = max(0.0, baseline_conf - occluded_conf)
                    sensitivity[d:d+patch_size, h:h+patch_size, w:w+patch_size] += drop
                    count[d:d+patch_size, h:h+patch_size, w:w+patch_size] += 1.0

                    patch_idx += 1
                    if patch_idx % 200 == 0:
                        logger.info("    Occlusion progress: %d/%d patches", patch_idx, total_patches)

    # Average overlapping regions
    mask = count > 0
    sensitivity[mask] /= count[mask]

    # Normalise to [0, 1]
    s_min, s_max = sensitivity.min(), sensitivity.max()
    if s_max - s_min > 1e-8:
        sensitivity = (sensitivity - s_min) / (s_max - s_min)
    return sensitivity.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════
def find_best_slices(label: Optional[np.ndarray], flair: np.ndarray,
                     n_slices: int = 5) -> List[int]:
    """Pick axial slices with most lesion content, or evenly spaced if no label."""
    D = flair.shape[2] if flair.ndim == 3 else flair.shape[0]
    if label is not None and label.sum() > 0:
        # Sum lesion voxels per axial slice
        if label.ndim == 3:
            sums = label.sum(axis=(0, 1))  # per z-slice
        else:
            sums = label.sum(axis=(1, 2))
        # Pick slices with highest lesion count
        indices = np.argsort(sums)[::-1][:n_slices]
        return sorted(indices)
    else:
        start, end = int(D * 0.2), int(D * 0.8)
        return list(np.linspace(start, end, num=n_slices, dtype=int))


def save_comparison_panel(flair: np.ndarray, label: Optional[np.ndarray],
                          heatmaps: Dict[str, np.ndarray],
                          subject_id: str, slice_indices: List[int],
                          output_dir: Path) -> Path:
    """
    Save a multi-row comparison panel:
    Columns: FLAIR | Ground Truth | method1 | method2 | ...
    Rows: one per axial slice
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available, skipping PNG export")
        return output_dir

    n_slices = len(slice_indices)
    has_label = label is not None
    n_cols = 2 + len(heatmaps) if has_label else 1 + len(heatmaps)
    method_names = list(heatmaps.keys())

    fig, axes = plt.subplots(n_slices, n_cols, figsize=(4 * n_cols, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row, sl in enumerate(slice_indices):
        col = 0

        # FLAIR
        flair_sl = flair[:, :, sl].T
        axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
        if row == 0:
            axes[row, col].set_title("FLAIR", fontsize=11)
        axes[row, col].set_ylabel(f"z={sl}", fontsize=9)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        col += 1

        # Ground truth
        if has_label:
            axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
            lbl_sl = label[:, :, sl].T.astype(float)
            axes[row, col].imshow(lbl_sl, cmap="Reds", alpha=0.5, origin="lower",
                                  vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title("Ground Truth", fontsize=11)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            col += 1

        # Heatmaps
        for method_name in method_names:
            hmap = heatmaps[method_name]
            hmap_sl = hmap[:, :, sl].T

            axes[row, col].imshow(flair_sl, cmap="gray", origin="lower")
            im = axes[row, col].imshow(hmap_sl, cmap="jet", alpha=0.5,
                                        origin="lower", vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(method_name, fontsize=11)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            col += 1

    fig.suptitle(f"{subject_id} — XAI Comparison", fontsize=14, y=1.01)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{subject_id}_xai_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def save_heatmap_nifti(heatmap: np.ndarray, reference_path: Path,
                       output_path: Path) -> None:
    """Save heatmap as NIfTI with same affine as reference."""
    ref = nib.load(str(reference_path))
    out_nii = nib.Nifti1Image(heatmap, affine=ref.affine, header=ref.header)
    out_nii.set_data_dtype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nii, str(output_path))


# ═══════════════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="XAI analysis for nnU-Net models")
    parser.add_argument("--methods", nargs="+", default=["gradcam", "rise", "occlusion"],
                        choices=["gradcam", "rise", "occlusion"],
                        help="XAI methods to run (default: all three)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Models to analyze (default: all)")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of test subjects")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Encoder layers for Grad-CAM (default: 3)")
    parser.add_argument("--rise-masks", type=int, default=500,
                        help="Number of RISE masks (default: 500)")
    parser.add_argument("--rise-res", type=int, default=8,
                        help="RISE mask resolution (default: 8)")
    parser.add_argument("--occ-patch", type=int, default=16,
                        help="Occlusion patch size in voxels (default: 16)")
    parser.add_argument("--occ-stride", type=int, default=8,
                        help="Occlusion stride (default: 8)")
    parser.add_argument("--n-slices", type=int, default=5,
                        help="PNG slices per subject (default: 5)")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG export")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Methods: %s", ", ".join(args.methods))
    logger.info("Models: %s", ", ".join(args.models))

    # Get test subjects
    subjects = get_test_subjects()
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    logger.info("Test subjects: %d (DS%03d)", len(subjects), TEST_DATASET_ID)

    for model_key in args.models:
        model_cfg = MODELS[model_key]
        logger.info("\n" + "═" * 70)
        logger.info("  Model: %s", model_cfg["label"])
        logger.info("═" * 70)

        model = load_model(model_key, device)
        model_out = OUTPUT_BASE / model_key

        for subj_idx, (subject_id, flair_path, t1_path, label_path) in enumerate(subjects):
            logger.info("\n  [%d/%d] Subject: %s", subj_idx + 1, len(subjects), subject_id)

            # Load data
            input_tensor, raw_flair, raw_t1, crop_slices = load_volume(
                flair_path, t1_path, device)
            label = load_label(label_path)
            if label is not None:
                label = label[crop_slices]  # crop label to match
            logger.info("    Volume shape: %s", list(input_tensor.shape))

            # Brain mask for removing padding artifacts from heatmaps
            flair_crop = raw_flair[crop_slices]
            brain_mask = make_brain_mask(flair_crop).astype(np.float32)

            heatmaps: Dict[str, np.ndarray] = {}

            # --- Grad-CAM ---
            if "gradcam" in args.methods:
                logger.info("    Running Grad-CAM (%d layers)...", args.num_layers)
                t0 = time.time()
                cam = run_gradcam(model, input_tensor, num_layers=args.num_layers)
                dt = time.time() - t0
                logger.info("    Grad-CAM done (%.1fs)", dt)
                cam *= brain_mask  # remove non-brain padding artifacts
                heatmaps["Grad-CAM"] = cam

                save_heatmap_nifti(cam, flair_path,
                                   model_out / "nifti" / f"{subject_id}_gradcam.nii.gz")

            # --- RISE ---
            if "rise" in args.methods:
                logger.info("    Running 3D RISE (%d masks, res=%d)...",
                            args.rise_masks, args.rise_res)
                t0 = time.time()
                rise_map = run_rise(model, input_tensor,
                                    n_masks=args.rise_masks,
                                    mask_res=args.rise_res)
                dt = time.time() - t0
                logger.info("    RISE done (%.1fs)", dt)
                rise_map *= brain_mask  # remove non-brain padding artifacts
                heatmaps["3D-RISE"] = rise_map

                save_heatmap_nifti(rise_map, flair_path,
                                   model_out / "nifti" / f"{subject_id}_rise.nii.gz")

            # --- Occlusion Sensitivity ---
            if "occlusion" in args.methods:
                logger.info("    Running Occlusion Sensitivity (patch=%d, stride=%d)...",
                            args.occ_patch, args.occ_stride)
                t0 = time.time()
                occ_map = run_occlusion(model, input_tensor,
                                         patch_size=args.occ_patch,
                                         stride=args.occ_stride)
                dt = time.time() - t0
                logger.info("    Occlusion done (%.1fs)", dt)
                occ_map *= brain_mask  # remove non-brain padding artifacts
                heatmaps["Occlusion"] = occ_map

                save_heatmap_nifti(occ_map, flair_path,
                                   model_out / "nifti" / f"{subject_id}_occlusion.nii.gz")

            # --- Comparison panel ---
            if not args.no_png and heatmaps and HAS_MPL:
                slices = find_best_slices(label, flair_crop, n_slices=args.n_slices)
                png_path = save_comparison_panel(
                    flair_crop, label, heatmaps, subject_id, slices,
                    model_out / "png",
                )
                logger.info("    Saved comparison panel: %s", png_path)

            # Cleanup
            del input_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("\n" + "═" * 70)
    logger.info("XAI analysis complete. Output: %s", OUTPUT_BASE)
    logger.info("═" * 70)


if __name__ == "__main__":
    main()
