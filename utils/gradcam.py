"""
utils/gradcam.py
================
GradCAM (Gradient-weighted Class Activation Mapping) for nnU-Net v2 models.

Provides visual explanations of which spatial regions drive the model's
lesion predictions — critical for clinical trust and XAI reporting.

Supports both the standard nnU-Net encoder and the ResEncL (Residual Encoder)
architecture by auto-detecting the encoder structure from the loaded checkpoint.

References
----------
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", ICCV 2017.
- Adapted for 3-D volumetric segmentation (no global-average-pool; we use
  spatial-mean of gradients directly).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    import nibabel as nib
except ImportError:
    nib = None  # type: ignore[assignment]

logger = logging.getLogger("nnunet_pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Hook helper
# ──────────────────────────────────────────────────────────────────────────────
class _FeatureHook:
    """Forward/backward hook that stores activations and gradients."""

    def __init__(self) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

    def forward_hook(self, module: torch.nn.Module, inp, out) -> None:  # noqa: ANN001
        self.activations = out.detach()

    def backward_hook(self, module: torch.nn.Module, grad_in, grad_out) -> None:  # noqa: ANN001
        self.gradients = grad_out[0].detach()


# ──────────────────────────────────────────────────────────────────────────────
# Layer auto-discovery
# ──────────────────────────────────────────────────────────────────────────────
def discover_encoder_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """
    Walk the model tree and return a list of ``(name, module)`` tuples for
    likely target layers (last Conv/Block in each encoder stage).

    Heuristic:
        - For the standard nnU-Net:  ``encoder.stages.{N}`` (ConvBlocks)
        - For ResEncL:               ``encoder.stages.{N}`` (ResBlocks)

    The returned list is ordered deepest-first (best for GradCAM — deepest
    has the most semantic information; shallowest preserves spatial detail).
    """
    candidates: List[Tuple[str, torch.nn.Module]] = []

    for name, module in model.named_modules():
        # nnU-Net v2 encoder stages are Sequential containers named
        # "encoder.stages.0", "encoder.stages.1", …
        if "encoder" in name and "stages" in name:
            parts = name.split(".")
            # We want exactly "encoder.stages.N" (depth 3)
            if len(parts) == 3 and parts[2].isdigit():
                candidates.append((name, module))

    if not candidates:
        # Fallback: grab any module whose name contains 'encoder' and ends
        # with a Conv3d or BatchNorm3d
        for name, module in model.named_modules():
            if "encoder" in name.lower() and isinstance(
                module, (torch.nn.Conv3d, torch.nn.BatchNorm3d)
            ):
                candidates.append((name, module))

    # Deepest stage first
    candidates = list(reversed(candidates))
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Core GradCAM computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int = 1,
) -> np.ndarray:
    """
    Compute a 3-D GradCAM heatmap for a single input volume.

    Parameters
    ----------
    model         : The nnU-Net network (already in eval mode, on correct device).
    input_tensor  : Shape ``(1, C, D, H, W)`` — a single preprocessed volume.
    target_layer  : The encoder layer to hook into.
    target_class  : Label index to explain (default 1 = lesion).

    Returns
    -------
    cam : np.ndarray of shape ``(D, H, W)`` with values in [0, 1].
    """
    hook = _FeatureHook()
    fwd_handle = target_layer.register_forward_hook(hook.forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(hook.backward_hook)

    model.zero_grad()
    input_tensor.requires_grad_(True)

    # ── Forward pass ─────────────────────────────────────────────────────
    output = model(input_tensor)  # (1, num_classes, D, H, W)

    if isinstance(output, (list, tuple)):
        # nnU-Net deep-supervision returns a list; take the full-res head
        output = output[0]

    # ── Backward pass on the target class ────────────────────────────────
    # We sum over all spatial locations for the target class channel to get
    # a scalar loss whose gradient tells us "what made the model predict
    # class=target_class at *any* location".
    target_score = output[0, target_class].sum()
    target_score.backward(retain_graph=False)

    # ── Weighted combination of feature maps ─────────────────────────────
    gradients = hook.gradients  # (1, F, d, h, w)
    activations = hook.activations  # (1, F, d, h, w)

    # Global-average-pool the gradients over spatial dims → channel weights
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, F, 1, 1, 1)

    # Weighted sum of activation maps
    cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, d, h, w)

    # ReLU — we only care about features with positive influence
    cam = F.relu(cam)

    # ── Upsample to input resolution ─────────────────────────────────────
    spatial = input_tensor.shape[2:]  # (D, H, W)
    cam = F.interpolate(cam, size=spatial, mode="trilinear", align_corners=False)

    # Normalise to [0, 1]
    cam = cam.squeeze().cpu().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # Clean up hooks
    fwd_handle.remove()
    bwd_handle.remove()

    return cam.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-layer GradCAM (average across encoder stages for a richer map)
# ──────────────────────────────────────────────────────────────────────────────
def compute_multilayer_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layers: List[torch.nn.Module],
    target_class: int = 1,
    layer_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Compute GradCAM from multiple encoder stages and merge them via
    weighted average.  Deeper layers contribute more semantic context;
    shallower layers add spatial precision.

    Parameters
    ----------
    model          : The nnU-Net network.
    input_tensor   : (1, C, D, H, W).
    target_layers  : List of encoder modules to hook (deepest first).
    target_class   : Label index to explain.
    layer_weights  : Per-layer importance (default: linearly decreasing
                     weight from deepest to shallowest).

    Returns
    -------
    cam : np.ndarray of shape (D, H, W) in [0, 1].
    """
    n = len(target_layers)
    if layer_weights is None:
        # More weight on deeper (more semantic) layers
        layer_weights = [1.0 - i / (2 * n) for i in range(n)]

    cams: List[np.ndarray] = []
    for layer in target_layers:
        cam = compute_gradcam(model, input_tensor, layer, target_class)
        cams.append(cam)

    combined = np.zeros_like(cams[0])
    for w, c in zip(layer_weights, cams):
        combined += w * c

    # Re-normalise
    cmin, cmax = combined.min(), combined.max()
    if cmax - cmin > 1e-8:
        combined = (combined - cmin) / (cmax - cmin)

    return combined.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Save GradCAM as NIfTI
# ──────────────────────────────────────────────────────────────────────────────
def save_gradcam_nifti(
    cam: np.ndarray,
    reference_nifti: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """
    Save a GradCAM heatmap as a NIfTI file, inheriting the affine and header
    from *reference_nifti* so it aligns in any medical viewer (e.g. ITK-SNAP).
    """
    if nib is None:
        raise ImportError("nibabel is required to save NIfTI GradCAM maps.")

    ref = nib.load(str(reference_nifti))
    out_nii = nib.Nifti1Image(cam, affine=ref.affine, header=ref.header)
    out_nii.set_data_dtype(np.float32)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nii, str(out_path))
    logger.info("  Saved GradCAM → %s", out_path)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Save representative 2-D PNG slices (axial/sagittal/coronal)
# ──────────────────────────────────────────────────────────────────────────────
def save_gradcam_slices_png(
    cam: np.ndarray,
    image_data: np.ndarray,
    output_dir: Path,
    subject_name: str,
    num_slices: int = 5,
) -> List[Path]:
    """
    Overlay GradCAM on axial FLAIR slices and save as PNGs for quick review.

    Picks *num_slices* evenly-spaced axial slices from the middle 60% of the
    volume (where lesions are most likely).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping PNG export.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    D = cam.shape[2] if cam.ndim == 3 else cam.shape[0]
    # Middle 60%
    start = int(D * 0.2)
    end = int(D * 0.8)
    indices = np.linspace(start, end, num=num_slices, dtype=int)

    for idx in indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Determine slicing axis (axial = last dim for RAS)
        img_slice = image_data[:, :, idx] if image_data.ndim == 3 else image_data[:, :, idx, 0]
        cam_slice = cam[:, :, idx]

        # 1) Raw FLAIR
        axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
        axes[0].set_title("FLAIR")
        axes[0].axis("off")

        # 2) GradCAM heatmap
        axes[1].imshow(cam_slice.T, cmap="jet", origin="lower", vmin=0, vmax=1)
        axes[1].set_title("GradCAM")
        axes[1].axis("off")

        # 3) Overlay
        axes[2].imshow(img_slice.T, cmap="gray", origin="lower")
        axes[2].imshow(cam_slice.T, cmap="jet", alpha=0.4, origin="lower", vmin=0, vmax=1)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        fig.suptitle(f"{subject_name} — slice {idx}", fontsize=12)
        fig.tight_layout()
        png_path = output_dir / f"{subject_name}_slice{idx:03d}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(png_path)

    return saved
