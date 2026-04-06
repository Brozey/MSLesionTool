"""
nnUNetTrainer_25D.py
====================
True 2.5D trainer for nnU-Net v2.

Uses 3d_fullres preprocessed data but applies 2D convolutions with K adjacent
slices stacked as input channels. Predicts segmentation for the center slice.

This provides inter-slice context (like 3D) with the computational efficiency
of 2D convolutions and large batch sizes.

Architecture:
    - 2D U-Net with K × C input channels (K adjacent slices × C modalities)
    - Standard 2D nnU-Net architecture (from the 2d plans configuration)
    - Output: segmentation for center slice only

Data flow:
    Training:  Load 3D volume → extract K-slice window → stack as channels → 2D augmentation → predict center slice
    Inference: For each target slice z → extract [z-k..z+k] → stack → 2D prediction → assign to slice z

Prerequisites:
    - Both 2d AND 3d_fullres must be planned:
      nnUNetv2_plan_and_preprocess -d DATASET --verify_dataset_integrity
    - 3d_fullres preprocessing must exist (for the 3D volumes)
    - 2d plans must exist (for the 2D architecture definition)

Usage:
    nnUNetv2_train DATASET 3d_fullres 0 -tr nnUNetTrainer_25D [--c]
    nnUNetv2_train DATASET 3d_fullres 0 -tr nnUNetTrainer_25D -p nnUNetResEncUNetLPlans [--c]
"""
from __future__ import annotations

import itertools
import warnings
from copy import deepcopy
from threading import Thread, Event
from queue import Queue, Full, Empty
from time import sleep
from typing import List, Tuple, Union

import numpy as np
import torch
import kornia.filters
from torch import autocast
from torch.nn import functional as F
from scipy.optimize import curve_fit
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset, infer_dataset_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy

from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

import multiprocessing
from nnunetv2.configuration import default_num_processes
from os.path import join

try:
    from batchviewer import maybe_mkdir_p
except ImportError:
    import os
    def maybe_mkdir_p(path):
        os.makedirs(path, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RAM-Preloaded Dataset Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

# Axis permutations for multi-view 2.5D.
# Preprocessed volumes are [C, D, H, W].  These permutations move the target
# slicing axis into position 1 (the D slot) so the DataLoader's thin-slab
# cropping always operates along the correct anatomical direction.
_SLICE_AXIS_PERM = {
    0: None,            # Axial:    D is already axis 1 → no transpose
    1: (0, 2, 1, 3),    # Coronal:  swap D↔H  → slice along H
    2: (0, 3, 2, 1),    # Sagittal: swap D↔W  → slice along W
}
_SLICE_AXIS_NAME = {0: 'axial', 1: 'coronal', 2: 'sagittal'}


class PreloadedDataset(nnUNetBaseDataset):
    """
    Wraps an nnUNetBaseDataset and eagerly decompresses ALL volumes into RAM
    at construction time.  Subsequent load_case() calls return cached numpy
    arrays instantly — zero disk I/O, zero blosc2 decompression.

    Memory cost: ~6 GB for 153 cases of DS003 (trivial on 92 GB machine).
    Speed gain:  eliminates the per-sample decompression that dominates CPU
                 data-loading time for threaded prefetching.

    If slice_axis != 0, volumes are transposed so the target slicing axis
    becomes the depth dimension (axis 1 in [C, D, H, W]).
    """

    def __init__(self, base_dataset: nnUNetBaseDataset, log_fn=None, slice_axis: int = 0):
        # Don't call super().__init__ — we copy attributes from the base.
        self.source_folder = base_dataset.source_folder
        self.folder_with_segs_from_previous_stage = base_dataset.folder_with_segs_from_previous_stage
        self.identifiers = base_dataset.identifiers
        self._slice_axis = slice_axis
        self._perm = _SLICE_AXIS_PERM[slice_axis]

        self._cache = {}  # identifier -> (data_np, seg_np, seg_prev_np, properties)
        _log = log_fn or print

        import time as _time
        t0 = _time.perf_counter()
        total_bytes = 0
        for idx, ident in enumerate(self.identifiers):
            data, seg, seg_prev, props = base_dataset.load_case(ident)
            # Force full decompression into contiguous numpy arrays
            data_np = np.array(data, dtype=np.float32)
            seg_np = np.array(seg)
            seg_prev_np = np.array(seg_prev) if seg_prev is not None else None
            # Transpose to put slicing axis in position 1 (depth)
            if self._perm is not None:
                data_np = np.ascontiguousarray(np.transpose(data_np, self._perm))
                seg_np = np.ascontiguousarray(np.transpose(seg_np, self._perm))
                if seg_prev_np is not None:
                    seg_prev_np = np.ascontiguousarray(np.transpose(seg_prev_np, self._perm))
            self._cache[ident] = (data_np, seg_np, seg_prev_np, props)
            total_bytes += data_np.nbytes + seg_np.nbytes
            if seg_prev_np is not None:
                total_bytes += seg_prev_np.nbytes
        elapsed = _time.perf_counter() - t0
        view_name = _SLICE_AXIS_NAME[slice_axis]
        _log(
            f"PreloadedDataset ({view_name}): {len(self.identifiers)} cases loaded into RAM "
            f"({total_bytes / 1024**3:.2f} GB) in {elapsed:.1f}s"
        )

    def load_case(self, identifier):
        data_np, seg_np, seg_prev_np, props = self._cache[identifier]
        return data_np, seg_np, seg_prev_np, props

    @staticmethod
    def save_case(data, seg, properties, output_filename_truncated):
        raise NotImplementedError("PreloadedDataset is read-only")

    @staticmethod
    def get_identifiers(folder):
        raise NotImplementedError("Use base dataset's get_identifiers")


# ═══════════════════════════════════════════════════════════════════════════════
# Custom 2.5D Data Loader
# ═══════════════════════════════════════════════════════════════════════════════

class nnUNetDataLoader25D(nnUNetDataLoader):
    """
    Data loader that extracts K adjacent slices from 3D volumes and stacks
    them as channels for 2D processing.

    Input:  3D volumes from 3d_fullres preprocessing [C, D, H, W]
    Output: 2D batches with stacked slice context [C*K, H, W]
            Segmentation target for center slice only [classes, H, W]
    """

    def __init__(self,
                 data,
                 batch_size: int,
                 patch_size_2d: Union[List[int], Tuple[int, ...]],
                 final_patch_size_2d: Union[List[int], Tuple[int, ...]],
                 label_manager,
                 num_adjacent_slices: int = 7,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities=None,
                 pad_sides=None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        self.num_adjacent_slices = num_adjacent_slices
        self.half_k = num_adjacent_slices // 2

        # Use a thin 3D patch: [K, H, W]
        # This leverages nnU-Net's standard 3D cropping to extract K-slice windows
        patch_size_3d = [num_adjacent_slices, *patch_size_2d]
        final_patch_size_3d = [num_adjacent_slices, *final_patch_size_2d]

        if pad_sides is not None:
            pad_sides = [0, *pad_sides]  # No padding in slice dimension

        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size=patch_size_3d,
            final_patch_size=final_patch_size_3d,
            label_manager=label_manager,
            oversample_foreground_percent=oversample_foreground_percent,
            sampling_probabilities=sampling_probabilities,
            pad_sides=pad_sides,
            probabilistic_oversampling=probabilistic_oversampling,
            transforms=transforms,
        )

    def generate_train_batch(self):
        """
        Extract K-slice 3D patches, apply transforms, then reshape for 2D network.
        """
        result = super().generate_train_batch()

        data = result['data']    # [B, C, K, H, W] (3D)
        target = result['target']  # [B, seg_ch, K, H, W] or list (deep supervision)

        B, C, K, H, W = data.shape
        center = K // 2

        # Reshape: [B, C, K, H, W] → [B, C*K, H, W]
        data = data.reshape(B, C * K, H, W)

        # Segmentation target: center slice only
        if isinstance(target, list):
            # Deep supervision: list of [B, seg_ch, K_ds, H_ds, W_ds]
            new_target = []
            for t in target:
                if t.ndim == 5:
                    k_ds = t.shape[2]
                    center_ds = k_ds // 2
                    new_target.append(t[:, :, center_ds])  # [B, seg_ch, H_ds, W_ds]
                else:
                    new_target.append(t)
            target = new_target
        else:
            if target.ndim == 5:
                target = target[:, :, center]  # [B, seg_ch, H, W]

        result['data'] = data
        result['target'] = target
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Custom 2.5D Predictor
# ═══════════════════════════════════════════════════════════════════════════════

class nnUNetPredictor25D(nnUNetPredictor):
    """
    Predictor that performs slice-by-slice inference with K-slice context.

    For each target slice z:
        1. Extract slices [z - K//2, ..., z + K//2] with zero-padding at edges
        2. Stack as channels: [C*K, H, W]
        3. Apply 2D sliding window prediction
        4. Store result for slice z
    """

    def __init__(self, num_adjacent_slices: int = 7, slice_axis: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_adjacent_slices = num_adjacent_slices
        self.half_k = num_adjacent_slices // 2
        self.slice_axis = slice_axis
        self._perm = _SLICE_AXIS_PERM[slice_axis]

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str]], None],
        checkpoint_name: str = 'checkpoint_final.pth',
    ):
        """Override to force using the '2d' configuration from plans.

        The 2.5D trainer saves configuration='3d_fullres' in the checkpoint
        (because it uses 3d_fullres preprocessed data), but the actual network
        architecture is defined in the '2d' plans config (2D convolutions,
        2D patch size). We intercept the parent method, build the network from
        the '2d' config, then load the 2D checkpoint weights.
        """
        from nnunetv2.utilities.file_path_utilities import load_json
        import nnunetv2

        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(
                model_training_output_dir, checkpoint_name
            )

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(
                join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                map_location=torch.device('cpu'), weights_only=False
            )
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                inference_allowed_mirroring_axes = checkpoint.get(
                    'inference_allowed_mirroring_axes', None
                )
            parameters.append(checkpoint['network_weights'])

        # Force '2d' config for the network architecture
        configuration_manager = plans_manager.get_configuration('2d')

        # Input channels = C * K (modalities × adjacent slices)
        num_input_channels = len(dataset_json['channel_names']) * self.num_adjacent_slices

        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name, 'nnunetv2.training.nnUNetTrainer'
        )
        if trainer_class is None:
            # Trainer not in nnunetv2 package — use default build
            from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
            network = get_network_from_plans(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                deep_supervision=False,
            )
        else:
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                enable_deep_supervision=False,
            )

        self.plans_manager = plans_manager
        # Use 3d_fullres config for preprocessing (spacing, resampling) since
        # the 2.5D trainer was trained on 3d_fullres preprocessed data.
        # The 2d config is only needed for network architecture (patch_size).
        self.configuration_manager = plans_manager.get_configuration('3d_fullres')
        self.configuration_manager_2d = configuration_manager  # for patch_size in sliding window
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        network.load_state_dict(parameters[0])

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor):
        """
        Override to handle 2.5D prediction.
        input_image: [C, D, H, W] — full 3D volume
        returns: [num_classes, D, H, W] — predictions for every slice
        """
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()
        empty_cache(self.device)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be 4D (C, D, H, W)'

            # Transpose input so the target slicing axis is in position 1 (depth)
            if self._perm is not None:
                input_image = input_image.permute(self._perm).contiguous()

            C, D, H, W = input_image.shape
            K = self.num_adjacent_slices
            half_k = self.half_k

            # Get 2D patch size from the 2d architecture config
            patch_size_2d = self.configuration_manager_2d.patch_size
            if len(patch_size_2d) == 3:
                patch_size_2d = patch_size_2d[1:]  # Skip the K dimension
            elif len(patch_size_2d) == 2:
                pass
            else:
                raise ValueError(f"Unexpected patch_size dimensionality: {patch_size_2d}")

            # Compute 2D sliding window steps
            from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
            steps = compute_steps_for_sliding_window(
                (H, W), patch_size_2d, self.tile_step_size
            )

            # Gaussian weight for 2D patches
            from nnunetv2.inference.sliding_window_prediction import compute_gaussian
            results_device = self.device
            gaussian = compute_gaussian(
                tuple(patch_size_2d), sigma_scale=1. / 8,
                value_scaling_factor=10, device=results_device
            ) if self.use_gaussian else 1

            # Pre-allocate output
            num_seg_heads = self.label_manager.num_segmentation_heads
            predicted_logits = torch.zeros(
                (num_seg_heads, D, H, W),
                dtype=torch.half, device=results_device
            )
            n_predictions = torch.zeros(
                (D, H, W), dtype=torch.half, device=results_device
            )

            # Pad spatial dims if needed
            pad_h = max(0, patch_size_2d[0] - H)
            pad_w = max(0, patch_size_2d[1] - W)
            if pad_h > 0 or pad_w > 0:
                input_image = F.pad(input_image, (0, pad_w, 0, pad_h), mode='constant', value=0)
                _, _, H_padded, W_padded = input_image.shape
                steps = compute_steps_for_sliding_window(
                    (H_padded, W_padded), patch_size_2d, self.tile_step_size
                )
                predicted_logits_padded = torch.zeros(
                    (num_seg_heads, D, H_padded, W_padded),
                    dtype=torch.half, device=results_device
                )
                n_predictions_padded = torch.zeros(
                    (D, H_padded, W_padded), dtype=torch.half, device=results_device
                )
            else:
                H_padded, W_padded = H, W
                predicted_logits_padded = predicted_logits
                n_predictions_padded = n_predictions

            total_steps = D * len(steps[0]) * len(steps[1])
            with tqdm(total=total_steps, desc="2.5D inference", disable=not self.allow_tqdm) as pbar:
                for z in range(D):
                    # Extract K adjacent slices with zero-padding at boundaries
                    # Must match training channel order: reshape([B,C,K,H,W] -> [B,C*K,H,W])
                    # = [ch0_s0, ch0_s1, ..., ch0_sK, ch1_s0, ch1_s1, ..., ch1_sK]
                    slices_per_channel = []
                    for c in range(C):
                        for dz in range(-half_k, half_k + 1):
                            zz = z + dz
                            if 0 <= zz < D:
                                slices_per_channel.append(input_image[c, zz])  # [H, W]
                            else:
                                slices_per_channel.append(torch.zeros_like(input_image[0, 0]))
                    # Stack: [C*K, H_padded, W_padded]
                    context_input = torch.stack(slices_per_channel, dim=0)

                    # 2D sliding window over this slice
                    for sx in steps[0]:
                        for sy in steps[1]:
                            patch = context_input[
                                None, :,
                                sx:sx + patch_size_2d[0],
                                sy:sy + patch_size_2d[1]
                            ].to(self.device, non_blocking=True)  # [1, C*K, pH, pW]

                            prediction = self._internal_maybe_mirror_and_predict(patch)[0]
                            prediction = prediction.to(results_device)

                            if self.use_gaussian:
                                prediction *= gaussian

                            predicted_logits_padded[
                                :, z,
                                sx:sx + patch_size_2d[0],
                                sy:sy + patch_size_2d[1]
                            ] += prediction  # [num_classes, pH, pW] — batch dim already removed

                            n_predictions_padded[
                                z,
                                sx:sx + patch_size_2d[0],
                                sy:sy + patch_size_2d[1]
                            ] += gaussian if isinstance(gaussian, torch.Tensor) else 1

                            pbar.update()

            # Normalize
            torch.div(predicted_logits_padded, n_predictions_padded, out=predicted_logits_padded)

            # Remove padding if applied
            if pad_h > 0 or pad_w > 0:
                predicted_logits = predicted_logits_padded[:, :, :H, :W]
            else:
                predicted_logits = predicted_logits_padded

            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array.')

            # Un-transpose prediction back to original orientation
            # _SLICE_AXIS_PERM permutations are self-inverse (simple axis swaps)
            if self._perm is not None:
                predicted_logits = predicted_logits.permute(self._perm).contiguous()

        return predicted_logits


# ═══════════════════════════════════════════════════════════════════════════════
# Thread-based Prefetcher (Windows-safe alternative to NonDetMultiThreadedAugmenter)
# ═══════════════════════════════════════════════════════════════════════════════

class ThreadedPrefetcher:
    """
    Thread-based batch prefetcher for CPU-GPU overlap.

    Creates N SingleThreadedAugmenter instances, each running in its own
    background thread, all feeding a shared queue. The main training thread
    pulls pre-computed batches while background threads prepare the next ones.

    This replaces batchgenerators' NonDetMultiThreadedAugmenter which uses
    multiprocessing.Process and causes heap corruption (0xc0000374) on Windows.

    Why threads work here:
        - NumPy releases the GIL during array operations (augmentation)
        - Disk I/O releases the GIL (loading .npz files)
        - PyTorch GPU ops release the GIL during kernel execution
        → Background threads achieve real parallelism with GPU and each other
    """

    def __init__(self, augmenters, queue_size=6):
        self._queue = Queue(maxsize=queue_size)
        self._stop = Event()
        self._error = None
        self._threads = []
        for i, aug in enumerate(augmenters):
            t = Thread(target=self._fill, args=(aug,), daemon=True,
                       name=f"prefetch-{i}")
            t.start()
            self._threads.append(t)

    def _fill(self, augmenter):
        """Background worker: generate batches and push to shared queue."""
        try:
            while not self._stop.is_set():
                batch = next(augmenter)
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=1.0)
                        break
                    except Full:
                        continue
        except Exception as e:
            if not self._stop.is_set():
                self._error = e

    def __next__(self):
        while True:
            if self._error is not None:
                raise self._error
            try:
                return self._queue.get(timeout=5.0)
            except Empty:
                if self._error is not None:
                    raise self._error
                # Keep waiting — augmentation for a batch may take seconds
                continue

    def __iter__(self):
        return self

    def shutdown(self):
        self._stop.set()
        # Drain queue to unblock any workers stuck on put()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        for t in self._threads:
            t.join(timeout=10)

    def __del__(self):
        self.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-side Data Augmentation
# ═══════════════════════════════════════════════════════════════════════════════

class GPUAugmentation:
    """
    Batch-level data augmentation executed on GPU.

    Moves the CPU-bound augmentation pipeline (SpatialTransform, intensity
    transforms, mirror) to the GPU where they run as vectorized batch ops.
    This eliminates the Python-GIL data-pipeline bottleneck that causes
    45% GPU idle time with threaded CPU augmentation.

    Applied in train_step() after data.to(device) and before forward().
    """

    def __init__(self, device, rotation_range, mirror_axes, ds_scales_2d):
        self.device = device
        self.rotation_range = rotation_range
        self.mirror_axes = mirror_axes
        self.ds_scales_2d = ds_scales_2d  # [[h_s, w_s], ...] for deep supervision

    @torch.no_grad()
    def __call__(self, data, seg):
        """
        Apply all augmentations to a batch on GPU.

        Args:
            data: [B, C*K, H, W] float on GPU
            seg:  [B, 1, H, W] int/float on GPU

        Returns:
            data:   augmented [B, C*K, H, W]
            target: list of [B, 1, H_s, W_s] at deep supervision scales
        """
        data, seg = self._spatial_transform(data, seg)
        data, seg = self._mirror_transform(data, seg)
        data = self._intensity_transforms(data)
        target = self._create_ds_targets(seg)
        return data, target

    # ── Spatial: rotation + scaling ──────────────────────────────────────────

    def _spatial_transform(self, data, seg,
                           p_rotation=0.2, p_scaling=0.2,
                           scaling_range=(0.7, 1.4)):
        B = data.shape[0]
        dev = self.device

        do_rotate = torch.rand(B, device=dev) < p_rotation
        do_scale  = torch.rand(B, device=dev) < p_scaling
        needs = do_rotate | do_scale
        if not needs.any():
            return data, seg

        # Random per-sample angles and scales
        angles = torch.zeros(B, device=dev)
        angles[do_rotate] = torch.empty(do_rotate.sum(), device=dev).uniform_(
            self.rotation_range[0], self.rotation_range[1]
        )
        scales = torch.ones(B, device=dev)
        scales[do_scale] = torch.empty(do_scale.sum(), device=dev).uniform_(
            *scaling_range
        )

        cos_a = torch.cos(angles) * scales
        sin_a = torch.sin(angles) * scales

        theta = torch.zeros(B, 2, 3, device=dev, dtype=data.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        idx = needs.nonzero(as_tuple=True)[0]

        grid = F.affine_grid(theta[idx], data[idx].shape, align_corners=False)
        data[idx] = F.grid_sample(
            data[idx], grid, mode='bilinear', padding_mode='zeros',
            align_corners=False
        )
        seg_f = seg[idx].float()
        seg[idx] = F.grid_sample(
            seg_f, grid, mode='nearest', padding_mode='zeros',
            align_corners=False
        ).to(seg.dtype)

        return data, seg

    # ── Mirror ───────────────────────────────────────────────────────────────

    def _mirror_transform(self, data, seg):
        for ax in self.mirror_axes:
            dim = ax + 2  # axes (0,1) → tensor dims (2,3)
            mask = torch.rand(data.shape[0], device=self.device) < 0.5
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0]
                data[idx] = torch.flip(data[idx], [dim])
                seg[idx]  = torch.flip(seg[idx],  [dim])
        return data, seg

    # ── Intensity transforms (data only) ─────────────────────────────────────

    def _intensity_transforms(self, data):
        B, C, H, W = data.shape
        dev = self.device

        # 1) Gaussian noise  (p=0.1, sync_channels=True)
        smask = (torch.rand(B, 1, 1, 1, device=dev) < 0.1)
        if smask.any():
            var = torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.1)
            noise = torch.randn(B, 1, H, W, device=dev) * var.sqrt()
            data = data + noise * smask

        # 2) Gaussian blur  (p=0.2, per_channel p=0.5)
        smask = (torch.rand(B, 1, 1, 1, device=dev) < 0.2)
        if smask.any():
            sigma = torch.empty(1).uniform_(0.5, 1.0).item()
            ks = max(3, int(2 * round(3 * sigma) + 1))
            if ks % 2 == 0:
                ks += 1
            ch_mask = (torch.rand(B, C, 1, 1, device=dev) < 0.5)
            combined = (smask & ch_mask).float()
            if combined.any():
                blurred = kornia.filters.gaussian_blur2d(
                    data, (ks, ks), (sigma, sigma)
                )
                data = combined * blurred + (1 - combined) * data

        # 3) Multiplicative brightness  (p=0.15, per_channel)
        smask = (torch.rand(B, 1, 1, 1, device=dev) < 0.15)
        if smask.any():
            factor = torch.empty(B, C, 1, 1, device=dev).uniform_(0.75, 1.25)
            data = torch.where(smask.expand_as(data), data * factor, data)

        # 4) Contrast  (p=0.15, per_channel, preserve_range=True)
        smask = (torch.rand(B, 1, 1, 1, device=dev) < 0.15)
        if smask.any():
            mean = data.mean(dim=(2, 3), keepdim=True)
            factor = torch.empty(B, C, 1, 1, device=dev).uniform_(0.75, 1.25)
            result = mean + (data - mean) * factor
            # preserve_range: clamp to [data.min, data.max] per channel
            lo = data.amin(dim=(2, 3), keepdim=True)
            hi = data.amax(dim=(2, 3), keepdim=True)
            result = result.clamp(lo, hi)
            data = torch.where(smask.expand_as(data), result, data)

        # 5) Simulate low resolution  (p=0.25, per_channel p=0.5, sync_axes)
        smask = (torch.rand(B, 1, 1, 1, device=dev) < 0.25)
        if smask.any():
            scale = torch.empty(1).uniform_(0.5, 1.0).item()
            if scale < 0.99:
                th = max(1, int(round(H * scale)))
                tw = max(1, int(round(W * scale)))
                ch_mask = (torch.rand(B, C, 1, 1, device=dev) < 0.5)
                combined = (smask & ch_mask).float()
                if combined.any():
                    down = F.interpolate(data, size=(th, tw), mode='bilinear',
                                         align_corners=False)
                    up = F.interpolate(down, size=(H, W), mode='bilinear',
                                       align_corners=False)
                    data = combined * up + (1 - combined) * data

        # 6) Gamma (inverted)  (p=0.1, retain_stats)
        smask = (torch.rand(B, 1, 1, 1, device=dev) < 0.1)
        if smask.any():
            data = self._gamma(data, smask, invert=True)

        # 7) Gamma (non-inverted)  (p=0.3, retain_stats)
        smask = (torch.rand(B, 1, 1, 1, device=dev) < 0.3)
        if smask.any():
            data = self._gamma(data, smask, invert=False)

        return data

    @staticmethod
    def _gamma(data, mask, invert):
        """Apply random gamma correction with retain_stats."""
        B, C, _, _ = data.shape
        dev = data.device

        src = -data if invert else data.clone()
        mn = src.amin(dim=(2, 3), keepdim=True)
        rng = src.amax(dim=(2, 3), keepdim=True) - mn + 1e-8

        # Statistics before gamma
        mean_before = src.mean(dim=(2, 3), keepdim=True)
        std_before  = src.std(dim=(2, 3), keepdim=True) + 1e-8

        # Normalize to [0, 1], apply gamma, rescale
        gamma = torch.empty(B, C, 1, 1, device=dev).uniform_(0.7, 1.5)
        norm = ((src - mn) / rng).clamp(0, 1)
        result = norm.pow(gamma) * rng + mn

        # Retain stats: match mean/std of original
        mean_after = result.mean(dim=(2, 3), keepdim=True)
        std_after  = result.std(dim=(2, 3), keepdim=True) + 1e-8
        result = (result - mean_after) / std_after * std_before + mean_before

        if invert:
            result = -result

        return torch.where(mask.expand_as(data), result, data)

    # ── Deep supervision target creation ─────────────────────────────────────

    def _create_ds_targets(self, seg):
        """Create multi-scale segmentation targets for deep supervision.
        
        Matches DownsampleSegForDSTransform: one target per ds_scale entry.
        The first ds_scale is typically [1.0, 1.0] (full res).
        """
        targets = []
        H, W = seg.shape[2], seg.shape[3]
        for h_s, w_s in self.ds_scales_2d:
            th = max(1, int(round(H * h_s)))
            tw = max(1, int(round(W * w_s)))
            if th == H and tw == W:
                targets.append(seg)
            else:
                ds = F.interpolate(seg.float(), size=(th, tw), mode='nearest')
                targets.append(ds.to(seg.dtype))
        return targets


# ═══════════════════════════════════════════════════════════════════════════════
# Custom 2.5D Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class nnUNetTrainer_25D(nnUNetTrainer):
    """
    True 2.5D nnU-Net trainer.

    Uses 3d_fullres preprocessed data with 2D convolutions.
    K adjacent slices are stacked as input channels, providing inter-slice
    context while maintaining 2D computational efficiency.

    Configuration:
        NUM_ADJACENT_SLICES: Number of slices in context window (default: 7)
            7 = center slice ± 3 neighbors = 21mm context at 3mm axial spacing.
            Chosen via data-driven analysis: K=7 covers 94.0% of all 17,262
            lesions in DS502 preprocessed space (3mm axial). See analysis of
            lesion axial extent distribution across DS1 (MSLesSeg) and DS3 (WMH).

    Features:
        - Weights & Biases (wandb) logging: auto-logs train/val loss, pseudo
          dice, EMA dice, learning rate, epoch time every epoch.
          Set env WANDB_MODE=disabled to turn off.
        - Early stopping: stops training when EMA pseudo Dice has not improved
          for EARLY_STOP_PATIENCE epochs (checked after warmup period).
    """

    NUM_ADJACENT_SLICES = 7
    SLICE_AXIS = 0             # 0=axial (default), 1=coronal, 2=sagittal
    NUM_PREFETCH_THREADS = 8   # Background threads for CPU augmentation overlap
    EARLY_STOP_PATIENCE = 150   # Stop if no improvement for this many epochs
    EARLY_STOP_WARMUP = 50     # Don't check early stopping before this epoch
    CURVE_FIT_MIN_EPOCHS = 60    # Minimum epochs before curve-fit early stopping
    CURVE_FIT_EPSILON = 0.001  # Stop when predicted remaining improvement < this
    CURVE_FIT_INTERVAL = 10    # Re-fit every N epochs (fitting is cheap but not free)
    CURVE_FIT_RECENCY_GUARD = 20  # Don't stop if new best EMA within last N epochs
    PRELOAD_TO_RAM = True       # Preload all volumes into RAM (eliminates blosc2 decompression per batch)

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # We need the 2D architecture from the plans
        # The trainer is called with 3d_fullres (for data), but we use 2d architecture
        try:
            self._2d_config = self.plans_manager.get_configuration('2d')
        except Exception:
            raise RuntimeError(
                "nnUNetTrainer_25D requires both '2d' and '3d_fullres' plans to exist. "
                "Run: nnUNetv2_plan_and_preprocess -d DATASET --verify_dataset_integrity "
                "(without -c flag to generate both configurations)"
            )

        # Store the 2D patch size for data loading and inference
        self._patch_size_2d = list(self._2d_config.patch_size)
        self._batch_size_2d = self._2d_config.batch_size

        # Guard: multi-view transposition requires preloading
        if self.SLICE_AXIS != 0 and not self.PRELOAD_TO_RAM:
            raise RuntimeError(
                f"PRELOAD_TO_RAM must be True for non-axial views (SLICE_AXIS={self.SLICE_AXIS}). "
                "Volume transposition is performed during preloading."
            )

        self.print_to_log_file(
            f"nnUNetTrainer_25D initialized:"
            f"\n  Adjacent slices (K): {self.NUM_ADJACENT_SLICES}"
            f"\n  Slice axis: {self.SLICE_AXIS} ({_SLICE_AXIS_NAME[self.SLICE_AXIS]})"
            f"\n  2D patch size: {self._patch_size_2d}"
            f"\n  Using 3d_fullres data with 2D architecture"
        )

    def initialize(self):
        """Override to build a 2D network with K*C input channels."""
        if not self.was_initialized:
            # Determine base number of input channels (modalities)
            base_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
            # 2.5D: K slices × C channels
            self.num_input_channels = base_channels * self.NUM_ADJACENT_SLICES
            self._base_channels = base_channels

            self.print_to_log_file(
                f"Input channels: {base_channels} base × {self.NUM_ADJACENT_SLICES} slices = "
                f"{self.num_input_channels}"
            )

            # Build 2D network using 2d architecture config but with modified input channels
            arch_kwargs = deepcopy(dict(self._2d_config.network_arch_init_kwargs))

            # Build the network
            self.network = self.build_network_architecture(
                self._2d_config.network_arch_class_name,
                arch_kwargs,
                self._2d_config.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            # --- GPU optimizations ---
            # 1) cuDNN benchmark: auto-tune convolution algorithms for fixed patch sizes
            torch.backends.cudnn.benchmark = True

            # 2) channels_last memory format: NHWC layout optimizes 2D conv on tensor cores
            self.network = self.network.to(memory_format=torch.channels_last)

            # 3) bf16 autocast: native Blackwell tensor core dtype, no GradScaler needed
            #    We override the parent's fp16 GradScaler with None to use bf16 instead
            if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
                self._use_bf16 = True
                self.grad_scaler = None  # bf16 doesn't need loss scaling
                torch.set_autocast_dtype('cuda', torch.bfloat16)
                self.print_to_log_file(
                    "GPU optimizations: cudnn.benchmark=True, channels_last, bf16 autocast (no GradScaler)"
                )
            else:
                self._use_bf16 = False
                self.print_to_log_file(
                    "GPU optimizations: cudnn.benchmark=True, channels_last, fp16 autocast (bf16 not supported)"
                )

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # VRAM probe AFTER network + optimizer are ready
            self._set_batch_size_and_oversample()

            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True
        else:
            raise RuntimeError("Already initialized.")

    # ── Weights & Biases integration ─────────────────────────────────────────

    def _init_wandb(self):
        """Initialize W&B run. Called from on_train_start."""
        if not WANDB_AVAILABLE:
            self.print_to_log_file("wandb not installed — skipping W&B logging")
            self._wandb_run = None
            return
        import os
        if os.environ.get('WANDB_MODE', '').lower() == 'disabled':
            self.print_to_log_file("WANDB_MODE=disabled — skipping W&B logging")
            self._wandb_run = None
            return
        try:
            ds_name = self.plans_manager.dataset_name
            plans_name = self.plans_manager.plans_name
            run_name = f"{self.__class__.__name__}__{plans_name}__{self.configuration_name}__fold{self.fold}"
            self._wandb_run = wandb.init(
                project="ms-lesion-seg",
                name=run_name,
                group=ds_name,
                tags=[ds_name, self.__class__.__name__, plans_name, self.configuration_name],
                config={
                    "dataset": ds_name,
                    "trainer": self.__class__.__name__,
                    "plans": plans_name,
                    "configuration": self.configuration_name,
                    "fold": self.fold,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "num_adjacent_slices": self.NUM_ADJACENT_SLICES,
                    "slice_axis": self.SLICE_AXIS,
                    "slice_axis_name": _SLICE_AXIS_NAME[self.SLICE_AXIS],
                    "patch_size_2d": self._patch_size_2d,
                    "early_stop_patience": self.EARLY_STOP_PATIENCE,
                    "early_stop_warmup": self.EARLY_STOP_WARMUP,
                },
                resume="allow",
                reinit=True,
                settings=wandb.Settings(init_timeout=120),
            )
            self.print_to_log_file(f"W&B run initialized: {self._wandb_run.url}")
        except Exception as e:
            self.print_to_log_file(f"W&B init failed: {e} — continuing without W&B")
            self._wandb_run = None

    @staticmethod
    def _asymptotic_model(x, a, b, c):
        """Exponential saturation model: y = a - b * exp(-c * x).
        a = predicted plateau, b = total improvement range, c = convergence rate."""
        return a - b * np.exp(-c * x)

    def _predict_convergence(self, ema_history: list) -> dict | None:
        """Fit asymptotic curve to EMA dice history and predict remaining improvement.

        Returns dict with fit results or None if fitting fails.
        """
        n = len(ema_history)
        if n < self.CURVE_FIT_MIN_EPOCHS:
            return None

        y = np.array(ema_history, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)

        # Initial guesses
        y_max = y.max()
        y_min = y[:max(1, n // 10)].mean()  # average of first 10% as baseline
        a_init = y_max * 1.02  # plateau slightly above current best
        b_init = max(a_init - y_min, 0.01)
        c_init = 3.0 / n  # assume ~63% convergence by 1/3 of training

        try:
            popt, pcov = curve_fit(
                self._asymptotic_model, x, y,
                p0=[a_init, b_init, c_init],
                bounds=([y_max * 0.95, 0.001, 1e-6],  # a >= 95% of current best
                        [1.0, 2.0, 1.0]),               # a <= 1.0 (dice), c <= 1.0
                maxfev=5000,
            )
            a, b, c = popt
            best = float(y.max())
            predicted_remaining = a - best
            # R² goodness of fit
            y_pred = self._asymptotic_model(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return {
                'plateau': a,
                'remaining': predicted_remaining,
                'rate': c,
                'r_squared': r_squared,
                'best': best,
            }
        except (RuntimeError, ValueError):
            return None

    def on_train_start(self):
        """Override to initialize W&B and early stopping state."""
        super().on_train_start()
        self._init_wandb()
        # Early stopping state — recover actual best epoch from logger history
        log = self.logger.my_fantastic_logging
        ema_hist = log.get('ema_fg_dice', [])
        if self._best_ema is not None and len(ema_hist) > 0:
            self._es_best_ema = self._best_ema
            self._es_best_epoch = max(range(len(ema_hist)), key=lambda i: ema_hist[i])
        else:
            self._es_best_ema = -1.0
            self._es_best_epoch = self.current_epoch
        self._early_stopped = False
        self._last_convergence_pred = None
        self.print_to_log_file(
            f"Early stopping: curve-fit (epsilon={self.CURVE_FIT_EPSILON}, "
            f"min_epochs={self.CURVE_FIT_MIN_EPOCHS}), "
            f"patience fallback={self.EARLY_STOP_PATIENCE}, "
            f"warmup={self.EARLY_STOP_WARMUP}"
        )

    def on_epoch_end(self):
        """Override to add W&B logging and early stopping check."""
        # Let parent do its thing (logging, checkpointing, best model, etc.)
        super().on_epoch_end()

        log = self.logger.my_fantastic_logging
        epoch = self.current_epoch - 1  # parent incremented it already

        # ── W&B logging ──
        if hasattr(self, '_wandb_run') and self._wandb_run is not None:
            try:
                metrics = {
                    "train_loss": log['train_losses'][-1],
                    "val_loss": log['val_losses'][-1],
                    "mean_fg_dice": log['mean_fg_dice'][-1],
                    "ema_fg_dice": log['ema_fg_dice'][-1],
                    "lr": log['lrs'][-1],
                    "epoch": epoch,
                }
                if len(log['epoch_end_timestamps']) > 0 and len(log['epoch_start_timestamps']) > 0:
                    metrics["epoch_time_s"] = (
                        log['epoch_end_timestamps'][-1] - log['epoch_start_timestamps'][-1]
                    )
                wandb.log(metrics, step=epoch)
            except Exception as e:
                self.print_to_log_file(f"W&B log error: {e}")

        # ── Early stopping (curve-fitting + patience fallback) ──
        if epoch >= self.EARLY_STOP_WARMUP and len(log['ema_fg_dice']) > 0:
            current_ema = log['ema_fg_dice'][-1]
            if current_ema > self._es_best_ema:
                self._es_best_ema = current_ema
                self._es_best_epoch = epoch

            # Curve-fitting check (every CURVE_FIT_INTERVAL epochs)
            if epoch >= self.CURVE_FIT_MIN_EPOCHS and epoch % self.CURVE_FIT_INTERVAL == 0:
                pred = self._predict_convergence(log['ema_fg_dice'])
                self._last_convergence_pred = pred
                if pred is not None:
                    self.print_to_log_file(
                        f"Convergence fit: plateau={pred['plateau']:.4f}, "
                        f"remaining={pred['remaining']:.4f}, "
                        f"R2={pred['r_squared']:.3f}, rate={pred['rate']:.4f}"
                    )
                    # Log to W&B
                    if hasattr(self, '_wandb_run') and self._wandb_run is not None:
                        try:
                            wandb.log({
                                "predicted_plateau": pred['plateau'],
                                "predicted_remaining": pred['remaining'],
                                "convergence_r_squared": pred['r_squared'],
                            }, step=epoch)
                        except Exception:
                            pass
                    # Stop if predicted remaining improvement is negligible,
                    # fit is good (R2 > 0.9), and no recent improvement
                    epochs_since_best = epoch - self._es_best_epoch
                    if (pred['remaining'] < self.CURVE_FIT_EPSILON
                            and pred['r_squared'] > 0.9
                            and epochs_since_best >= self.CURVE_FIT_RECENCY_GUARD):
                        self.print_to_log_file(
                            f"\n{'='*60}\n"
                            f"EARLY STOPPING (curve-fit) at epoch {epoch}\n"
                            f"Predicted plateau: {pred['plateau']:.4f}\n"
                            f"Best EMA Dice: {self._es_best_ema:.4f} at epoch {self._es_best_epoch}\n"
                            f"Predicted remaining: {pred['remaining']:.4f} < {self.CURVE_FIT_EPSILON}\n"
                            f"Fit R2: {pred['r_squared']:.3f}\n"
                            f"Epochs since best: {epochs_since_best} >= {self.CURVE_FIT_RECENCY_GUARD}\n"
                            f"{'='*60}"
                        )
                        self._early_stopped = True
                    elif (pred['remaining'] < self.CURVE_FIT_EPSILON
                            and pred['r_squared'] > 0.9):
                        self.print_to_log_file(
                            f"Curve-fit would stop but recency guard active "
                            f"(best {self._es_best_ema:.4f} was {epochs_since_best} epochs ago, "
                            f"need {self.CURVE_FIT_RECENCY_GUARD})")

            # Patience fallback (in case curve fit never triggers)
            if not self._early_stopped:
                epochs_since_best = epoch - self._es_best_epoch
                if epochs_since_best >= self.EARLY_STOP_PATIENCE:
                    self.print_to_log_file(
                        f"\n{'='*60}\n"
                        f"EARLY STOPPING (patience) at epoch {epoch}\n"
                        f"No improvement in ema_fg_dice for {self.EARLY_STOP_PATIENCE} epochs\n"
                        f"Best EMA Dice: {self._es_best_ema:.4f} at epoch {self._es_best_epoch}\n"
                        f"{'='*60}"
                    )
                    self._early_stopped = True

    def on_train_end(self):
        """Override to finalize W&B run."""
        super().on_train_end()
        if hasattr(self, '_wandb_run') and self._wandb_run is not None:
            try:
                wandb.finish()
                self.print_to_log_file("W&B run finished.")
            except Exception as e:
                self.print_to_log_file(f"W&B finish error: {e}")

    def run_training(self):
        """Override to add early stopping break to the training loop."""
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            # Check early stopping flag (set in on_epoch_end)
            if hasattr(self, '_early_stopped') and self._early_stopped:
                break

        self.on_train_end()

    # VRAM-aware batch sizing: probe GPU memory to fill ~80% of available VRAM.
    # GPU-side augmentation removes the CPU bottleneck, so larger batches help.
    VRAM_TARGET_FRACTION = 0.80   # use up to 80% of total VRAM
    VRAM_PROBE_BATCH = 2          # probe with 2 samples to measure per-sample cost
    MIN_BATCH_SIZE = 2
    MAX_BATCH_SIZE = 32            # capped for 32GB system RAM (8GB preload + 8 prefetch threads + queue)
    VRAM_AUGMENTATION_OVERHEAD = 3.0  # multiplier: GPU augmentation + DS costs ~3x probe-only estimate

    def _set_batch_size_and_oversample(self):
        """Set batch size via VRAM probing.

        The 2D plan assumes C input channels but 2.5D uses C*K, so the plan's
        batch size (e.g. 303 for 2ch) is wildly wrong for 18ch. We probe GPU
        memory to find the actual optimal batch size.
        """
        import gc

        if self.device.type != 'cuda':
            self.batch_size = max(self.MIN_BATCH_SIZE, self._batch_size_2d)
            self.oversample_foreground_percent = 0.33
            return

        patch_h, patch_w = self._patch_size_2d
        in_ch = self.num_input_channels
        amp_dtype = torch.bfloat16 if getattr(self, '_use_bf16', False) else torch.float16
        probe_model = getattr(self.network, '_orig_mod', self.network)

        def _run_probe(n):
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(self.device)
            dummy = torch.randn(n, in_ch, patch_h, patch_w,
                                device=self.device, dtype=torch.float32
                                ).to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                out = probe_model(dummy)
                if isinstance(out, (list, tuple)):
                    loss = sum(
                        torch.nn.functional.cross_entropy(
                            o, torch.zeros((n,) + o.shape[2:], device=self.device, dtype=torch.long),
                            ignore_index=-1) for o in out)
                else:
                    loss = torch.nn.functional.cross_entropy(
                        out, torch.zeros(n, *out.shape[2:], device=self.device, dtype=torch.long),
                        ignore_index=-1)
            loss.backward()
            self.optimizer.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(self.device)
            del dummy, out, loss
            torch.cuda.empty_cache()
            gc.collect()
            return peak

        mem_before = torch.cuda.memory_allocated(self.device)
        total_vram = torch.cuda.get_device_properties(self.device).total_memory
        target_mem = total_vram * self.VRAM_TARGET_FRACTION

        try:
            peak_1 = _run_probe(1)
            try:
                peak_2 = _run_probe(2)
                per_sample_mem = peak_2 - peak_1
                fixed_overhead = max(peak_1 - mem_before - per_sample_mem, 0)
                method = "differential (n=1 vs n=2)"
            except RuntimeError:
                torch.cuda.empty_cache()
                gc.collect()
                per_sample_mem = (peak_1 - mem_before) * 0.6
                fixed_overhead = (peak_1 - mem_before) * 0.4
                peak_2 = None
                method = "single-sample estimate (n=2 OOM)"

            # Scale per-sample cost to account for GPU augmentation pipeline
            # (spatial transforms, intensity, mirror, deep supervision targets)
            # which are NOT captured by the network-only probe
            effective_per_sample = per_sample_mem * self.VRAM_AUGMENTATION_OVERHEAD
            available = target_mem - mem_before - fixed_overhead
            if effective_per_sample > 0 and available > 0:
                optimal_bs = int(available / effective_per_sample)
                optimal_bs = max(self.MIN_BATCH_SIZE, min(optimal_bs, self.MAX_BATCH_SIZE))
            else:
                optimal_bs = self.MIN_BATCH_SIZE

            self.batch_size = optimal_bs
            self.oversample_foreground_percent = 0.33
            log_msg = (
                f"VRAM probe results ({method}):"
                f"\n  Total VRAM: {total_vram / 1024**3:.1f} GB"
                f"\n  Model + optimizer: {mem_before / 1024**3:.2f} GB"
                f"\n  Fixed overhead: {fixed_overhead / 1024**2:.1f} MB"
                f"\n  Peak n=1: {peak_1 / 1024**3:.2f} GB"
            )
            if peak_2 is not None:
                log_msg += f"\n  Peak n=2: {peak_2 / 1024**3:.2f} GB"
            log_msg += (
                f"\n  Per-sample cost: {per_sample_mem / 1024**2:.1f} MB"
                f"\n  Target ({self.VRAM_TARGET_FRACTION*100:.0f}% VRAM): {target_mem / 1024**3:.1f} GB"
                f"\n  2D plan BS: {self._batch_size_2d} -> Optimal: {optimal_bs}"
                f"\n  (2D plan BS based on {self._base_channels}ch; actual: {in_ch}ch)"
            )
            self.print_to_log_file(log_msg)
        except Exception as e:
            fallback = max(self.MIN_BATCH_SIZE, self._batch_size_2d // self.NUM_ADJACENT_SLICES)
            self.batch_size = min(fallback, self.MAX_BATCH_SIZE)
            self.oversample_foreground_percent = 0.33
            self.print_to_log_file(f"VRAM probe failed ({e}), fallback BS: {self.batch_size}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def _probe_vram_and_set_batch_size(self):
        """Differential VRAM probe: run with n_small and n_large to isolate
        the true marginal per-sample cost, eliminating fixed overhead
        (CUDA context, cuDNN workspace, one-time activation buffers).

        NOTE: Currently unused — _set_batch_size_and_oversample() is called
        instead, since 2.5D patches are small and VRAM is not the bottleneck.
        Kept for reference / future use.
        """
        import gc

        n_small, n_large = 1, 4
        patch_h, patch_w = self._patch_size_2d
        in_ch = self.num_input_channels
        amp_dtype = torch.bfloat16 if getattr(self, '_use_bf16', False) else torch.float16
        # Use uncompiled model for probing — torch.compile produces
        # artificially low per-sample cost at small batch sizes
        probe_model = getattr(self.network, '_orig_mod', self.network)
        is_compiled = probe_model is not self.network

        def _run_probe(n):
            """Forward+backward with n samples, return peak memory."""
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(self.device)

            dummy = torch.randn(
                n, in_ch, patch_h, patch_w,
                device=self.device, dtype=torch.float32
            ).to(memory_format=torch.channels_last)

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                out = probe_model(dummy)
                if isinstance(out, (list, tuple)):
                    loss = sum(
                        torch.nn.functional.cross_entropy(
                            o,
                            torch.zeros((n,) + o.shape[2:], device=self.device, dtype=torch.long),
                            ignore_index=-1
                        ) for o in out
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        out,
                        torch.zeros(n, *out.shape[2:], device=self.device, dtype=torch.long),
                        ignore_index=-1
                    )
            loss.backward()
            self.optimizer.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(self.device)
            del dummy, out, loss
            torch.cuda.empty_cache()
            gc.collect()
            return peak

        mem_before = torch.cuda.memory_allocated(self.device)

        try:
            peak_small = _run_probe(n_small)
            peak_large = _run_probe(n_large)

            per_sample_mem = (peak_large - peak_small) / (n_large - n_small)
            # Fixed overhead = peak_small - (per_sample * n_small) - model_mem
            fixed_overhead = peak_small - mem_before - per_sample_mem * n_small
            fixed_overhead = max(fixed_overhead, 0)

            total_vram = torch.cuda.get_device_properties(self.device).total_memory
            target_mem = total_vram * self.VRAM_TARGET_FRACTION
            available_for_batch = target_mem - mem_before - fixed_overhead

            if per_sample_mem > 0 and available_for_batch > 0:
                optimal_bs = int(available_for_batch / per_sample_mem)
                optimal_bs = max(self.MIN_BATCH_SIZE, min(optimal_bs, self.MAX_BATCH_SIZE))
            else:
                optimal_bs = max(self.MIN_BATCH_SIZE, self._batch_size_2d)

            self.batch_size = optimal_bs
            self.oversample_foreground_percent = 0.33

            self.print_to_log_file(
                f"VRAM probe results (differential{', eager-mode' if is_compiled else ''}):"
                f"\n  Total VRAM: {total_vram / 1024**3:.1f} GB"
                f"\n  Model + optimizer: {mem_before / 1024**3:.2f} GB"
                f"\n  Fixed overhead: {fixed_overhead / 1024**2:.1f} MB"
                f"\n  Peak n={n_small}: {peak_small / 1024**3:.2f} GB"
                f"\n  Peak n={n_large}: {peak_large / 1024**3:.2f} GB"
                f"\n  Marginal per-sample cost: {per_sample_mem / 1024**2:.1f} MB"
                f"\n  Target ({self.VRAM_TARGET_FRACTION*100:.0f}% VRAM): {target_mem / 1024**3:.1f} GB"
                f"\n  Optimal batch size: {optimal_bs}"
                f"\n  2D plan batch size was: {self._batch_size_2d}"
            )

        except Exception as e:
            fallback_bs = max(
                self.MIN_BATCH_SIZE,
                int(self._batch_size_2d / (1 + 0.2 * (self.NUM_ADJACENT_SLICES - 1)))
            )
            self.batch_size = min(fallback_bs, self.MAX_BATCH_SIZE)
            self.oversample_foreground_percent = 0.33
            self.print_to_log_file(
                f"VRAM probe failed ({e}), using fallback batch size: {self.batch_size}"
            )
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def _get_deep_supervision_scales(self):
        """Use 2D deep supervision scales, extended to 3D for the thin-slab data.
        
        The data is [C, K, H, W] during transforms (before reshape to 2D),
        so deep supervision scales must be 3D with scale=1.0 for the K dimension.
        """
        if self.enable_deep_supervision:
            pool_op_kernel_sizes = self._2d_config.pool_op_kernel_sizes
            # Get 2D scales
            scales_2d = list(
                list(i) for i in 1 / np.cumprod(
                    np.vstack(pool_op_kernel_sizes), axis=0
                )
            )[:-1]
            # Prepend 1.0 for the K (depth) dimension — no downsampling along slices
            deep_supervision_scales = [[1.0] + list(s) for s in scales_2d]
        else:
            deep_supervision_scales = None
        return deep_supervision_scales

    def train_step(self, batch: dict) -> dict:
        """Override to apply GPU-side augmentation before forward pass.

        Data flow:
            CPU: load .npz → crop [K,H,W] → RemoveLabel → queue
            GPU: spatial → mirror → intensity → DS targets → forward → loss → backward
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # --- GPU Augmentation ---
        # data: [B, C*K, H, W], target: [B, 1, H, W] (no DS yet)
        if self.network.training and hasattr(self, '_gpu_aug'):
            # target is a single tensor (no DownsampleSegForDS on CPU)
            data, target = self._gpu_aug(data, target)

        # --- channels_last conversion ---
        data = data.contiguous(memory_format=torch.channels_last)

        # --- Standard training step ---
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Use 2D augmentation parameters."""
        patch_size = np.array(self._patch_size_2d)

        if max(patch_size) / min(patch_size) > 1.5:
            rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
        else:
            rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)

        do_dummy_2d_data_aug = False  # Already 2D
        mirror_axes = (0, 1)

        # No oversized initial patch needed — spatial transforms run on GPU now
        initial_patch_size = patch_size

        self.print_to_log_file(f'2.5D augmentation: rotation range {rotation_for_DA}, mirror axes {mirror_axes}')
        self.print_to_log_file(f'GPU-side augmentation enabled (spatial + intensity + mirror + DS on GPU)')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
            foreground_labels=None, regions=None, ignore_label=None,
    ) -> BasicTransform:
        """Minimal CPU-side transforms — heavy augmentation runs on GPU in train_step."""
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
        from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

        transforms = []

        # MaskImageTransform *before* augmentation: zero out voxels outside brain mask
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm))
                                   if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(RemoveLabelTansform(-1, 0))

        if regions is not None:
            from batchgeneratorsv2.transforms.utils.seg_to_regions import (
                ConvertSegmentationToRegionsTransform,
            )
            transforms.append(ConvertSegmentationToRegionsTransform(
                regions=list(regions) + [ignore_label]
                if ignore_label is not None else regions,
                channel_in_seg=0,
            ))

        # NOTE: SpatialTransform, intensity transforms, mirror, and DS
        # downsampling are all done on GPU in train_step() via GPUAugmentation.
        # This eliminates the CPU/GIL bottleneck that caused 45% GPU idle time.

        return ComposeTransforms(transforms)

    def get_dataloaders(self):
        """Override to use 2.5D data loader with GPU-side augmentation."""
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size_2d = np.array(self._patch_size_2d)
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # --- GPU Augmentation setup ---
        # Extract 2D deep supervision scales (strip the K-dimension prefix)
        if deep_supervision_scales is not None:
            ds_scales_2d = [s[1:] for s in deep_supervision_scales]  # [[h_s, w_s], ...]
        else:
            ds_scales_2d = []

        self._gpu_aug = GPUAugmentation(
            device=self.device,
            rotation_range=rotation_for_DA,
            mirror_axes=mirror_axes,
            ds_scales_2d=ds_scales_2d,
        )
        self.print_to_log_file(
            f"GPU augmentation: {len(ds_scales_2d)} DS scales, "
            f"rotation {rotation_for_DA}, mirror {mirror_axes}"
        )

        # Minimal CPU transforms (only RemoveLabel + MaskImage)
        patch_size_for_transforms = np.array([self.NUM_ADJACENT_SLICES, *self._patch_size_2d])
        tr_transforms = self.get_training_transforms(
            patch_size_for_transforms,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug=True,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # Optionally preload all volumes into RAM to eliminate blosc2 decompression.
        if self.PRELOAD_TO_RAM:
            dataset_tr = PreloadedDataset(dataset_tr, log_fn=self.print_to_log_file,
                                          slice_axis=self.SLICE_AXIS)
            dataset_val = PreloadedDataset(dataset_val, log_fn=self.print_to_log_file,
                                           slice_axis=self.SLICE_AXIS)

        # Thread-based prefetching for CPU-GPU overlap.
        # With GPU augmentation, CPU threads only do: load .npz → crop → RemoveLabel
        # This is I/O-bound, so threads are very effective.
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

        num_train_threads = self.NUM_PREFETCH_THREADS
        num_val_threads = max(1, num_train_threads // 2)

        if num_train_threads > 0:
            train_augmenters = []
            for _ in range(num_train_threads):
                dl = nnUNetDataLoader25D(
                    dataset_tr, self.batch_size,
                    initial_patch_size.tolist(),
                    self._patch_size_2d,
                    self.label_manager,
                    num_adjacent_slices=self.NUM_ADJACENT_SLICES,
                    oversample_foreground_percent=self.oversample_foreground_percent,
                    sampling_probabilities=None,
                    pad_sides=None,
                    transforms=tr_transforms,
                    probabilistic_oversampling=self.probabilistic_oversampling,
                )
                train_augmenters.append(SingleThreadedAugmenter(dl, None))

            val_augmenters = []
            for _ in range(num_val_threads):
                dl = nnUNetDataLoader25D(
                    dataset_val, self.batch_size,
                    self._patch_size_2d,
                    self._patch_size_2d,
                    self.label_manager,
                    num_adjacent_slices=self.NUM_ADJACENT_SLICES,
                    oversample_foreground_percent=self.oversample_foreground_percent,
                    sampling_probabilities=None,
                    pad_sides=None,
                    transforms=val_transforms,
                    probabilistic_oversampling=self.probabilistic_oversampling,
                )
                val_augmenters.append(SingleThreadedAugmenter(dl, None))

            mt_gen_train = ThreadedPrefetcher(
                train_augmenters, queue_size=num_train_threads + 2
            )
            mt_gen_val = ThreadedPrefetcher(
                val_augmenters, queue_size=num_val_threads + 1
            )
        else:
            dl_tr = nnUNetDataLoader25D(
                dataset_tr, self.batch_size,
                initial_patch_size.tolist(),
                self._patch_size_2d,
                self.label_manager,
                num_adjacent_slices=self.NUM_ADJACENT_SLICES,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling,
            )
            dl_val = nnUNetDataLoader25D(
                dataset_val, self.batch_size,
                self._patch_size_2d,
                self._patch_size_2d,
                self.label_manager,
                num_adjacent_slices=self.NUM_ADJACENT_SLICES,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling,
            )
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)

        self.print_to_log_file(
            f"Data augmentation: {num_train_threads} prefetch threads (train), "
            f"{num_val_threads} (val)"
            + (f", queue buffer={num_train_threads + 2}" if num_train_threads > 0
               else ", single-threaded (no prefetch)")
        )

        # Warm up the generators
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Override to use 2.5D sliding window inference."""
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor25D(
            num_adjacent_slices=self.NUM_ADJACENT_SLICES,
            slice_axis=self.SLICE_AXIS,
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network, self.plans_manager, self.configuration_manager, None,
            self.dataset_json, self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            dataset_val = self.dataset_class(
                self.preprocessed_dataset_folder, val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            )

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(
                    segmentation_export_pool, worker_list, results, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        segmentation_export_pool, worker_list, results, allowed_num_queued=2
                    )

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager,
                             self.plans_manager, self.dataset_json,
                             output_filename_truncated, save_probabilities),
                        )
                    )
                )

            _ = [r.get() for r in results]

        self.set_deep_supervision_enabled(True)
        compute_metrics = self.label_manager.has_regions or not self.label_manager.has_ignore_label
        if compute_metrics:
            from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
            _ = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                validation_output_folder,
                join(validation_output_folder, 'summary.json'),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions if self.label_manager.has_regions
                    else self.label_manager.foreground_labels,
                self.label_manager.ignore_label,
            )
            self.print_to_log_file("Validation complete!")

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-view 2.5D Trainer Subclasses (Axial, Coronal, Sagittal)
# ═══════════════════════════════════════════════════════════════════════════════

class nnUNetTrainer_25D_FoldAll(nnUNetTrainer_25D):
    """
    2.5D trainer for fold_all (trains on all data).
    Early stopping is disabled because there is no held-out validation set,
    making the EMA-based stopping signal unreliable.
    The model runs the full num_epochs (cosine-decay schedule).
    """
    EARLY_STOP_PATIENCE = 10_000   # Effectively infinite
    CURVE_FIT_MIN_EPOCHS = 10_000  # Never triggers curve-fit stopping

    def on_epoch_end(self):
        """Same as parent but clears _early_stopped so training never breaks."""
        super().on_epoch_end()
        self._early_stopped = False  # override any stopping decision


class nnUNetTrainer_25D_Axial(nnUNetTrainer_25D):
    """
    2.5D trainer: Axial view (default)
    K=9 (covers >99% of lesions in axial direction)
    """
    SLICE_AXIS = 0
    NUM_ADJACENT_SLICES = 9

class nnUNetTrainer_25D_Coronal(nnUNetTrainer_25D):
    """
    2.5D trainer: Coronal view
    K=11 (coronal lesions are longer)
    """
    SLICE_AXIS = 1
    NUM_ADJACENT_SLICES = 11

class nnUNetTrainer_25D_Sagittal(nnUNetTrainer_25D):
    """
    2.5D trainer: Sagittal view
    K=9 (matches axial for symmetry)
    """
    SLICE_AXIS = 2
    NUM_ADJACENT_SLICES = 9
