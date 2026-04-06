"""
nnUNetTrainer_WandB - 3D nnUNetTrainer with enhanced augmentation, W&B logging + early stopping.

A subclass that adds:
  - Weights & Biases live logging (train/val loss, dice, LR)
  - Early stopping (patience=100, warmup=50)
  - bf16 autocast when GPU supports it (Ada/Hopper/Blackwell)
  - VRAM-aware batch sizing: probes GPU memory to fill ~80% of available VRAM
  - Dropout3d (p=0.15) injected into the network architecture
  - Enhanced augmentation pipeline (Rician noise, elastic deformation,
    boosted Gaussian noise / gamma probabilities)
  - Gradient checkpointing on encoder stages (~50% activation memory savings)
  - Gradient accumulation (3 steps, effective BS = 3x micro-BS)
  - channels_last_3d memory format for faster cuDNN kernels
  - TF32 tensor core precision + cuDNN benchmark auto-tuning

Optimizations informed by: Kuijf et al., "Standardized Assessment of Automatic
Segmentation of White Matter Hyperintensities", IEEE TMI 2019 (WMH Challenge).
Top methods used: dropout, aggressive augmentation, ensembles.
"""

import gc
import os
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import curve_fit
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.noise.rician import RicianNoiseTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class nnUNetTrainer_WandB(nnUNetTrainer):
    """Standard 3D nnUNetTrainer with Weights & Biases logging and early stopping."""

    # Early stopping settings — same as nnUNetTrainer_25D
    EARLY_STOP_PATIENCE = 150
    EARLY_STOP_WARMUP = 50
    CURVE_FIT_MIN_EPOCHS = 60
    CURVE_FIT_EPSILON = 0.001
    CURVE_FIT_INTERVAL = 10

    # VRAM-aware batch sizing
    VRAM_TARGET_FRACTION = 0.80   # use up to 80% of total VRAM
    VRAM_PROBE_BATCH = 2          # probe with 2 samples to measure per-sample cost
    MIN_BATCH_SIZE = 2
    MAX_BATCH_SIZE = 128          # safety cap (3D patches are large)

    # Gradient accumulation — dynamically set in _probe_vram_and_set_batch_size
    # to match planned batch size: GRAD_ACCUM_STEPS = max(1, plan_bs // optimal_bs)
    GRAD_ACCUM_STEPS = 1

    # ── Network architecture with dropout ────────────────────────────────

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """Inject Dropout3d(p=0.15) into the architecture.

        Top WMH Challenge methods all used dropout for regularization.
        nnU-Net default sets dropout_op=None. We override to add it.
        """
        kwargs = deepcopy(arch_init_kwargs)
        kwargs['dropout_op'] = 'torch.nn.Dropout3d'
        kwargs['dropout_op_kwargs'] = {'p': 0.15, 'inplace': True}

        # Make sure dropout_op is in the list of keys requiring import
        req_import = list(arch_init_kwargs_req_import)
        if 'dropout_op' not in req_import:
            req_import.append('dropout_op')

        return get_network_from_plans(
            architecture_class_name,
            kwargs,
            req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

    # ── Enhanced augmentation pipeline ───────────────────────────────────

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """Enhanced augmentation pipeline vs. stock nnU-Net.

        Changes from default:
          1. Elastic deformation re-enabled (p=0.1, low magnitude)
             — used by WMH Challenge top methods (k2, ipmi-bern)
          2. Rician noise added (p=0.15, variance 0–0.1)
             — simulates MRI noise characteristics for scanner robustness
          3. Gaussian noise probability increased (0.1 → 0.15)
          4. Gamma (normal) probability increased (0.3 → 0.35)
        """
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # Spatial: rotation + scaling + elastic deformation (re-enabled)
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0.1,               # was 0 in stock nnU-Net
                elastic_deform_scale=(0, 0.2),
                elastic_deform_magnitude=(0, 0.2),
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # --- Intensity / noise transforms (enhanced) ---

        # Gaussian noise — probability boosted 0.1 → 0.15
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.15
        ))

        # Rician noise — NEW: simulates MRI-specific noise for scanner robustness
        transforms.append(RandomTransform(
            RicianNoiseTransform(
                noise_variance=(0, 0.1),
            ), apply_probability=0.15
        ))

        # Gaussian blur (unchanged)
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))

        # Multiplicative brightness (unchanged)
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))

        # Contrast (unchanged)
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))

        # Simulate low resolution (unchanged)
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))

        # Gamma with inversion (unchanged)
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))

        # Gamma without inversion — probability boosted 0.3 → 0.35
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.35
        ))

        # Mirroring (unchanged)
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(allowed_axes=mirror_axes)
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, \
                'We need foreground_labels for cascade augmentations'
            from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
            from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
            from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
                RemoveRandomConnectedComponentFromOneHotEncodingTransform
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            from batchgeneratorsv2.transforms.utils.region_based_training import \
                ConvertSegmentationToRegionsTransform
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform(
                    ds_scales=deep_supervision_scales
                )
            )

        if ignore_label is not None:
            transforms.append(
                RemoveLabelTansform(ignore_label, 0)
            )

        return ComposeTransforms(transforms)

    # ── VRAM-aware batch sizing ──────────────────────────────────────────

    def initialize(self):
        """Override to add gradient checkpointing, channels_last, TF32,
        cuDNN benchmark, and VRAM probe after parent initialization."""
        super().initialize()

        # ── TF32 tensor cores + cuDNN benchmark ──
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')       # TF32
            torch.backends.cudnn.benchmark = True            # auto-tune kernels
            self.print_to_log_file(
                "GPU optimization: TF32 matmul + cuDNN benchmark enabled"
            )

        # ── channels_last_3d memory layout ──
        # NHWC layout avoids transpose overhead for cuDNN convolutions
        if self.device.type == 'cuda':
            raw_net = getattr(self.network, '_orig_mod', self.network)
            raw_net.to(memory_format=torch.channels_last_3d)
            self.print_to_log_file(
                "GPU optimization: channels_last_3d memory format enabled"
            )

        # ── Gradient checkpointing on encoder stages ──
        # Trades ~25% compute for ~50% activation memory savings.
        # Critical for fitting BS=2 with 160x192x160 patches on 16GB.
        self._enable_gradient_checkpointing()

        # ── VRAM probe ──
        self._probe_vram_and_set_batch_size()

        # Log effective batch size with accumulation
        eff_bs = self.batch_size * self.GRAD_ACCUM_STEPS
        self.print_to_log_file(
            f"Gradient accumulation: {self.GRAD_ACCUM_STEPS} steps, "
            f"micro-BS={self.batch_size}, effective BS={eff_bs}"
        )

    def _enable_gradient_checkpointing(self):
        """Wrap encoder stages with gradient checkpointing.

        Instead of storing activations for all encoder stages during forward,
        recompute them during backward. Saves ~50% activation memory at
        the cost of ~25-30% extra compute.
        """
        raw_net = getattr(self.network, '_orig_mod', self.network)
        encoder = raw_net.encoder

        if not hasattr(encoder, 'stages'):
            self.print_to_log_file(
                "WARNING: encoder has no 'stages' attribute "
                "-- gradient checkpointing not applied"
            )
            return

        original_forward = encoder.forward

        def checkpointed_forward(self_enc, x):
            if hasattr(self_enc, 'stem') and self_enc.stem is not None:
                x = self_enc.stem(x)
            ret = []
            for s in self_enc.stages:
                x = grad_checkpoint(s, x, use_reentrant=False)
                ret.append(x)
            if self_enc.return_skips:
                return ret
            else:
                return ret[-1]

        import types
        encoder.forward = types.MethodType(checkpointed_forward, encoder)
        self.print_to_log_file(
            f"GPU optimization: gradient checkpointing enabled on "
            f"{len(encoder.stages)} encoder stages"
        )

    def _probe_vram_and_set_batch_size(self):
        """Differential VRAM probe: run with n=1 and n=2 to isolate
        the true marginal per-sample cost, eliminating fixed overhead
        (CUDA context, cuDNN workspace, one-time activation buffers).

        Falls back gracefully: if n=2 OOMs, uses n=1 peak to estimate.
        If n=1 OOMs, forces MIN_BATCH_SIZE and hopes for the best.
        """
        if self.device.type != 'cuda':
            self.print_to_log_file(
                f"No CUDA device -- keeping plan batch size: {self.batch_size}"
            )
            return

        plan_bs = self.batch_size
        patch_size = self.configuration_manager.patch_size
        num_input_channels = self.num_input_channels

        amp_dtype = torch.float16
        if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        # Use uncompiled model for probing
        probe_model = getattr(self.network, '_orig_mod', self.network)
        is_compiled = probe_model is not self.network

        def _run_probe(n):
            """Forward+backward with n samples, return peak memory."""
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(self.device)

            dummy = torch.randn(
                n, num_input_channels, *patch_size,
                device=self.device, dtype=torch.float32
            )
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
        total_vram = torch.cuda.get_device_properties(self.device).total_memory
        target_mem = total_vram * self.VRAM_TARGET_FRACTION

        try:
            # Probe with n=1
            peak_1 = _run_probe(1)

            # Try n=2 for differential measurement
            try:
                peak_2 = _run_probe(2)
                per_sample_mem = peak_2 - peak_1
                fixed_overhead = peak_1 - mem_before - per_sample_mem
                fixed_overhead = max(fixed_overhead, 0)
                method = "differential (n=1 vs n=2)"
            except RuntimeError:
                # n=2 OOM: estimate from n=1 alone (conservative)
                torch.cuda.empty_cache()
                gc.collect()
                per_sample_mem = (peak_1 - mem_before) * 0.6  # ~60% is per-sample
                fixed_overhead = (peak_1 - mem_before) * 0.4
                peak_2 = None
                method = "single-sample estimate (n=2 OOM)"

            available_for_batch = target_mem - mem_before - fixed_overhead
            if per_sample_mem > 0 and available_for_batch > 0:
                optimal_bs = int(available_for_batch / per_sample_mem)
                optimal_bs = max(self.MIN_BATCH_SIZE, min(optimal_bs, self.MAX_BATCH_SIZE))
            else:
                optimal_bs = self.MIN_BATCH_SIZE

            self.batch_size = optimal_bs
            self.oversample_foreground_percent = float(1 / self.batch_size)

            # Dynamically set gradient accumulation to approximate plan batch size
            # plan_bs=3, optimal_bs=2 → accum=1 (no accumulation, 250 optimizer steps/epoch)
            # plan_bs=6, optimal_bs=2 → accum=3 (effective BS=6, still 83 steps/epoch)
            self.GRAD_ACCUM_STEPS = max(1, plan_bs // optimal_bs)

            log_msg = (
                f"VRAM probe results ({method}{', eager-mode' if is_compiled else ''}):"
                f"\n  Total VRAM: {total_vram / 1024**3:.1f} GB"
                f"\n  Model + optimizer: {mem_before / 1024**3:.2f} GB"
                f"\n  Fixed overhead: {fixed_overhead / 1024**2:.1f} MB"
                f"\n  Peak n=1: {peak_1 / 1024**3:.2f} GB"
            )
            if peak_2 is not None:
                log_msg += f"\n  Peak n=2: {peak_2 / 1024**3:.2f} GB"
            log_msg += (
                f"\n  Marginal per-sample cost: {per_sample_mem / 1024**2:.1f} MB"
                f"\n  Target ({self.VRAM_TARGET_FRACTION*100:.0f}% VRAM): {target_mem / 1024**3:.1f} GB"
                f"\n  Plan batch size: {plan_bs} -> Optimal: {optimal_bs}"
            )
            self.print_to_log_file(log_msg)

        except Exception as e:
            # Even n=1 failed -- force minimum batch size
            self.print_to_log_file(
                f"VRAM probe failed ({e}), forcing batch size: {self.MIN_BATCH_SIZE}"
            )
            self.batch_size = self.MIN_BATCH_SIZE
            self.oversample_foreground_percent = float(1 / self.batch_size)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # ── W&B integration ──────────────────────────────────────────────────

    def _init_wandb(self):
        """Initialize W&B run. Called from on_train_start."""
        if not WANDB_AVAILABLE:
            self.print_to_log_file("wandb not installed — skipping W&B logging")
            self._wandb_run = None
            return

        if os.environ.get('WANDB_MODE', '').lower() == 'disabled':
            self.print_to_log_file("WANDB_MODE=disabled — skipping W&B logging")
            self._wandb_run = None
            return

        try:
            ds_name = self.plans_manager.dataset_name
            plans_name = self.plans_manager.plans_name
            run_name = (
                f"{self.__class__.__name__}__{plans_name}"
                f"__{self.configuration_name}__fold{self.fold}"
            )
            self._wandb_run = wandb.init(
                project="ms-lesion-seg",
                name=run_name,
                group=ds_name,
                tags=[ds_name, self.__class__.__name__, plans_name,
                      self.configuration_name],
                config={
                    "dataset": ds_name,
                    "trainer": self.__class__.__name__,
                    "plans": plans_name,
                    "configuration": self.configuration_name,
                    "fold": self.fold,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "early_stop_patience": self.EARLY_STOP_PATIENCE,
                    "early_stop_warmup": self.EARLY_STOP_WARMUP,
                },
                resume="allow",
                reinit=True,
                settings=wandb.Settings(init_timeout=120),
            )
            self.print_to_log_file(
                f"W&B run initialized: {self._wandb_run.url}"
            )
        except Exception as e:
            self.print_to_log_file(
                f"W&B init failed: {e} — continuing without W&B"
            )
            self._wandb_run = None

    # ── Curve-fitting early stopping ─────────────────────────────────────

    @staticmethod
    def _asymptotic_model(x, a, b, c):
        """Exponential saturation model: y = a - b * exp(-c * x).
        a = predicted plateau, b = total improvement range, c = convergence rate."""
        return a - b * np.exp(-c * x)

    def _predict_convergence(self, ema_history: list) -> dict | None:
        """Fit asymptotic curve to EMA dice history and predict remaining improvement."""
        n = len(ema_history)
        if n < self.CURVE_FIT_MIN_EPOCHS:
            return None

        y = np.array(ema_history, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)

        y_max = y.max()
        y_min = y[:max(1, n // 10)].mean()
        a_init = y_max * 1.02
        b_init = max(a_init - y_min, 0.01)
        c_init = 3.0 / n

        try:
            popt, pcov = curve_fit(
                self._asymptotic_model, x, y,
                p0=[a_init, b_init, c_init],
                bounds=([y_max * 0.95, 0.001, 1e-6],
                        [1.0, 2.0, 1.0]),
                maxfev=5000,
            )
            a, b, c = popt
            current = y[-1]
            predicted_remaining = a - current
            y_pred = self._asymptotic_model(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return {
                'plateau': a,
                'remaining': predicted_remaining,
                'rate': c,
                'r_squared': r_squared,
                'current': current,
            }
        except (RuntimeError, ValueError):
            return None

    # ── Training lifecycle hooks ─────────────────────────────────────────

    def on_train_start(self):
        """Initialize W&B, early stopping, and bf16 autocast."""
        super().on_train_start()
        self._init_wandb()

        # ── bf16 autocast ──
        # Ada/Hopper/Blackwell GPUs natively support bf16 tensor cores.
        # bf16 has the same exponent range as fp32 (no GradScaler needed),
        # unlike fp16 which requires loss scaling to avoid underflow.
        if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
            self._use_bf16 = True
            self.grad_scaler = None  # bf16 doesn't need loss scaling
            torch.set_autocast_dtype('cuda', torch.bfloat16)
            self.print_to_log_file(
                "GPU optimization: bf16 autocast enabled (no GradScaler)"
            )
        else:
            self._use_bf16 = False
            self.print_to_log_file(
                "GPU optimization: using default fp16 autocast + GradScaler"
            )

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
        """Add W&B logging and early stopping check after each epoch."""
        super().on_epoch_end()

        log = self.logger.my_fantastic_logging
        epoch = self.current_epoch - 1  # parent already incremented

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
                if (len(log['epoch_end_timestamps']) > 0
                        and len(log['epoch_start_timestamps']) > 0):
                    metrics["epoch_time_s"] = (
                        log['epoch_end_timestamps'][-1]
                        - log['epoch_start_timestamps'][-1]
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
                    # Stop if predicted remaining improvement is negligible
                    # and fit is good (R² > 0.9)
                    if (pred['remaining'] < self.CURVE_FIT_EPSILON
                            and pred['r_squared'] > 0.9):
                        self.print_to_log_file(
                            f"\n{'='*60}\n"
                            f"EARLY STOPPING (curve-fit) at epoch {epoch}\n"
                            f"Predicted plateau: {pred['plateau']:.4f}\n"
                            f"Current EMA Dice: {current_ema:.4f}\n"
                            f"Predicted remaining: {pred['remaining']:.4f} < {self.CURVE_FIT_EPSILON}\n"
                            f"Fit R2: {pred['r_squared']:.3f}\n"
                            f"Best EMA Dice: {self._es_best_ema:.4f} at epoch {self._es_best_epoch}\n"
                            f"{'='*60}"
                        )
                        self._early_stopped = True

            # Patience fallback (in case curve fit never triggers)
            if not self._early_stopped:
                epochs_since_best = epoch - self._es_best_epoch
                if epochs_since_best >= self.EARLY_STOP_PATIENCE:
                    self.print_to_log_file(
                        f"\n{'='*60}\n"
                        f"EARLY STOPPING (patience) at epoch {epoch}\n"
                        f"No improvement in ema_fg_dice for "
                        f"{self.EARLY_STOP_PATIENCE} epochs\n"
                        f"Best EMA Dice: {self._es_best_ema:.4f} "
                        f"at epoch {self._es_best_epoch}\n"
                        f"{'='*60}"
                    )
                    self._early_stopped = True

    def on_train_end(self):
        """Finalize W&B run."""
        super().on_train_end()
        if hasattr(self, '_wandb_run') and self._wandb_run is not None:
            try:
                wandb.finish()
                self.print_to_log_file("W&B run finished.")
            except Exception as e:
                self.print_to_log_file(f"W&B finish error: {e}")

    def train_step(self, batch: dict) -> dict:
        """Micro-step for gradient accumulation.

        Unlike the parent, this does NOT call optimizer.step().
        The accumulation logic in run_training handles optimizer stepping
        every GRAD_ACCUM_STEPS micro-steps.
        """
        from torch.amp import autocast
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import dummy_context

        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        # channels_last_3d for cuDNN performance
        if self.device.type == 'cuda':
            data = data.to(memory_format=torch.channels_last_3d)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast('cuda', enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        # Scale loss by accumulation steps for correct gradient magnitude
        l_scaled = l / self.GRAD_ACCUM_STEPS

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l_scaled).backward()
        else:
            l_scaled.backward()

        return {'loss': l.detach().cpu().numpy()}  # log unscaled loss

    def run_training(self):
        """Override with gradient accumulation + early stopping.

        Accumulates gradients over GRAD_ACCUM_STEPS micro-batches before
        stepping the optimizer. This simulates a larger effective batch
        size without increasing peak VRAM usage.
        """
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []

            self.optimizer.zero_grad(set_to_none=True)
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(
                    self.train_step(next(self.dataloader_train))
                )

                # Optimizer step every GRAD_ACCUM_STEPS or at epoch end
                is_accum_step = (
                    (batch_id + 1) % self.GRAD_ACCUM_STEPS == 0
                    or batch_id == self.num_iterations_per_epoch - 1
                )
                if is_accum_step:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.network.parameters(), 12
                        )
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.network.parameters(), 12
                        )
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(
                        self.validation_step(next(self.dataloader_val))
                    )
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            # Early stopping check
            if hasattr(self, '_early_stopped') and self._early_stopped:
                self.print_to_log_file("Training stopped early.")
                break

        self.on_train_end()
