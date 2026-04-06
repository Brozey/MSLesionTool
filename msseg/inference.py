"""
msseg.inference – nnUNet predictor factories and segmentation thread.
"""
import os
import logging
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from .constants import ARCHITECTURES, resolve_model_dir

_logger = logging.getLogger("MSLesionTool")

# Optional imports
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    _HAS_TORCH = False

try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    _HAS_NNUNET = True
except ImportError:
    nnUNetPredictor = None
    _HAS_NNUNET = False


# ─── Hardware detection ───────────────────────────────────────────

def detect_best_device():
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            a = torch.ones(2, 2, device="cuda")
            _ = (a @ a).sum().item()
            del a
            return "cuda"
        except Exception:
            pass
    return "cpu"


# ─── Predictor factories ─────────────────────────────────────────

def _verify_cuda(device_str):
    """Verify CUDA actually works, fall back to CPU if not."""
    if device_str != "cuda":
        return device_str
    try:
        a = torch.ones(2, 2, device="cuda")
        _ = (a @ a).sum().item(); del a
        return "cuda"
    except Exception:
        return "cpu"


def _safe_initialize_predictor(predictor, model_dir, folds, checkpoint_name="checkpoint_best.pth"):
    """Initialize predictor with fallback for missing custom trainer classes.

    Custom trainers (nnUNetTrainer_WandB, etc.) may not be available in
    frozen .exe builds.  Their build_network_architecture is identical to
    the base nnUNetTrainer, so we fall back to get_network_from_plans.
    """
    from os.path import join
    from nnunetv2.utilities.file_path_utilities import load_json
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2

    if folds is None:
        folds = nnUNetPredictor.auto_detect_available_folds(model_dir, checkpoint_name)

    dataset_json = load_json(join(model_dir, 'dataset.json'))
    plans = load_json(join(model_dir, 'plans.json'))
    plans_manager = PlansManager(plans)

    parameters = []
    trainer_name = None
    inference_allowed_mirroring_axes = None
    for i, f in enumerate(folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(
            join(model_dir, f'fold_{f}', checkpoint_name),
            map_location=torch.device('cpu'), weights_only=False)
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            inference_allowed_mirroring_axes = checkpoint.get(
                'inference_allowed_mirroring_axes', None)
        parameters.append(checkpoint['network_weights'])

    configuration_name = '3d_fullres'
    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = len(dataset_json['channel_names'])

    # Try to find trainer class; fall back if not available (e.g. in .exe)
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name, 'nnunetv2.training.nnUNetTrainer')

    if trainer_class is not None:
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False)
    else:
        _logger.warning("Trainer class '%s' not found, using generic network builder", trainer_name)
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            deep_supervision=False)

    predictor.plans_manager = plans_manager
    predictor.configuration_manager = configuration_manager
    predictor.list_of_parameters = parameters
    predictor.network = network
    predictor.dataset_json = dataset_json
    predictor.trainer_name = trainer_name
    predictor.allowed_mirroring_axes = inference_allowed_mirroring_axes
    predictor.label_manager = plans_manager.get_label_manager(dataset_json)
    network.load_state_dict(parameters[0])


def _create_predictor_3d(model_dir, device_str, folds=(0,)):
    device_str = _verify_cuda(device_str)
    device = torch.device(device_str)
    on_device = device_str != "cpu"

    if device_str == "cpu":
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        step_sz = 0.8
    else:
        step_sz = 0.5

    predictor = nnUNetPredictor(
        tile_step_size=step_sz, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=on_device,
        device=device, verbose=False, verbose_preprocessing=False,
        allow_tqdm=False,
    )
    _safe_initialize_predictor(predictor, model_dir, tuple(folds))
    return predictor, device_str


# ─── 2.5D predictor ──────────────────────────────────────────────

_SLICE_AXIS_PERM = {
    0: None,
    1: (0, 2, 1, 3),
    2: (0, 3, 2, 1),
}

if _HAS_NNUNET and _HAS_TORCH:
    from os.path import join
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.helpers import empty_cache, dummy_context
    from tqdm import tqdm

    class nnUNetPredictor25D(nnUNetPredictor):
        """Predictor that performs slice-by-slice inference with K-slice context."""

        def __init__(self, num_adjacent_slices: int = 7, slice_axis: int = 0, **kwargs):
            super().__init__(**kwargs)
            self.num_adjacent_slices = num_adjacent_slices
            self.half_k = num_adjacent_slices // 2
            self.slice_axis = slice_axis
            self._perm = _SLICE_AXIS_PERM[slice_axis]

        def initialize_from_trained_model_folder(
            self, model_training_output_dir: str,
            use_folds=None, checkpoint_name: str = 'checkpoint_best.pth',
        ):
            from nnunetv2.utilities.file_path_utilities import load_json
            import nnunetv2

            if use_folds is None:
                use_folds = nnUNetPredictor.auto_detect_available_folds(
                    model_training_output_dir, checkpoint_name)

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
                    map_location=torch.device('cpu'), weights_only=False)
                if i == 0:
                    trainer_name = checkpoint['trainer_name']
                    inference_allowed_mirroring_axes = checkpoint.get(
                        'inference_allowed_mirroring_axes', None)
                parameters.append(checkpoint['network_weights'])

            configuration_manager = plans_manager.get_configuration('2d')
            num_input_channels = len(dataset_json['channel_names']) * self.num_adjacent_slices

            from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
            trainer_class = recursive_find_python_class(
                join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                trainer_name, 'nnunetv2.training.nnUNetTrainer')
            if trainer_class is None:
                from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
                network = get_network_from_plans(
                    configuration_manager.network_arch_class_name,
                    configuration_manager.network_arch_init_kwargs,
                    configuration_manager.network_arch_init_kwargs_req_import,
                    num_input_channels,
                    plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                    deep_supervision=False)
            else:
                network = trainer_class.build_network_architecture(
                    configuration_manager.network_arch_class_name,
                    configuration_manager.network_arch_init_kwargs,
                    configuration_manager.network_arch_init_kwargs_req_import,
                    num_input_channels,
                    plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                    enable_deep_supervision=False)

            self.plans_manager = plans_manager
            self.configuration_manager = plans_manager.get_configuration('3d_fullres')
            self.configuration_manager_2d = configuration_manager
            self.list_of_parameters = parameters
            self.network = network
            self.dataset_json = dataset_json
            self.trainer_name = trainer_name
            self.allowed_mirroring_axes = inference_allowed_mirroring_axes
            self.label_manager = plans_manager.get_label_manager(dataset_json)
            network.load_state_dict(parameters[0])

        @torch.inference_mode()
        def predict_sliding_window_return_logits(self, input_image: torch.Tensor):
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()
            empty_cache(self.device)

            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be 4D (C, D, H, W)'
                if self._perm is not None:
                    input_image = input_image.permute(self._perm).contiguous()

                C, D, H, W = input_image.shape
                K = self.num_adjacent_slices
                half_k = self.half_k

                patch_size_2d = self.configuration_manager_2d.patch_size
                if len(patch_size_2d) == 3:
                    patch_size_2d = patch_size_2d[1:]
                elif len(patch_size_2d) != 2:
                    raise ValueError(f"Unexpected patch_size dimensionality: {patch_size_2d}")

                from nnunetv2.inference.sliding_window_prediction import (
                    compute_steps_for_sliding_window, compute_gaussian)

                steps = compute_steps_for_sliding_window(
                    (H, W), patch_size_2d, self.tile_step_size)

                results_device = self.device
                gaussian = compute_gaussian(
                    tuple(patch_size_2d), sigma_scale=1. / 8,
                    value_scaling_factor=10, device=results_device,
                    dtype=torch.float32
                ).float() if self.use_gaussian else 1

                num_seg_heads = self.label_manager.num_segmentation_heads
                predicted_logits = torch.zeros(
                    (num_seg_heads, D, H, W), dtype=torch.float32, device=results_device)
                n_predictions = torch.zeros(
                    (D, H, W), dtype=torch.float32, device=results_device)

                pad_h = max(0, patch_size_2d[0] - H)
                pad_w = max(0, patch_size_2d[1] - W)
                if pad_h > 0 or pad_w > 0:
                    input_image = F.pad(input_image, (0, pad_w, 0, pad_h), mode='constant', value=0)
                    _, _, H_padded, W_padded = input_image.shape
                    steps = compute_steps_for_sliding_window(
                        (H_padded, W_padded), patch_size_2d, self.tile_step_size)
                    predicted_logits_padded = torch.zeros(
                        (num_seg_heads, D, H_padded, W_padded),
                        dtype=torch.float32, device=results_device)
                    n_predictions_padded = torch.zeros(
                        (D, H_padded, W_padded), dtype=torch.float32, device=results_device)
                else:
                    H_padded, W_padded = H, W
                    predicted_logits_padded = predicted_logits
                    n_predictions_padded = n_predictions

                total_steps = D * len(steps[0]) * len(steps[1])
                with tqdm(total=total_steps, desc="2.5D inference",
                          disable=not self.allow_tqdm) as pbar:
                    for z in range(D):
                        slices_per_channel = []
                        for c in range(C):
                            for dz in range(-half_k, half_k + 1):
                                zz = z + dz
                                if 0 <= zz < D:
                                    slices_per_channel.append(input_image[c, zz])
                                else:
                                    slices_per_channel.append(torch.zeros_like(input_image[0, 0]))
                        context_input = torch.stack(slices_per_channel, dim=0)

                        for sx in steps[0]:
                            for sy in steps[1]:
                                patch = context_input[
                                    None, :,
                                    sx:sx + patch_size_2d[0],
                                    sy:sy + patch_size_2d[1]
                                ].to(self.device, non_blocking=True)

                                prediction = self._internal_maybe_mirror_and_predict(patch)[0]
                                prediction = prediction.to(results_device)

                                if self.use_gaussian:
                                    prediction *= gaussian

                                predicted_logits_padded[
                                    :, z,
                                    sx:sx + patch_size_2d[0],
                                    sy:sy + patch_size_2d[1]
                                ] += prediction

                                n_predictions_padded[
                                    z,
                                    sx:sx + patch_size_2d[0],
                                    sy:sy + patch_size_2d[1]
                                ] += gaussian if isinstance(gaussian, torch.Tensor) else 1

                                pbar.update()

                torch.div(predicted_logits_padded, n_predictions_padded, out=predicted_logits_padded)

                if pad_h > 0 or pad_w > 0:
                    predicted_logits = predicted_logits_padded[:, :, :H, :W]
                else:
                    predicted_logits = predicted_logits_padded

                if torch.any(torch.isinf(predicted_logits)):
                    raise RuntimeError('Encountered inf in predicted array.')

                if self._perm is not None:
                    predicted_logits = predicted_logits.permute(self._perm).contiguous()

            return predicted_logits


def _create_predictor_25d(model_dir, device_str, folds=(0,), K=7, slice_axis=0):
    """Create a 2.5D predictor for slice-by-slice inference with K-context."""
    device_str = _verify_cuda(device_str)
    device = torch.device(device_str)
    on_device = device_str != "cpu"

    if device_str == "cpu":
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        step_sz = 0.8
    else:
        step_sz = 0.5

    predictor = nnUNetPredictor25D(
        num_adjacent_slices=K, slice_axis=slice_axis,
        tile_step_size=step_sz, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=on_device,
        device=device, verbose=False, verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=tuple(folds), checkpoint_name="checkpoint_best.pth",
    )
    return predictor, device_str


def create_predictor(arch_key, model_dir, device_str, folds=(0,)):
    """Dispatcher: create the right predictor type for the architecture."""
    pred_type = ARCHITECTURES[arch_key][1]
    if pred_type == "25d":
        return _create_predictor_25d(model_dir, device_str, folds)
    else:
        return _create_predictor_3d(model_dir, device_str, folds)


# ─── Worker threads ───────────────────────────────────────────────

class PreloadModelThread(QThread):
    """Background thread to preload model weights into RAM."""
    progress = pyqtSignal(str)

    def __init__(self, ensemble_config, device_str, predictors_cache=None, parent=None):
        super().__init__(parent)
        self._ensemble_config = ensemble_config
        self._device_str = device_str
        self._predictors = predictors_cache if predictors_cache is not None else {}

    def run(self):
        try:
            total = sum(len(folds) for folds in self._ensemble_config.values())
            loaded = 0
            for arch_key, folds in self._ensemble_config.items():
                subdir = ARCHITECTURES[arch_key][0]
                display = ARCHITECTURES[arch_key][2]
                model_dir = resolve_model_dir(subdir)
                if model_dir is None:
                    self.progress.emit(f"WARNING: {display} not found, skipping")
                    continue
                arch_cache = self._predictors.setdefault(arch_key, {})
                for fold in folds:
                    if fold in arch_cache:
                        loaded += 1
                        continue
                    self.progress.emit(f"Preloading [{loaded+1}/{total}] {display} fold {fold}...")
                    predictor, _ = create_predictor(arch_key, model_dir, self._device_str, folds=(fold,))
                    arch_cache[fold] = predictor
                    loaded += 1
            self.progress.emit(f"All {loaded} models preloaded into RAM.")
        except Exception as e:
            self.progress.emit(f"Preload failed: {e}")


class SegmentationThread(QThread):
    """Background thread that runs multi-architecture ensemble inference."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict, float)  # {(arch, fold): probs}, elapsed_time

    def __init__(self, flair, t1, spacing, ensemble_config, device_str,
                 predictors_cache, parent=None):
        super().__init__(parent)
        self._flair = flair
        self._t1 = t1
        self._spacing = spacing
        self._ensemble_config = ensemble_config
        self._device_str = device_str
        self._predictors_cache = predictors_cache

    def run(self):
        import time
        t0 = time.time()
        fold_probs = {}
        try:
            img = np.stack([self._flair, self._t1], axis=0).astype(np.float32)
            props = {'spacing': list(self._spacing)}

            total = sum(len(folds) for folds in self._ensemble_config.values())
            done = 0

            for arch_key, folds in self._ensemble_config.items():
                subdir = ARCHITECTURES[arch_key][0]
                display = ARCHITECTURES[arch_key][2]
                model_dir = resolve_model_dir(subdir)
                if model_dir is None:
                    continue

                arch_cache = self._predictors_cache.setdefault(arch_key, {})

                for fold in folds:
                    done += 1
                    self.progress.emit(f"[{done}/{total}] {display} fold {fold} – predicting...")

                    if fold not in arch_cache:
                        predictor, _ = create_predictor(
                            arch_key, model_dir, self._device_str, folds=(fold,))
                        arch_cache[fold] = predictor
                    else:
                        predictor = arch_cache[fold]

                    if self._device_str == "cuda":
                        predictor.network.half()
                    else:
                        predictor.network.float()
                    _seg, probs = predictor.predict_single_npy_array(
                        img, props, None, None, True)
                    fold_probs[(arch_key, fold)] = probs

            elapsed = time.time() - t0
            self.finished.emit(fold_probs, elapsed)
        except Exception as e:
            _logger.error("Segmentation error: %s", e)
            self.finished.emit({}, 0.0)
