"""
msseg.inference_ort – ONNX Runtime inference backend.

Subclasses the nnUNet predictors, replacing only the network forward pass
with ONNX Runtime sessions. All preprocessing and postprocessing remain
identical to the PyTorch path, guaranteeing numerical equivalence.
"""
import os
import logging
import numpy as np

_logger = logging.getLogger("MSLesionTool")

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    ort = None
    _HAS_ORT = False

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

from .constants import ARCHITECTURES, resolve_model_dir


def get_ort_providers():
    """Return the best available ORT execution providers."""
    available = ort.get_available_providers() if _HAS_ORT else []
    providers = []
    if 'TensorrtExecutionProvider' in available:
        providers.append('TensorrtExecutionProvider')
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    return providers


def check_onnx_available(model_dir, fold):
    """Check if an ONNX model exists for the given model dir and fold."""
    return os.path.isfile(os.path.join(model_dir, f'fold_{fold}', 'model.onnx'))


class _OnnxNetworkProxy:
    """A minimal object that stands in for a PyTorch network during ORT inference.
    nnUNet's predictor expects self.network to have .half(), .to(), .load_state_dict() etc."""
    def half(self):
        return self
    def to(self, *args, **kwargs):
        return self
    def eval(self):
        return self
    def load_state_dict(self, *args, **kwargs):
        pass  # ORT weights are baked into the session
    def parameters(self):
        return iter([])  # no PyTorch parameters
    def __call__(self, x):
        raise RuntimeError("Direct call not supported - use ORT session")


if _HAS_ORT and _HAS_NNUNET and _HAS_TORCH:
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.file_path_utilities import load_json
    from .inference import nnUNetPredictor25D, _SLICE_AXIS_PERM

    # TTA mirror axis combinations (matching nnUNet convention)
    # For 3D patches (batch, C, D, H, W): mirror spatial dims 2,3,4
    _MIRROR_AXES_3D = [(), (4,), (3,), (2,), (4,3), (4,2), (3,2), (4,3,2)]
    # For 2D patches (batch, C, H, W): mirror spatial dims 2,3
    _MIRROR_AXES_2D = [(), (3,), (2,), (3,2)]

    class ORTPredictor3D(nnUNetPredictor):
        """3D nnUNet predictor using ONNX Runtime for the forward pass."""

        def __init__(self, ort_session, **kwargs):
            super().__init__(**kwargs)
            self._ort_session = ort_session
            self._input_name = ort_session.get_inputs()[0].name
            self._output_name = ort_session.get_outputs()[0].name
            # Detect session dtype (fp16 models use 'tensor(float16)')
            inp_type = ort_session.get_inputs()[0].type
            self._use_fp16 = 'float16' in inp_type
            self._np_dtype = np.float16 if self._use_fp16 else np.float32
            self._torch_dtype = torch.float16 if self._use_fp16 else torch.float32

        def _use_gpu_iobinding(self):
            if not _HAS_TORCH or not torch.cuda.is_available():
                return False
            try:
                active = self._ort_session.get_providers()
                return any('CUDA' in p or 'Tensorrt' in p for p in active)
            except Exception:
                return False

        def _run_patch_gpu(self, patch_gpu, out_shape):
            """Run one patch through ORT on GPU via IOBinding. Returns GPU tensor."""
            if self._use_fp16:
                patch_gpu = patch_gpu.half()
            out_gpu = torch.empty(out_shape, dtype=self._torch_dtype, device='cuda')
            io_binding = self._ort_session.io_binding()
            io_binding.bind_input(
                self._input_name, 'cuda', 0,
                self._np_dtype, list(patch_gpu.shape), patch_gpu.data_ptr())
            io_binding.bind_output(
                self._output_name, 'cuda', 0,
                self._np_dtype, out_shape, out_gpu.data_ptr())
            self._ort_session.run_with_iobinding(io_binding)
            return out_gpu

        def _predict_patch_gpu(self, patch_gpu, out_shape):
            """Run patch with optional TTA mirroring on GPU."""
            if not self.use_mirroring:
                return self._run_patch_gpu(patch_gpu, out_shape)[0].cpu()
            # TTA: mirror along all axis combos, average (keep batch dim for correct flip axes)
            # fp16 models may produce NaN on certain mirror orientations — skip those
            result = self._run_patch_gpu(patch_gpu, out_shape).float()
            count = 1
            for axes in _MIRROR_AXES_3D[1:]:
                flipped = torch.flip(patch_gpu, list(axes)).contiguous()
                out = self._run_patch_gpu(flipped, out_shape).float()
                out_unflipped = torch.flip(out, list(axes))
                if not torch.isnan(out_unflipped).any():
                    result += out_unflipped
                    count += 1
            result /= count
            return result[0].cpu()

        def _predict_patch_cpu(self, patch_np):
            """Run patch with optional TTA mirroring on CPU."""
            patch_np = patch_np.astype(self._np_dtype)
            logits = self._ort_session.run(
                [self._output_name], {self._input_name: patch_np})[0]
            if not self.use_mirroring:
                return torch.from_numpy(logits[0].astype(np.float32))
            # Keep batch dim for correct flip axes; skip mirrors that produce NaN
            result = torch.from_numpy(logits.astype(np.float32))
            count = 1
            patch_t = torch.from_numpy(patch_np.astype(np.float32))
            for axes in _MIRROR_AXES_3D[1:]:
                flipped = torch.flip(patch_t, list(axes)).contiguous().numpy().astype(self._np_dtype)
                out = self._ort_session.run(
                    [self._output_name], {self._input_name: flipped})[0]
                out_t = torch.flip(torch.from_numpy(out.astype(np.float32)), list(axes))
                if not torch.isnan(out_t).any():
                    result += out_t
                    count += 1
            result /= count
            return result[0]

        def predict_sliding_window_return_logits(self, input_image):
            """Override: use ORT session instead of PyTorch network."""
            from nnunetv2.inference.sliding_window_prediction import (
                compute_steps_for_sliding_window, compute_gaussian)

            assert isinstance(input_image, torch.Tensor)
            assert input_image.ndim == 4

            use_gpu = self._use_gpu_iobinding()

            C, D, H, W = input_image.shape
            patch_size = self.configuration_manager.patch_size

            steps = compute_steps_for_sliding_window(
                [D, H, W], patch_size, self.tile_step_size)

            results_device = torch.device('cpu')
            gaussian = compute_gaussian(
                tuple(patch_size), sigma_scale=1. / 8,
                value_scaling_factor=10, device=results_device, dtype=torch.float32
            ).float() if self.use_gaussian else 1

            num_seg_heads = self.label_manager.num_segmentation_heads
            predicted_logits = torch.zeros(
                (num_seg_heads, D, H, W), dtype=torch.float32, device=results_device)
            n_predictions = torch.zeros(
                (D, H, W), dtype=torch.float32, device=results_device)

            pad_d = max(0, patch_size[0] - D)
            pad_h = max(0, patch_size[1] - H)
            pad_w = max(0, patch_size[2] - W)
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                input_image = F.pad(input_image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
                _, D_p, H_p, W_p = input_image.shape
                steps = compute_steps_for_sliding_window(
                    [D_p, H_p, W_p], patch_size, self.tile_step_size)
                predicted_logits = torch.zeros(
                    (num_seg_heads, D_p, H_p, W_p), dtype=torch.float32, device=results_device)
                n_predictions = torch.zeros(
                    (D_p, H_p, W_p), dtype=torch.float32, device=results_device)
            else:
                D_p, H_p, W_p = D, H, W

            tta_str = f" (TTA x{len(_MIRROR_AXES_3D)})" if self.use_mirroring else ""
            if use_gpu:
                input_gpu = input_image.cuda().contiguous()
                _logger.info("3D ORT: GPU IOBinding%s", tta_str)
                out_shape = [1, num_seg_heads] + list(patch_size)

                for sx in steps[0]:
                    for sy in steps[1]:
                        for sz in steps[2]:
                            patch = input_gpu[
                                :, sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]
                            ].unsqueeze(0).contiguous()

                            prediction = self._predict_patch_gpu(patch, out_shape)
                            if self.use_gaussian:
                                prediction *= gaussian
                            predicted_logits[:, sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]] += prediction
                            n_predictions[sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]] += gaussian if isinstance(gaussian, torch.Tensor) else 1

                del input_gpu
                torch.cuda.empty_cache()
            else:
                _logger.info("3D ORT: CPU path%s", tta_str)
                input_np = input_image.numpy()
                for sx in steps[0]:
                    for sy in steps[1]:
                        for sz in steps[2]:
                            patch = input_np[np.newaxis, :, sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]].astype(self._np_dtype)
                            prediction = self._predict_patch_cpu(patch)
                            if self.use_gaussian:
                                prediction *= gaussian
                            predicted_logits[:, sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]] += prediction
                            n_predictions[sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]] += gaussian if isinstance(gaussian, torch.Tensor) else 1

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                predicted_logits = predicted_logits[:, :D, :H, :W]
            return predicted_logits


    class ORTPredictor25D(nnUNetPredictor25D):
        """2.5D nnUNet predictor using ONNX Runtime for the forward pass."""

        def __init__(self, ort_session, **kwargs):
            super().__init__(**kwargs)
            self._ort_session = ort_session
            self._input_name = ort_session.get_inputs()[0].name
            self._output_name = ort_session.get_outputs()[0].name
            inp_type = ort_session.get_inputs()[0].type
            self._use_fp16 = 'float16' in inp_type
            self._np_dtype = np.float16 if self._use_fp16 else np.float32
            self._torch_dtype = torch.float16 if self._use_fp16 else torch.float32

        def _use_gpu_iobinding(self):
            if not _HAS_TORCH or not torch.cuda.is_available():
                return False
            try:
                active = self._ort_session.get_providers()
                return any('CUDA' in p or 'Tensorrt' in p for p in active)
            except Exception:
                return False

        def _run_patch_gpu(self, patch_gpu, out_shape):
            if self._use_fp16:
                patch_gpu = patch_gpu.half()
            out_gpu = torch.empty(out_shape, dtype=self._torch_dtype, device='cuda')
            io_binding = self._ort_session.io_binding()
            io_binding.bind_input(self._input_name, 'cuda', 0, self._np_dtype, list(patch_gpu.shape), patch_gpu.data_ptr())
            io_binding.bind_output(self._output_name, 'cuda', 0, self._np_dtype, out_shape, out_gpu.data_ptr())
            self._ort_session.run_with_iobinding(io_binding)
            return out_gpu

        def _predict_patch_gpu(self, patch_gpu, out_shape):
            if not self.use_mirroring:
                return self._run_patch_gpu(patch_gpu, out_shape)[0].cpu()
            # Keep batch dim for correct flip axes; skip NaN mirrors
            result = self._run_patch_gpu(patch_gpu, out_shape).float()
            count = 1
            for axes in _MIRROR_AXES_2D[1:]:
                flipped = torch.flip(patch_gpu, list(axes)).contiguous()
                out = self._run_patch_gpu(flipped, out_shape).float()
                out_unflipped = torch.flip(out, list(axes))
                if not torch.isnan(out_unflipped).any():
                    result += out_unflipped
                    count += 1
            result /= count
            return result[0].cpu()

        def _predict_patch_cpu(self, patch_np):
            patch_np = patch_np.astype(self._np_dtype)
            logits = self._ort_session.run([self._output_name], {self._input_name: patch_np})[0]
            if not self.use_mirroring:
                return torch.from_numpy(logits[0].astype(np.float32))
            # Keep batch dim for correct flip axes; skip NaN mirrors
            result = torch.from_numpy(logits.astype(np.float32))
            count = 1
            patch_t = torch.from_numpy(patch_np.astype(np.float32))
            for axes in _MIRROR_AXES_2D[1:]:
                flipped = torch.flip(patch_t, list(axes)).contiguous().numpy().astype(self._np_dtype)
                out = self._ort_session.run([self._output_name], {self._input_name: flipped})[0]
                out_t = torch.flip(torch.from_numpy(out.astype(np.float32)), list(axes))
                if not torch.isnan(out_t).any():
                    result += out_t
                    count += 1
            result /= count
            return result[0]

        @torch.inference_mode()
        def predict_sliding_window_return_logits(self, input_image):
            from nnunetv2.inference.sliding_window_prediction import (
                compute_steps_for_sliding_window, compute_gaussian)

            assert isinstance(input_image, torch.Tensor)
            assert input_image.ndim == 4
            use_gpu = self._use_gpu_iobinding()

            if self._perm is not None:
                input_image = input_image.permute(self._perm).contiguous()

            C, D, H, W = input_image.shape
            half_k = self.half_k

            patch_size_2d = self.configuration_manager_2d.patch_size
            if len(patch_size_2d) == 3:
                patch_size_2d = patch_size_2d[1:]

            steps = compute_steps_for_sliding_window((H, W), patch_size_2d, self.tile_step_size)

            results_device = torch.device('cpu')
            gaussian = compute_gaussian(
                tuple(patch_size_2d), sigma_scale=1./8, value_scaling_factor=10,
                device=results_device, dtype=torch.float32
            ).float() if self.use_gaussian else 1

            num_seg_heads = self.label_manager.num_segmentation_heads
            predicted_logits = torch.zeros((num_seg_heads, D, H, W), dtype=torch.float32, device=results_device)
            n_predictions = torch.zeros((D, H, W), dtype=torch.float32, device=results_device)

            pad_h = max(0, patch_size_2d[0] - H)
            pad_w = max(0, patch_size_2d[1] - W)
            if pad_h > 0 or pad_w > 0:
                input_image = F.pad(input_image, (0, pad_w, 0, pad_h), mode='constant', value=0)
                _, _, H_p, W_p = input_image.shape
                steps = compute_steps_for_sliding_window((H_p, W_p), patch_size_2d, self.tile_step_size)
                predicted_logits_padded = torch.zeros((num_seg_heads, D, H_p, W_p), dtype=torch.float32, device=results_device)
                n_predictions_padded = torch.zeros((D, H_p, W_p), dtype=torch.float32, device=results_device)
            else:
                H_p, W_p = H, W
                predicted_logits_padded = predicted_logits
                n_predictions_padded = n_predictions

            tta_str = f" (TTA x{len(_MIRROR_AXES_2D)})" if self.use_mirroring else ""

            if use_gpu:
                input_gpu = input_image.cuda().contiguous()
                _logger.info("2.5D ORT: GPU IOBinding%s", tta_str)
                zeros_gpu = torch.zeros((H_p, W_p), dtype=torch.float32, device='cuda')
                out_shape = [1, num_seg_heads, patch_size_2d[0], patch_size_2d[1]]

                for z in range(D):
                    context_slices = []
                    for c in range(C):
                        for dz in range(-half_k, half_k + 1):
                            zz = z + dz
                            context_slices.append(input_gpu[c, zz] if 0 <= zz < D else zeros_gpu)
                    context = torch.stack(context_slices, dim=0)

                    for sx in steps[0]:
                        for sy in steps[1]:
                            patch = context[:, sx:sx+patch_size_2d[0], sy:sy+patch_size_2d[1]].unsqueeze(0).contiguous()
                            prediction = self._predict_patch_gpu(patch, out_shape)
                            if self.use_gaussian:
                                prediction *= gaussian
                            predicted_logits_padded[:, z, sx:sx+patch_size_2d[0], sy:sy+patch_size_2d[1]] += prediction
                            n_predictions_padded[z, sx:sx+patch_size_2d[0], sy:sy+patch_size_2d[1]] += gaussian if isinstance(gaussian, torch.Tensor) else 1

                del input_gpu, zeros_gpu
                torch.cuda.empty_cache()
            else:
                _logger.info("2.5D ORT: CPU path%s", tta_str)
                input_np = input_image.numpy()
                for z in range(D):
                    slices = []
                    for c in range(C):
                        for dz in range(-half_k, half_k + 1):
                            zz = z + dz
                            slices.append(input_np[c, zz] if 0 <= zz < D else np.zeros((H_p, W_p), dtype=np.float32))
                    context = np.stack(slices, axis=0)

                    for sx in steps[0]:
                        for sy in steps[1]:
                            patch = context[np.newaxis, :, sx:sx+patch_size_2d[0], sy:sy+patch_size_2d[1]].astype(self._np_dtype)
                            prediction = self._predict_patch_cpu(patch)
                            if self.use_gaussian:
                                prediction *= gaussian
                            predicted_logits_padded[:, z, sx:sx+patch_size_2d[0], sy:sy+patch_size_2d[1]] += prediction
                            n_predictions_padded[z, sx:sx+patch_size_2d[0], sy:sy+patch_size_2d[1]] += gaussian if isinstance(gaussian, torch.Tensor) else 1

            torch.div(predicted_logits_padded, n_predictions_padded, out=predicted_logits_padded)
            if pad_h > 0 or pad_w > 0:
                predicted_logits = predicted_logits_padded[:, :, :H, :W]
            else:
                predicted_logits = predicted_logits_padded
            if self._perm is not None:
                predicted_logits = predicted_logits.permute(self._perm).contiguous()
            return predicted_logits


def create_ort_predictor(arch_key, model_dir, folds=(0,), device_str="onnx-cuda", use_mirroring=True):
    """Create an ORT-backed predictor for the given architecture."""
    if not _HAS_ORT:
        raise RuntimeError("onnxruntime not installed")
    if not _HAS_NNUNET:
        raise RuntimeError("nnunetv2 required for preprocessing")

    pred_type = ARCHITECTURES[arch_key][1]
    fold = folds[0]
    onnx_path = os.path.join(model_dir, f'fold_{fold}', 'model.onnx')

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # Select providers and options based on device string
    if device_str == "onnx-trt":
        trt_cache_dir = os.path.join(model_dir, 'trt_cache')
        os.makedirs(trt_cache_dir, exist_ok=True)
        providers = [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': str(2 * 1024 * 1024 * 1024),  # 2GB
                'trt_fp16_enable': 'True',
                'trt_engine_cache_enable': 'True',
                'trt_engine_cache_path': trt_cache_dir,
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif device_str == "onnx-cuda":
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif device_str == "onnx-cpu":
        providers = ['CPUExecutionProvider']
    else:
        providers = get_ort_providers()
    _logger.info("Creating ORT session: %s (providers: %s)", onnx_path, providers)

    # For GPU modes, use fp16 model to halve VRAM (matches PyTorch .half())
    use_fp16 = device_str in ("onnx-cuda", "onnx-trt")
    if use_fp16:
        fp16_path = os.path.join(model_dir, f'fold_{fold}', 'model_fp16.onnx')
        if not os.path.isfile(fp16_path):
            _logger.info("Converting ONNX model to fp16: %s", fp16_path)
            import onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16
            model_proto = onnx.load(onnx_path)
            model_fp16 = convert_float_to_float16(
                model_proto, keep_io_types=False,
                force_fp16_initializers=True)
            onnx.save(model_fp16, fp16_path)
            del model_proto, model_fp16
            _logger.info("Saved fp16 model: %s", fp16_path)
        onnx_path = fp16_path

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)

    plans = load_json(os.path.join(model_dir, 'plans.json'))
    dataset_json = load_json(os.path.join(model_dir, 'dataset.json'))
    plans_manager = PlansManager(plans)

    if pred_type == "25d":
        predictor = ORTPredictor25D(
            ort_session=session,
            num_adjacent_slices=7, slice_axis=0,
            tile_step_size=0.5, use_gaussian=True, use_mirroring=use_mirroring,
            perform_everything_on_device=False,
            device=torch.device('cpu'), verbose=False, verbose_preprocessing=False,
            allow_tqdm=False,
        )
        # Initialize plans/config but skip loading a PyTorch network
        configuration_manager_2d = plans_manager.get_configuration('2d')
        num_input_channels = len(dataset_json['channel_names']) * 7

        predictor.plans_manager = plans_manager
        predictor.configuration_manager = plans_manager.get_configuration('3d_fullres')
        predictor.configuration_manager_2d = configuration_manager_2d
        predictor.list_of_parameters = [None]
        predictor.network = _OnnxNetworkProxy()
        predictor.dataset_json = dataset_json
        predictor.trainer_name = "nnUNetTrainer_25D"
        predictor.allowed_mirroring_axes = None
        predictor.label_manager = plans_manager.get_label_manager(dataset_json)
    else:
        predictor = ORTPredictor3D(
            ort_session=session,
            tile_step_size=0.5, use_gaussian=True, use_mirroring=use_mirroring,
            perform_everything_on_device=False,
            device=torch.device('cpu'), verbose=False, verbose_preprocessing=False,
            allow_tqdm=False,
        )
        configuration_manager = plans_manager.get_configuration('3d_fullres')

        predictor.plans_manager = plans_manager
        predictor.configuration_manager = configuration_manager
        predictor.list_of_parameters = [None]
        predictor.network = _OnnxNetworkProxy()
        predictor.dataset_json = dataset_json
        predictor.trainer_name = "nnUNetTrainer"
        predictor.allowed_mirroring_axes = None
        predictor.label_manager = plans_manager.get_label_manager(dataset_json)

    return predictor, "onnx"
