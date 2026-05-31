#!/usr/bin/env python3
"""
Export nnUNet models to ONNX format for ONNX Runtime inference.

Usage:
    python scripts/export_onnx.py                          # Export default Best-2/arch (6 models)
    python scripts/export_onnx.py --all                    # Export all 15 models
    python scripts/export_onnx.py --arch cnn3d --folds 1 3 # Export specific arch+folds

Output: model.onnx alongside each checkpoint_best.pth in the fold directory.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import onnx


def build_network_from_trainer(model_dir, fold, config_name, num_input_channels=None):
    """Use nnUNet's own machinery to reconstruct the network (handles version compat)."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.file_path_utilities import load_json

    plans = load_json(os.path.join(model_dir, 'plans.json'))
    dataset_json = load_json(os.path.join(model_dir, 'dataset.json'))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(config_name)

    checkpoint_path = os.path.join(model_dir, f'fold_{fold}', 'checkpoint_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    trainer_name = checkpoint['trainer_name']

    if num_input_channels is None:
        num_input_channels = len(dataset_json['channel_names'])

    num_seg_heads = plans_manager.get_label_manager(dataset_json).num_segmentation_heads

    # Try using the trainer class to build the network (handles version differences)
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    trainer_class = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name, 'nnunetv2.training.nnUNetTrainer')

    if trainer_class is not None:
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_seg_heads,
            enable_deep_supervision=False)
    else:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_seg_heads,
            deep_supervision=False)

    network.load_state_dict(checkpoint['network_weights'])
    network.eval()
    return network, configuration_manager.patch_size


def export_model(model_dir, fold, config_name, num_input_channels=None, opset=17):
    """Export a single nnUNet model (one fold) to ONNX."""
    checkpoint_path = os.path.join(model_dir, f'fold_{fold}', 'checkpoint_best.pth')
    output_path = os.path.join(model_dir, f'fold_{fold}', 'model.onnx')

    if not os.path.exists(checkpoint_path):
        print(f"  SKIP: {checkpoint_path} not found")
        return None

    print(f"  Loading checkpoint: fold_{fold}/checkpoint_best.pth")
    network, patch_size = build_network_from_trainer(model_dir, fold, config_name, num_input_channels)

    # Determine input shape
    if num_input_channels is None:
        dataset_json = json.load(open(os.path.join(model_dir, 'dataset.json')))
        num_input_channels = len(dataset_json['channel_names'])

    dummy = torch.randn(1, num_input_channels, *patch_size)

    print(f"  Input shape: {list(dummy.shape)}")
    print(f"  Exporting to ONNX (opset {opset})...")

    torch.onnx.export(
        network,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
    )

    # Validate
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  OK: {output_path} ({size_mb:.1f} MB)")
    return output_path


# Architecture configurations
ARCH_CONFIGS = {
    "cnn3d": {
        "config_name": "3d_fullres",
        "num_input_channels": 2,  # FLAIR + T1
    },
    "resencl3d": {
        "config_name": "3d_fullres",
        "num_input_channels": 2,
    },
    "conv25d": {
        "config_name": "2d",
        "num_input_channels": 14,  # 2 channels * 7 context slices
    },
}

DEFAULT_BEST2 = {"cnn3d": [1, 3], "resencl3d": [1, 3], "conv25d": [1, 3]}


def main():
    parser = argparse.ArgumentParser(description="Export nnUNet models to ONNX")
    parser.add_argument("--model-root", default=None,
                        help="Root directory containing msseg/{cnn3d,resencl3d,conv25d}/")
    parser.add_argument("--arch", choices=list(ARCH_CONFIGS.keys()), nargs="+",
                        help="Architectures to export (default: all 3)")
    parser.add_argument("--folds", type=int, nargs="+",
                        help="Folds to export (default: Best-2 = 1, 3)")
    parser.add_argument("--all", action="store_true",
                        help="Export all 5 folds for all architectures")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    # Find model root
    if args.model_root:
        root = args.model_root
    else:
        # Auto-detect: look in MSLesionTool_portable/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, '..', 'MSLesionTool_portable'),
            os.path.join(script_dir, '..'),
            script_dir,
        ]
        root = None
        for c in candidates:
            if os.path.isdir(os.path.join(c, 'msseg', 'cnn3d')):
                root = os.path.abspath(c)
                break
        if root is None:
            print("ERROR: Could not find msseg/ model directory. Use --model-root.")
            sys.exit(1)

    archs = args.arch or list(ARCH_CONFIGS.keys())

    exported = []
    for arch in archs:
        cfg = ARCH_CONFIGS[arch]
        model_dir = os.path.join(root, 'msseg', arch)
        if not os.path.isdir(model_dir):
            print(f"WARNING: {model_dir} not found, skipping {arch}")
            continue

        if args.all:
            folds = list(range(5))
        elif args.folds:
            folds = args.folds
        else:
            folds = DEFAULT_BEST2.get(arch, [1, 3])

        print(f"\n{'='*60}")
        print(f"Architecture: {arch} | Config: {cfg['config_name']} | Folds: {folds}")
        print(f"{'='*60}")

        for fold in folds:
            result = export_model(
                model_dir, fold,
                config_name=cfg['config_name'],
                num_input_channels=cfg['num_input_channels'],
                opset=args.opset,
            )
            if result:
                exported.append(result)

    print(f"\n{'='*60}")
    print(f"Exported {len(exported)} ONNX models successfully.")
    total_mb = sum(os.path.getsize(p) / 1024 / 1024 for p in exported)
    print(f"Total size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
