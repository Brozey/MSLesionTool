#!/usr/bin/env python3
"""
Download MSLesionTool model weights from HuggingFace Hub.

Usage:
    python download_models.py                   # Default: Best-2 PyTorch checkpoints
    python download_models.py --onnx            # Also download ONNX models
    python download_models.py --all-folds       # All 5 folds (not just folds 1,3)
    python download_models.py --onnx-only       # Only ONNX models (smaller download)
"""
import argparse
import os
import sys

REPO_ID = "Broozey/MSLesionTool"

ARCHS = ["cnn3d", "resencl3d", "conv25d"]
BEST2_FOLDS = [1, 3]
ALL_FOLDS = [0, 1, 2, 3, 4]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def download():
    parser = argparse.ArgumentParser(description="Download MSLesionTool model weights")
    parser.add_argument("--onnx", action="store_true", help="Also download ONNX models")
    parser.add_argument("--onnx-only", action="store_true", help="Only download ONNX models")
    parser.add_argument("--all-folds", action="store_true", help="Download all 5 folds")
    parser.add_argument("--repo", default=REPO_ID, help="HuggingFace repo ID")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    folds = ALL_FOLDS if args.all_folds else BEST2_FOLDS
    download_pth = not args.onnx_only
    download_onnx = args.onnx or args.onnx_only

    # Build file list
    files = []
    for arch in ARCHS:
        for fold in folds:
            if download_pth:
                files.append(f"{arch}/fold_{fold}/checkpoint_best.pth")
            if download_onnx:
                files.append(f"{arch}/fold_{fold}/model.onnx")

    total_count = len(files)
    print(f"Downloading {total_count} files from {args.repo}...")
    print(f"  Architectures: {', '.join(ARCHS)}")
    print(f"  Folds: {folds}")
    print(f"  PyTorch: {'yes' if download_pth else 'no'}")
    print(f"  ONNX: {'yes' if download_onnx else 'no'}")
    print()

    msseg_dir = os.path.join(SCRIPT_DIR, "msseg")

    for i, rel_path in enumerate(files, 1):
        local_path = os.path.join(msseg_dir, rel_path)
        if os.path.isfile(local_path):
            print(f"  [{i}/{total_count}] {rel_path} (already exists, skipping)")
            continue

        print(f"  [{i}/{total_count}] {rel_path}...", end=" ", flush=True)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        hf_hub_download(
            repo_id=args.repo,
            filename=rel_path,
            local_dir=msseg_dir,
            local_dir_use_symlinks=False,
        )
        print("OK")

    print(f"\nDone! Models saved to {msseg_dir}")


if __name__ == "__main__":
    download()
