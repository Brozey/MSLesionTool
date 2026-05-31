#!/usr/bin/env python3
"""
Upload MSLesionTool model weights to HuggingFace Hub.

Run this once to push your trained weights. Requires:
    pip install huggingface_hub
    huggingface-cli login

Usage:
    python upload_models.py                     # Upload Best-2 PyTorch + ONNX
    python upload_models.py --all-folds         # Upload all 5 folds
"""
import argparse
import os
import sys

REPO_ID = "Broozey/MSLesionTool"

ARCHS = ["cnn3d", "resencl3d", "conv25d"]
BEST2_FOLDS = [1, 3]
ALL_FOLDS = [0, 1, 2, 3, 4]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def upload():
    parser = argparse.ArgumentParser(description="Upload MSLesionTool weights to HuggingFace")
    parser.add_argument("--all-folds", action="store_true", help="Upload all 5 folds")
    parser.add_argument("--repo", default=REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--create-repo", action="store_true", help="Create the repo if it doesn't exist")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    if args.create_repo:
        create_repo(args.repo, repo_type="model", exist_ok=True)
        print(f"Repo {args.repo} ready.")

    folds = ALL_FOLDS if args.all_folds else BEST2_FOLDS
    msseg_dir = os.path.join(SCRIPT_DIR, "msseg")

    # Collect files to upload
    files = []
    for arch in ARCHS:
        for fold in folds:
            for fname in ["checkpoint_best.pth", "model.onnx"]:
                local = os.path.join(msseg_dir, arch, f"fold_{fold}", fname)
                remote = f"{arch}/fold_{fold}/{fname}"
                if os.path.isfile(local):
                    files.append((local, remote))

    print(f"Uploading {len(files)} files to {args.repo}...")
    for i, (local, remote) in enumerate(files, 1):
        size_mb = os.path.getsize(local) / 1048576
        print(f"  [{i}/{len(files)}] {remote} ({size_mb:.0f} MB)...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="model",
        )
        print("OK")

    print(f"\nDone! View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    upload()
