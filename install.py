#!/usr/bin/env python3
"""
MSLesionTool installer — detects CUDA, installs PyTorch from the correct
wheel index, then installs all remaining dependencies.

Usage:
    python install.py              # Auto-detect everything
    python install.py --cpu        # Force CPU-only PyTorch
    python install.py --cuda 12.8  # Force specific CUDA version
    python install.py --cli-only   # Minimal CLI/inference dependencies
    python install.py --verify-only  # Only run verification
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered from highest to lowest — first match wins
CUDA_INDEX_MAP = [
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]

CUSTOM_TRAINERS = [
    "nnUNetTrainer_WandB.py",
    "nnUNetTrainer_25D.py",
]


def detect_cuda_version():
    """Detect the CUDA version supported by the installed NVIDIA driver."""
    for cmd, pattern in [
        (["nvidia-smi"], r"CUDA Version:\s+(\d+)\.(\d+)"),
        (["nvcc", "--version"], r"release (\d+)\.(\d+)"),
    ]:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
            m = re.search(pattern, out)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def select_torch_index(cuda_version):
    """Return (index_url, tag) for the best matching PyTorch CUDA wheel."""
    if cuda_version is None:
        return "https://download.pytorch.org/whl/cpu", "cpu"
    for min_ver, tag in CUDA_INDEX_MAP:
        if cuda_version >= min_ver:
            return f"https://download.pytorch.org/whl/{tag}", tag
    return "https://download.pytorch.org/whl/cpu", "cpu"


def pip_run(*args):
    """Run pip as a subprocess."""
    cmd = [sys.executable, "-m", "pip", "install"] + list(args)
    subprocess.check_call(cmd)


def get_installed_version(package):
    """Get the installed version of a package."""
    out = subprocess.check_output(
        [sys.executable, "-m", "pip", "show", package],
        text=True, stderr=subprocess.DEVNULL,
    )
    for line in out.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def install_torch(index_url):
    """Install PyTorch and torchvision from the specified wheel index."""
    pip_run("torch", "torchvision", "--index-url", index_url)


def install_base_deps(cli_only=False):
    """Install non-torch, non-nnunet dependencies."""
    req = "requirements-inference.txt" if cli_only else "requirements-base.txt"
    req_path = os.path.join(SCRIPT_DIR, req)
    if not os.path.exists(req_path):
        print(f"  WARNING: {req} not found, skipping base deps")
        return
    pip_run("-r", req_path)


def install_nnunet(torch_version):
    """Install nnunetv2 with a constraint that pins the current torch version."""
    fd, constraint_path = tempfile.mkstemp(suffix=".txt", prefix="torch_pin_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"torch=={torch_version}\n")
        pip_run("nnunetv2>=2.5", "-c", constraint_path)
    finally:
        os.unlink(constraint_path)


def copy_trainers():
    """Copy custom nnUNet trainers into the installed nnunetv2 package."""
    try:
        import nnunetv2
        dst_dir = os.path.join(
            os.path.dirname(nnunetv2.__file__),
            "training", "nnUNetTrainer",
        )
    except ImportError:
        print("  WARNING: nnunetv2 not installed, skipping trainer copy")
        return

    src_dir = os.path.join(SCRIPT_DIR, "trainers")
    if not os.path.isdir(src_dir):
        print(f"  WARNING: trainers/ directory not found at {src_dir}")
        return

    for fn in CUSTOM_TRAINERS:
        src = os.path.join(src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  {fn} -> {dst_dir}")
        else:
            print(f"  WARNING: {fn} not found in trainers/")


def install_onnxruntime(cuda_version):
    """Install onnxruntime (GPU or CPU variant)."""
    pkg = "onnxruntime-gpu>=1.16" if cuda_version else "onnxruntime>=1.16"
    try:
        pip_run(pkg)
    except subprocess.CalledProcessError:
        print(f"  WARNING: {pkg} install failed, ONNX backend will be unavailable")


def verify():
    """Print installation diagnostics."""
    print("\n" + "=" * 50)
    print("  Installation Verification")
    print("=" * 50)

    # Torch
    try:
        import torch
        print(f"  PyTorch:   {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA:      {torch.version.cuda}")
            print(f"  GPU:       {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability()
            arch_list = torch.cuda.get_arch_list()
            supported = f"sm_{cap[0]}{cap[1]}" in arch_list
            print(f"  Arch:      sm_{cap[0]}{cap[1]} ({'supported' if supported else 'NOT supported — will use CPU'})")
        else:
            print("  CUDA:      not available (CPU mode)")
    except ImportError:
        print("  PyTorch:   NOT INSTALLED")

    # nnUNet
    try:
        import nnunetv2
        ver = getattr(nnunetv2, "__version__", "installed")
        print(f"  nnUNet:    {ver}")
        trainer_dir = os.path.join(
            os.path.dirname(nnunetv2.__file__),
            "training", "nnUNetTrainer",
        )
        for fn in CUSTOM_TRAINERS:
            status = "OK" if os.path.isfile(os.path.join(trainer_dir, fn)) else "MISSING"
            print(f"    {fn}: {status}")
    except ImportError:
        print("  nnUNet:    NOT INSTALLED")

    # Key deps
    for name in ["PyQt6", "pyqtgraph", "SimpleITK", "nibabel", "scipy", "kornia"]:
        try:
            __import__(name)
            print(f"  {name:12s} OK")
        except ImportError:
            print(f"  {name:12s} missing")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        print(f"  ORT:       {ort.__version__} ({', '.join(provs)})")
    except ImportError:
        print("  ORT:       not installed (optional)")

    print()


def main():
    parser = argparse.ArgumentParser(description="MSLesionTool installer")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only PyTorch")
    parser.add_argument("--cuda", type=str, default=None, help="Force CUDA version (e.g. 12.8)")
    parser.add_argument("--cli-only", action="store_true", help="Minimal CLI dependencies (no GUI)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification")
    parser.add_argument("--skip-trainers", action="store_true", help="Skip custom trainer copy")
    args = parser.parse_args()

    if args.verify_only:
        verify()
        return

    # Python version check
    if sys.version_info < (3, 9):
        print(f"ERROR: Python 3.9+ required (you have {sys.version})")
        sys.exit(1)

    # Detect environment
    in_venv = sys.prefix != sys.base_prefix
    in_conda = "CONDA_PREFIX" in os.environ
    if not in_venv and not in_conda:
        print("WARNING: Not in a virtual environment.")
        print("         Recommend: python -m venv .venv && .venv\\Scripts\\activate")
        if not args.yes:
            ans = input("Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                sys.exit(0)

    # Determine CUDA
    if args.cpu:
        cuda_ver = None
    elif args.cuda:
        parts = args.cuda.split(".")
        cuda_ver = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    elif sys.platform == "darwin":
        cuda_ver = None
        print("macOS detected: installing CPU-only PyTorch (MPS supported natively)")
    else:
        cuda_ver = detect_cuda_version()

    index_url, cuda_tag = select_torch_index(cuda_ver)

    # Summary
    print()
    print("  MSLesionTool Installer")
    print("  " + "-" * 40)
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  CUDA:       {f'{cuda_ver[0]}.{cuda_ver[1]}' if cuda_ver else 'None'}")
    print(f"  PyTorch:    {cuda_tag}")
    print(f"  Mode:       {'CLI only' if args.cli_only else 'Full GUI'}")
    print()

    if not args.yes:
        ans = input("Proceed? [Y/n] ").strip().lower()
        if ans == "n":
            sys.exit(0)

    # Step 1: PyTorch
    print("\n[1/5] Installing PyTorch...")
    install_torch(index_url)

    torch_ver = get_installed_version("torch")
    print(f"  Installed torch {torch_ver}")

    # Step 2: Base dependencies
    print("\n[2/5] Installing dependencies...")
    install_base_deps(cli_only=args.cli_only)

    # Step 3: nnunetv2
    print("\n[3/5] Installing nnunetv2...")
    install_nnunet(torch_ver)

    # Step 4: Custom trainers
    if not args.skip_trainers:
        print("\n[4/5] Installing custom trainers...")
        copy_trainers()
    else:
        print("\n[4/5] Skipping custom trainers")

    # Step 5: ONNX Runtime (optional)
    print("\n[5/5] Installing ONNX Runtime...")
    install_onnxruntime(cuda_ver)

    # Verify
    verify()

    print("=" * 50)
    if args.cli_only:
        print("  Ready! Run:")
        print("  python -m msseg.cli --flair f.nii.gz --t1 t.nii.gz -o seg.nii.gz")
    else:
        print("  Ready! Run:")
        print("  python msseg_app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
