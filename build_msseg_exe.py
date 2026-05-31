import subprocess
import sys
import os

APP_NAME = "MSLesionTool"
MAIN_SCRIPT = "msseg_app.py"

# Only exclude packages that are DEFINITELY not needed by torch/nnUNet/app
EXCLUDE_MODULES = [
    "PyQt5", "tensorflow", "keras", "tensorboard",
    "IPython", "tkinter", "torchvision", "torchaudio",
    "h5py", "grpc", "grpcio", "google.protobuf",
    "jedi", "Pythonwin", "hf_xet",
    "langchain", "langchain_core", "langchain_community",
    "langchain_text_splitters", "langsmith",
    "mlflow", "databricks",
    "bitsandbytes",
    "wandb",
    "cupy", "cupy_backends",
    "triton",
    "flask",
    "uvicorn", "starlette", "fastapi", "httpx",
    "boto3", "botocore", "s3transfer",
    "google.cloud", "google.auth", "google.api_core",
    "azure",
    "openai",
]


def build():
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", APP_NAME,
        "--windowed",
        "--onedir",
    ]

    for mod in EXCLUDE_MODULES:
        cmd.extend(["--exclude-module", mod])

    cmd.extend([
        "--hidden-import", "numpy",
        "--hidden-import", "torch",
        "--hidden-import", "nnunetv2",
        "--hidden-import", "imageio",
        "--hidden-import", "SimpleITK",
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.ndimage",
        "--hidden-import", "skimage",
        "--hidden-import", "matplotlib",
        "--hidden-import", "pyqtgraph",
        "--hidden-import", "dynamic_network_architectures",
        "--hidden-import", "nnunetv2.training.nnUNetTrainer.nnUNetTrainer_WandB",
        "--collect-data", "nnunetv2",
        "--collect-binaries", "PyQt6",
    ])

    # Model weights + configs are downloaded on first launch from HuggingFace
    # Do NOT bundle msseg/cnn3d etc. — empty dirs confuse model resolution

    cmd.append(MAIN_SCRIPT)

    print("Running PyInstaller with:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode == 0:
        _cleanup_stale_msvc_dlls(APP_NAME)
        print()
        print(f"Build successful!  Look in:  dist/{APP_NAME}/")
        print(f"Run:  dist/{APP_NAME}/{APP_NAME}.exe")
    else:
        print("Build FAILED.")
        sys.exit(1)


def _cleanup_stale_msvc_dlls(app_name):
    """Remove old MSVCP140*.dll from PyQt6's Qt6/bin directory."""
    qt_bin = os.path.join("dist", app_name, "_internal", "PyQt6", "Qt6", "bin")
    if not os.path.isdir(qt_bin):
        return
    for fn in os.listdir(qt_bin):
        if fn.lower().startswith("msvcp140") and fn.lower().endswith(".dll"):
            path = os.path.join(qt_bin, fn)
            try:
                os.remove(path)
                print(f"  Removed stale DLL: {path}")
            except Exception as e:
                print(f"  Failed to remove {path}: {e}")


if __name__ == "__main__":
    build()
