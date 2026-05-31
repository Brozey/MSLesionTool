"""
msseg.constants – Architecture registry, labels, and configuration.
"""
import os, sys, logging

_logger = logging.getLogger("MSLesionTool")

_APP_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Architecture registry: key -> (subdir, predictor_type, display_name)
ARCHITECTURES = {
    "cnn3d":     ("msseg/cnn3d",     "3d",  "CNN 3D"),
    "resencl3d": ("msseg/resencl3d", "3d",  "ResEncL 3D"),
    "conv25d":   ("msseg/conv25d",   "25d", "2.5D (K=7)"),
}

# Best 2 folds per architecture (validation-selected, EMA fg_dice)
DEFAULT_BEST2 = {"cnn3d": (1, 3), "resencl3d": (1, 3), "conv25d": (1, 3)}

# HuggingFace repo for model weights
HF_REPO_ID = "Broozey/MSLesionTool"

# User-local model cache directory (%APPDATA%/MSLesionTool on Windows, ~/.mslesion on others)
if sys.platform == "win32":
    _USER_MODEL_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "MSLesionTool", "models")
else:
    _USER_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".mslesion", "models")

# Search: app bundle first, then user-local cache
_MODEL_SEARCH_DIRS = [_APP_DIR, _USER_MODEL_DIR]

# nnUNet output labels
SEG_LABELS = {
    0: ("Background", (0, 0, 0)),
    1: ("Lesion",     (255, 80, 80)),
}

# Classification labels for MS lesion types
CLASS_LABELS = {
    0: ("Unclassified",           (255, 255, 255)),
    1: ("Periventricular",        (255, 165, 0)),
    2: ("Juxtacortical/Cortical", (0, 220, 220)),
    3: ("Infratentorial",         (220, 0, 220)),
    4: ("Spinal Cord",            (255, 255, 0)),
    5: ("CVS+",                   (0, 255, 128)),
    6: ("Delete",                 (255, 50, 50)),
}

VIEWS = ["Axial", "Sagittal", "Coronal"]


def resolve_model_dir(subdir):
    """Find a model subdirectory in the search paths."""
    for base in _MODEL_SEARCH_DIRS:
        candidate = os.path.join(base, subdir)
        if os.path.isdir(candidate):
            return candidate
    return None


def models_missing():
    """Return list of (arch_key, fold) tuples whose checkpoint is not on disk."""
    missing = []
    for arch_key, (subdir, _ptype, _display) in ARCHITECTURES.items():
        for fold in DEFAULT_BEST2[arch_key]:
            d = resolve_model_dir(subdir)
            if d is None or not os.path.isfile(os.path.join(d, f"fold_{fold}", "checkpoint_best.pth")):
                missing.append((arch_key, fold))
    return missing


def get_user_model_dir():
    """Return the user-local model cache directory."""
    return _USER_MODEL_DIR
