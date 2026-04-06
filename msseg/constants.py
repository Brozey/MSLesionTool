"""
msseg.constants – Architecture registry, labels, and configuration.
"""
import os, sys

_APP_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Architecture registry: key -> (subdir, predictor_type, display_name)
ARCHITECTURES = {
    "cnn3d":     ("msseg/cnn3d",     "3d",  "CNN 3D"),
    "resencl3d": ("msseg/resencl3d", "3d",  "ResEncL 3D"),
    "conv25d":   ("msseg/conv25d",   "25d", "2.5D (K=7)"),
}

# Best 2 folds per architecture (validation-selected, EMA fg_dice)
DEFAULT_BEST2 = {"cnn3d": (1, 3), "resencl3d": (1, 3), "conv25d": (1, 3)}

_MODEL_SEARCH_DIRS = [_APP_DIR]

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
