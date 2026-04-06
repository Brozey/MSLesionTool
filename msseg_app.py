"""
MSLesionTool – MS Lesion Segmentation & Dataset Annotation Tool
================================================================
© Jindřich Brož.  All rights reserved.

3D nnUNet-based MS lesion segmentation from brain MRI (FLAIR, T1, T2),
multi-planar viewing, probability-based lesion growth, manual editing,
lesion classification, and dataset export.
"""

import sys, os, time, tempfile, logging, collections, json
import concurrent.futures
import numpy as np
import matplotlib.cm as cm

_APP_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("nnUNet_results", os.path.join(_APP_DIR, "nnUNet_results"))
os.environ.setdefault("nnUNet_raw", os.path.join(_APP_DIR, "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(_APP_DIR, "nnUNet_preprocessed"))

# Workaround: disable torch.compile to avoid torchvision ops dependency
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Fix Qt platform plugin path when PyQt6-WebEngine-Qt6 is installed
# (it ships a separate Qt6 base that moves plugins to Library/lib/qt6/plugins)
if not os.environ.get("QT_PLUGIN_PATH"):
    _conda_prefix = os.path.dirname(sys.executable)
    _qt_plugins = os.path.join(_conda_prefix, "Library", "lib", "qt6", "plugins")
    if os.path.isdir(_qt_plugins):
        os.environ["QT_PLUGIN_PATH"] = _qt_plugins

# Gracefully handle broken torchvision installation —
# purge it from sys.modules so nnUNet doesn't find a half-loaded module.
try:
    import torchvision  # noqa: test import
except Exception:
    _tv_keys = [k for k in sys.modules if k == "torchvision" or k.startswith("torchvision.")]
    for _k in _tv_keys:
        del sys.modules[_k]

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSlider, QGroupBox,
    QMessageBox, QSplitter, QSizePolicy, QStatusBar, QComboBox,
    QProgressBar, QSpinBox, QGridLayout, QCheckBox,
    QToolButton, QButtonGroup, QMenu,
    QDialog, QLineEdit, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QThread, QSettings, QTimer, QObject
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QFontMetricsF,
    QPalette, QIcon, QShortcut, QKeySequence, QBrush,
    QLinearGradient,
)

def _resource_path(relative):
    return os.path.join(_APP_DIR, relative)

# ── Show 3D splash screen IMMEDIATELY before heavy imports ────────
_splash_app = None  # prevent GC of QApplication before main()
_splash_3d = None
_splash_start_time = None

def _show_splash():
    global _splash_app, _splash_3d, _splash_start_time
    _splash_app = QApplication.instance() or QApplication(sys.argv)
    try:
        from msseg.splash3d import Splash3DWidget
        _splash_3d = Splash3DWidget()
        _splash_3d.show()
        _splash_3d.raise_()
        _splash_3d.activateWindow()
        _splash_3d.set_progress(5, "STARTING...")
        _splash_start_time = time.time()
    except Exception:
        pass
    _splash_app.processEvents()

def _update_splash(val, text=""):
    if _splash_3d:
        try:
            _splash_3d.set_progress(val, text)
        except Exception:
            pass

_show_splash()

# ── Heavy imports ────────────────────────────────────────────────────
_update_splash(10, "LOADING SIMPLEITK...")
try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    sitk = None
    _HAS_SITK = False

_update_splash(25, "LOADING NIBABEL...")
try:
    import nibabel as nib
    _HAS_NIB = True
except ImportError:
    nib = None
    _HAS_NIB = False

_update_splash(40, "LOADING PYTORCH...")
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    _HAS_TORCH = False

_update_splash(60, "LOADING NNUNET...")

try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    _HAS_NNUNET = True
except ImportError:
    nnUNetPredictor = None
    _HAS_NNUNET = False

_update_splash(75, "LOADING ONNX RUNTIME...")
try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    ort = None
    _HAS_ORT = False

_update_splash(80, "LOADING SCIPY...")
try:
    from scipy.ndimage import label as ndimage_label, binary_dilation, generate_binary_structure
    from scipy.ndimage import binary_fill_holes
    _HAS_SCIPY = True
except ImportError:
    ndimage_label = None
    binary_dilation = None
    generate_binary_structure = None
    binary_fill_holes = None
    _HAS_SCIPY = False

_update_splash(85, "LOADING 3D VIEWER...")
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    _HAS_PG = True
except ImportError:
    pg, gl = None, None
    _HAS_PG = False

_update_splash(90, "LOADING SCIKIT-IMAGE...")
try:
    from skimage.measure import marching_cubes
    _HAS_SKM = True
except ImportError:
    marching_cubes = None
    _HAS_SKM = False

# ── Logging ──────────────────────────────────────────────────────────
_update_splash(95, "BUILDING INTERFACE...")
_LOG_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
try:
    _LOG_FILE = os.path.join(_LOG_DIR, "MSLesionTool_log.txt")
    open(_LOG_FILE, "a").close()
except OSError:
    _LOG_FILE = os.path.join(tempfile.gettempdir(), "MSLesionTool_log.txt")
_logger = logging.getLogger("MSLesionTool")
_logger.setLevel(logging.INFO)
_fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger.addHandler(_fh)
_sh = logging.StreamHandler(sys.stderr)
_sh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
_logger.addHandler(_sh)

# ── Constants ────────────────────────────────────────────────────────
# Architecture registry: key -> (subdir, predictor_type, display_name)
ARCHITECTURES = {
    "cnn3d":     ("msseg/cnn3d",     "3d",  "CNN 3D"),
    "resencl3d": ("msseg/resencl3d", "3d",  "ResEncL 3D"),
    "conv25d":   ("msseg/conv25d",   "25d", "2.5D (K=7)"),
}
# Best 2 folds per architecture (validation-selected, EMA fg_dice)
DEFAULT_BEST2 = {"cnn3d": (1, 3), "resencl3d": (1, 3), "conv25d": (1, 3)}
_MODEL_SEARCH_DIRS = [_APP_DIR]


def _resolve_model_dir(subdir):
    """Find a model subdirectory in the search paths."""
    for base in _MODEL_SEARCH_DIRS:
        candidate = os.path.join(base, subdir)
        if os.path.isdir(candidate):
            return candidate
    return None

# nnUNet output labels (from dataset.json)
SEG_LABELS = {
    0: ("Background",    (0, 0, 0)),
    1: ("Lesion",  (255, 80, 80)),
}

# Classification labels for MS lesion types
CLASS_LABELS = {
    0: ("Unclassified",            (255, 255, 255)),
    1: ("Periventricular",         (255, 165, 0)),
    2: ("Juxtacortical/Cortical",  (0, 220, 220)),
    3: ("Infratentorial",          (220, 0, 220)),
    4: ("Spinal Cord",             (255, 255, 0)),
    5: ("CVS+",                    (0, 255, 128)),
    6: ("Delete",                  (255, 50, 50)),
}

VIEWS = ["Axial", "Sagittal", "Coronal"]

# ── Hardware detection ───────────────────────────────────────────────
def _cuda_arch_supported():
    """Check if CUDA actually works (forward compat handles newer GPUs)."""
    try:
        a = torch.ones(2, 2, device="cuda")
        _ = (a @ a).sum().item()
        del a
        return True
    except Exception:
        return False


def detect_best_device():
    if _HAS_TORCH and torch.cuda.is_available() and _cuda_arch_supported():
        try:
            a = torch.ones(2, 2, device="cuda")
            _ = (a @ a).sum().item()
            del a
            return "cuda"
        except Exception:
            pass
    return "cpu"


def detect_compute_resources():
    """Detect GPU VRAM (MB) and system RAM (MB). Returns dict."""
    info = {"gpu_name": None, "vram_total_mb": 0, "vram_free_mb": 0,
            "ram_total_mb": 0, "ram_free_mb": 0}
    # System RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_mb"] = mem.total // (1024 * 1024)
        info["ram_free_mb"] = mem.available // (1024 * 1024)
    except ImportError:
        pass
    # GPU VRAM via torch
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            info["vram_total_mb"] = total // (1024 * 1024)
            info["vram_free_mb"] = (total - torch.cuda.memory_reserved(0)) // (1024 * 1024)
        except Exception:
            pass
    # GPU VRAM via ORT (fallback if torch not available)
    if info["vram_total_mb"] == 0 and _HAS_ORT:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free,name",
                 "--format=csv,noheader,nounits"], text=True, timeout=5)
            parts = out.strip().split(", ")
            if len(parts) >= 3:
                info["vram_total_mb"] = int(parts[0])
                info["vram_free_mb"] = int(parts[1])
                info["gpu_name"] = parts[2]
        except Exception:
            pass
    return info


# Approximate peak VRAM per model during inference (MB) — empirical estimates
_MODEL_VRAM_MB = {
    "cnn3d": 800,      # ~119MB weights + activations
    "resencl3d": 2500,  # ~390MB weights + large activations
    "conv25d": 500,     # ~79MB weights + 2D patches
}


def plan_parallel_execution(ensemble_config, vram_mb, ram_mb):
    """Decide how many models to run in parallel based on resources.

    Returns max_workers (int) — number of concurrent model inferences.
    """
    if vram_mb <= 0:
        # CPU-only: use RAM. Each model needs ~2-4GB RAM for inference.
        if ram_mb > 16000:
            return 3
        elif ram_mb > 8000:
            return 2
        return 1

    # Estimate total VRAM needed for all models in the ensemble
    model_costs = []
    for arch_key, folds in ensemble_config.items():
        cost = _MODEL_VRAM_MB.get(arch_key, 1000)
        for _ in folds:
            model_costs.append((cost, arch_key))
    model_costs.sort(reverse=True)  # largest first

    # Greedy bin packing: how many fit in available VRAM?
    if not model_costs:
        return 1

    # Try fitting 2 models simultaneously
    if len(model_costs) >= 2:
        two_largest = model_costs[0][0] + model_costs[1][0]
        if two_largest < vram_mb * 0.85:  # 85% safety margin
            # Check if 3 fit
            if len(model_costs) >= 3:
                three = two_largest + model_costs[2][0]
                if three < vram_mb * 0.85:
                    return 3
            return 2

    return 1


# ── NIfTI loading ────────────────────────────────────────────────────
def load_nifti(path):
    """Load a NIfTI file, return (3D numpy array, affine, spacing, sitk_image).

    Reorients to LPS (DICOM standard) so array axes always map to
    (z=Superior→Inferior, y=Anterior→Posterior, x=Left→Right) regardless
    of the acquisition orientation.  spacing is returned as (sx, sy, sz).
    """
    if _HAS_SITK:
        img = sitk.ReadImage(path)
        # Squeeze 4-D single-frame volumes (e.g. shape [X,Y,Z,1])
        if img.GetDimension() == 4 and img.GetSize()[3] == 1:
            img = img[:, :, :, 0]
        # Reorient to LPS – handles oblique/rotated acquisitions
        try:
            img = sitk.DICOMOrient(img, "LPS")
        except Exception:
            _logger.warning("DICOMOrient failed for %s, using raw orientation", path)
        arr = sitk.GetArrayFromImage(img)  # shape: (D, H, W)
        if arr.ndim == 4:
            arr = arr[..., 0] if arr.shape[-1] == 1 else arr[0]
        spacing = img.GetSpacing()[:3]  # (sx, sy, sz)
        return arr, np.eye(4), spacing, img
    elif _HAS_NIB:
        nii = nib.load(path)
        # Reorient to closest canonical (RAS) then flip to LPS array order
        try:
            nii = nib.as_closest_canonical(nii)
        except Exception:
            _logger.warning("as_closest_canonical failed for %s", path)
        arr = np.asanyarray(nii.dataobj).squeeze()
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))  # to (D, H, W)
        affine = nii.affine
        spacing = tuple(np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)))[:3]
        return arr, affine, spacing, nii
    else:
        raise RuntimeError("SimpleITK or nibabel required to load NIfTI files.")


def find_nifti_files_recursive(folder):
    """Recursively find all NIfTI files in a folder and subfolders."""
    nifti = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            up = f.upper()
            if up.endswith(".NII") or up.endswith(".NII.GZ"):
                nifti.append(os.path.join(root, f))
    return nifti


def auto_assign_sequences(nifti_list):
    """Try to auto-assign NIfTI files to FLAIR/T1/T2/MASK based on filenames.

    Returns {"FLAIR": path_or_None, "T1": ..., "T2": ..., "MASK": ...}.
    """
    results = {"FLAIR": None, "T1": None, "T2": None, "MASK": None}
    for path in nifti_list:
        up = os.path.basename(path).upper()
        if results["FLAIR"] is None and "FLAIR" in up:
            results["FLAIR"] = path
        elif results["MASK"] is None and any(k in up for k in ("MASK", "SEG", "LABEL", "WMH")):
            results["MASK"] = path
        elif results["T2"] is None and "T2" in up:
            results["T2"] = path
        elif results["T1"] is None and ("T1" in up or "3DT1" in up):
            results["T1"] = path

    # Also try nnUNet channel convention: *_0000 = FLAIR, *_0001 = T1, *_0002 = T2
    if results["FLAIR"] is None or results["T1"] is None:
        for path in nifti_list:
            base = os.path.basename(path).upper()
            if results["FLAIR"] is None and "_0000" in base:
                results["FLAIR"] = path
            elif results["T1"] is None and "_0001" in base:
                results["T1"] = path
            elif results["T2"] is None and "_0002" in base:
                results["T2"] = path

    return results


# ── Slice extraction ─────────────────────────────────────────────────
def extract_slice(volume, axis, index):
    """Extract a 2D slice from a 3D volume along the given axis.
    Flips vertically so display matches radiological convention."""
    if volume is None:
        return None
    D, H, W = volume.shape
    if axis == 0:  # Axial
        if 0 <= index < D:
            return np.flipud(volume[index, :, :])
    elif axis == 1:  # Sagittal
        if 0 <= index < W:
            return np.flipud(volume[:, :, index])
    elif axis == 2:  # Coronal
        if 0 <= index < H:
            return np.flipud(volume[:, index, :])
    return None


def slice_dims(vol_shape, axis):
    """Return (height, width) of a 2D slice along the given axis."""
    D, H, W = vol_shape
    if axis == 0:
        return H, W
    elif axis == 1:
        return D, H
    elif axis == 2:
        return D, W
    return 1, 1


def volume_to_qimage(arr2d, ww=None, wc=None):
    """Convert a 2D numpy array to QImage (grayscale)."""
    if arr2d is None:
        return None
    arr = arr2d.astype(np.float64)
    if ww is not None and wc is not None:
        lo = wc - ww / 2
        hi = wc + ww / 2
        arr = np.clip(arr, lo, hi)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    arr = arr.astype(np.uint8)
    h, w = arr.shape
    return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


def labels_to_rgba(seg, colormap=None):
    """Convert a 2D label array to RGBA overlay."""
    if colormap is None:
        colormap = {k: v[1] for k, v in SEG_LABELS.items() if k > 0}
    h, w = seg.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for label, color in colormap.items():
        mask = seg == label
        if not np.any(mask):
            continue
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = 255
    return rgba


def prob_to_heatmap_rgba(prob_2d, threshold=0.05):
    """Convert a 2D float32 probability array to a colormap RGBA overlay."""
    h, w = prob_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Only map probabilities above threshold to avoid cluttering background
    mask = prob_2d >= threshold
    if not np.any(mask):
        return rgba
        
    vals = np.clip(prob_2d[mask], 0, 1)
    # Get turbo colormap from matplotlib (returns N, 4 floats 0-1)
    colors = cm.turbo(vals)
    
    rgba[mask, 0] = (colors[:, 0] * 255).astype(np.uint8)
    rgba[mask, 1] = (colors[:, 1] * 255).astype(np.uint8)
    rgba[mask, 2] = (colors[:, 2] * 255).astype(np.uint8)
    # Scale alpha by probability so lower probabilities fade out
    rgba[mask, 3] = (vals * 255).astype(np.uint8)
    
    return rgba

def classification_to_rgba(class_vol_2d):
    """Convert a 2D classification array to RGBA with CLASS_LABELS colors."""
    cmap = {k: v[1] for k, v in CLASS_LABELS.items() if k > 0}
    return labels_to_rgba(class_vol_2d, cmap)


# ── Bresenham ────────────────────────────────────────────────────────
def _bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy


# ─────────────────────────────────────────────────────────────────────
# SliceCanvas – zoomable/pannable viewer for one orthogonal plane
# ─────────────────────────────────────────────────────────────────────
class SliceCanvas(QWidget):
    scrollRequested = pyqtSignal(int)
    viewChanged = pyqtSignal(float, float, float)
    crosshairMoved = pyqtSignal(float, float)  # image-space x, y
    brushStroke = pyqtSignal(float, float, bool)
    brushReleased = pyqtSignal()
    lesionPicked = pyqtSignal(float, float)  # for growth/classification
    doubleClicked = pyqtSignal()

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._overlay_pixmap = None
        self._class_overlay_pixmap = None
        self._overlay_alpha = 0.45
        self._scale = 0.0
        self._aspect = (1.0, 1.0)  # (sx_mult, sy_mult) for voxel spacing
        self._offset = QPointF(0, 0)
        self._dragging = False
        self._last_mouse = QPointF()
        self._title = title
        self._crosshair = None  # (img_x, img_y)
        self._draw_mode = False
        self._painting = False
        self._brush_size = 3
        self._grow_mode = False
        self._classify_mode = False
        self._cursor_pos = None
        self._show_class_overlay = True
        # Scan animation state
        self._scan_line_h = None     # normalized 0..1 horizontal line position
        self._scan_line_v = None     # normalized 0..1 vertical line position
        self._scan_glow = False      # True = primary view, full illumination pulse
        self._scan_glow_h = None     # trailing glow end for horizontal line
        self._scan_glow_v = None     # trailing glow end for vertical line
        self._scan_phase = None      # "scan" | "reveal" | None
        self._lesion_flash_rects = []  # [(QRectF, alpha)] for flashing lesions
        self._scan_vibrate = 0.0     # horizontal shake offset in pixels
        self._lesion_rows = {}       # {row: [(x0,x1), ...]} precomputed for reveal
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def sync_view(self, scale, ox, oy):
        self._scale = scale
        self._offset = QPointF(ox, oy)
        self.update()

    def set_image(self, qimg, title=""):
        if qimg is None:
            self._pixmap = None
        else:
            self._pixmap = QPixmap.fromImage(qimg)
        if title:
            self._title = title
        if self._scale == 0.0:
            self._fit_image()
        self.update()

    def set_overlay(self, rgba_array):
        if rgba_array is None:
            self._overlay_pixmap = None
        else:
            h, w, _ = rgba_array.shape
            qimg = QImage(rgba_array.data, w, h, 4 * w,
                          QImage.Format.Format_RGBA8888).copy()
            self._overlay_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_class_overlay(self, rgba_array):
        if rgba_array is None:
            self._class_overlay_pixmap = None
        else:
            h, w, _ = rgba_array.shape
            qimg = QImage(rgba_array.data, w, h, 4 * w,
                          QImage.Format.Format_RGBA8888).copy()
            self._class_overlay_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_overlay_alpha(self, a):
        self._overlay_alpha = a
        self.update()

    def set_aspect(self, sx, sy):
        """Set pixel aspect ratio multipliers from voxel spacing."""
        self._aspect = (sx, sy)
        self._scale = 0.0  # force re-fit

    def set_crosshair(self, x, y):
        self._crosshair = (x, y)
        self.update()

    def clear(self):
        self._pixmap = None
        self._overlay_pixmap = None
        self._class_overlay_pixmap = None
        self._crosshair = None
        self.update()

    def _fit_image(self):
        if self._pixmap is None:
            return
        ax, ay = self._aspect
        pw, ph = self._pixmap.width() * ax, self._pixmap.height() * ay
        ww, wh = self.width(), self.height()
        sx = ww / pw if pw else 1
        sy = wh / ph if ph else 1
        self._scale = min(sx, sy) * 0.92
        self._offset = QPointF((ww - pw * self._scale) / 2,
                               (wh - ph * self._scale) / 2)

    def _img_to_widget(self, pt):
        ax, ay = self._aspect
        return QPointF(pt.x() * self._scale * ax + self._offset.x(),
                       pt.y() * self._scale * ay + self._offset.y())

    def _widget_to_img(self, pt):
        ax, ay = self._aspect
        return QPointF((pt.x() - self._offset.x()) / (self._scale * ax),
                       (pt.y() - self._offset.y()) / (self._scale * ay))

    def paintEvent(self, _ev):
        if self._pixmap is None:
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(4, 8, 16))
            p.setPen(QColor(74, 240, 255, 100))
            p.setFont(QFont("Courier New", 10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._title or "No data")
            p.end()
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(self._offset)
        ax, ay = self._aspect
        p.scale(self._scale * ax, self._scale * ay)
        p.drawPixmap(0, 0, self._pixmap)
        if self._overlay_pixmap is not None:
            p.setOpacity(self._overlay_alpha)
            p.drawPixmap(0, 0, self._overlay_pixmap)
            p.setOpacity(1.0)
        if self._class_overlay_pixmap is not None and self._show_class_overlay:
            p.setOpacity(self._overlay_alpha * 0.8)
            p.drawPixmap(0, 0, self._class_overlay_pixmap)
            p.setOpacity(1.0)

        # ── Scan animation (drawn in image-space while transform is active) ──
        _has_scan = (self._scan_line_h is not None or self._scan_line_v is not None
                     or self._scan_glow)
        if _has_scan and self._pixmap is not None:
            img_h = self._pixmap.height()
            img_w = self._pixmap.width()
            line_w = max(1.5, 2.0 / self._scale)
            vib = self._scan_vibrate

            # Full-view glow pulse (primary axis being scanned)
            if self._scan_glow:
                p.setBrush(QColor(74, 240, 255, 18))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(QRectF(0, 0, img_w, img_h))

            # Horizontal scanline
            if self._scan_line_h is not None:
                y_px = self._scan_line_h * img_h
                # Trailing glow
                if self._scan_glow_h is not None:
                    g_px = self._scan_glow_h * img_h
                    g_h = y_px - g_px
                    if g_h > 0:
                        grad = QLinearGradient(0, g_px, 0, y_px)
                        grad.setColorAt(0.0, QColor(74, 240, 255, 0))
                        grad.setColorAt(1.0, QColor(74, 240, 255, 40))
                        p.setBrush(QBrush(grad))
                        p.setPen(Qt.PenStyle.NoPen)
                        p.drawRect(QRectF(0, g_px, img_w, g_h))
                    elif g_h < 0:  # glow above line (reverse sweep)
                        grad = QLinearGradient(0, y_px, 0, g_px)
                        grad.setColorAt(0.0, QColor(74, 240, 255, 40))
                        grad.setColorAt(1.0, QColor(74, 240, 255, 0))
                        p.setBrush(QBrush(grad))
                        p.setPen(Qt.PenStyle.NoPen)
                        p.drawRect(QRectF(0, y_px, img_w, -g_h))
                # Line
                p.setPen(QPen(QColor(74, 240, 255, 200), line_w))
                p.drawLine(QPointF(vib, y_px), QPointF(img_w + vib, y_px))
                # Illumination band
                band = max(3, img_h * 0.02)
                p.setBrush(QColor(74, 240, 255, 35))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(QRectF(vib, y_px - band / 2, img_w, band))

            # Vertical scanline
            if self._scan_line_v is not None:
                x_px = self._scan_line_v * img_w
                # Trailing glow
                if self._scan_glow_v is not None:
                    g_px = self._scan_glow_v * img_w
                    g_w = x_px - g_px
                    if g_w > 0:
                        grad = QLinearGradient(g_px, 0, x_px, 0)
                        grad.setColorAt(0.0, QColor(74, 240, 255, 0))
                        grad.setColorAt(1.0, QColor(74, 240, 255, 40))
                        p.setBrush(QBrush(grad))
                        p.setPen(Qt.PenStyle.NoPen)
                        p.drawRect(QRectF(g_px, 0, g_w, img_h))
                # Line
                p.setPen(QPen(QColor(74, 240, 255, 200), line_w))
                p.drawLine(QPointF(x_px, vib), QPointF(x_px, img_h + vib))
                # Illumination band
                band = max(3, img_w * 0.02)
                p.setBrush(QColor(74, 240, 255, 35))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(QRectF(x_px - band / 2, vib, band, img_h))

            # Lesion flash rects (reveal phase)
            for rect, alpha in self._lesion_flash_rects:
                p.setBrush(QColor(255, 51, 102, int(alpha)))
                p.setPen(QPen(QColor(74, 240, 255, int(alpha * 0.7)), line_w))
                p.drawRect(rect)

        p.resetTransform()

        # crosshair
        if self._crosshair is not None and self._pixmap is not None:
            cx, cy = self._crosshair
            pw, ph = self._pixmap.width(), self._pixmap.height()
            # horizontal line
            wl = self._img_to_widget(QPointF(0, cy))
            wr = self._img_to_widget(QPointF(pw, cy))
            # vertical line
            wt = self._img_to_widget(QPointF(cx, 0))
            wb = self._img_to_widget(QPointF(cx, ph))
            p.setPen(QPen(QColor(74, 240, 255, 80), 1, Qt.PenStyle.DashLine))
            p.drawLine(wl, wr)
            p.drawLine(wt, wb)

        # title
        if self._title:
            p.setPen(QColor(74, 240, 255, 180))
            p.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
            p.drawText(8, 16, self._title)

        # brush cursor
        if (self._draw_mode or self._grow_mode or self._classify_mode) and self._cursor_pos is not None:
            img_c = self._widget_to_img(self._cursor_pos)
            r = self._brush_size if self._draw_mode else 5
            ax, ay = self._aspect
            r_wx = r * self._scale * ax
            r_wy = r * self._scale * ay
            wc = self._img_to_widget(img_c)
            if self._draw_mode:
                pen = QPen(QColor(255, 255, 255, 180), 1.5)
            elif self._grow_mode:
                pen = QPen(QColor(80, 255, 80, 220), 1.5, Qt.PenStyle.DashLine)
            else:
                pen = QPen(QColor(255, 200, 0, 220), 1.5, Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(wc, r_wx, r_wy)

        # Corner bracket accents
        p.setPen(QPen(QColor(74, 240, 255, 50), 1.5))
        L = 14
        ww, wh = self.width() - 1, self.height() - 1
        p.drawLine(1, 1, L, 1);   p.drawLine(1, 1, 1, L)
        p.drawLine(ww, 1, ww - L, 1);  p.drawLine(ww, 1, ww, L)
        p.drawLine(1, wh, L, wh); p.drawLine(1, wh, 1, wh - L)
        p.drawLine(ww, wh, ww - L, wh); p.drawLine(ww, wh, ww, wh - L)

        p.end()

    def wheelEvent(self, ev):
        if self._pixmap is None:
            return
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            old = self._scale
            factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
            self._scale = max(0.05, min(self._scale * factor, 50))
            cursor = ev.position()
            self._offset = cursor - (cursor - self._offset) * (self._scale / old)
            self.update()
            self.viewChanged.emit(self._scale, self._offset.x(), self._offset.y())
        else:
            delta = 1 if ev.angleDelta().y() < 0 else -1
            self.scrollRequested.emit(delta)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()

    def mousePressEvent(self, ev):
        if self._pixmap is None:
            return
        if (ev.button() == Qt.MouseButton.MiddleButton or
                (ev.button() == Qt.MouseButton.LeftButton and
                 ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
            self._dragging = True
            self._last_mouse = ev.position()
        elif ev.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_img(ev.position())
            if self._grow_mode:
                self.lesionPicked.emit(img_pt.x(), img_pt.y())
            elif self._classify_mode:
                self.lesionPicked.emit(img_pt.x(), img_pt.y())
            elif self._draw_mode:
                self._painting = True
                self.brushStroke.emit(img_pt.x(), img_pt.y(), True)
            else:
                self.crosshairMoved.emit(img_pt.x(), img_pt.y())

    def mouseMoveEvent(self, ev):
        if self._draw_mode or self._grow_mode or self._classify_mode:
            self._cursor_pos = ev.position()
            self.update()
        if self._dragging:
            delta = ev.position() - self._last_mouse
            self._offset += delta
            self._last_mouse = ev.position()
            self.update()
            self.viewChanged.emit(self._scale, self._offset.x(), self._offset.y())
        elif self._painting:
            img_pt = self._widget_to_img(ev.position())
            self.brushStroke.emit(img_pt.x(), img_pt.y(), False)
        elif bool(ev.buttons() & Qt.MouseButton.LeftButton) and not (self._draw_mode or self._grow_mode or self._classify_mode):
            img_pt = self._widget_to_img(ev.position())
            self.crosshairMoved.emit(img_pt.x(), img_pt.y())

    def mouseReleaseEvent(self, ev):
        if ev.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            if self._painting:
                self._painting = False
                self.brushReleased.emit()
            self._dragging = False

    def leaveEvent(self, ev):
        self._cursor_pos = None
        self.update()
        super().leaveEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._pixmap:
            self._fit_image()


# ─────────────────────────────────────────────────────────────────────
# Scan Animation (during AI prediction)
# ─────────────────────────────────────────────────────────────────────
class ScanAnimator(QObject):
    """Drives a coordinated 3-axis scanline animation across all 3 slice canvases.

    Scan phase: sweeps through D, then W, then H — each sweep shows lines on
    the two perpendicular views and a glow pulse on the primary view.
    Reveal phase: fast simultaneous sweep on all views with lesion flash.

    State machine: idle → scan (looping) → wait → reveal → done
    """
    finished = pyqtSignal()

    SWEEP_MS = 2700             # ms per axis sweep (3 axes ≈ 8s)
    REVEAL_DURATION_MS = 1500   # fast sweep after prediction
    TICK = 33                   # ~30 fps for reveal/idle
    SCAN_TICK = 50              # ~20 fps during scan (proven zero impact on inference)

    def __init__(self, canvases, parent=None):
        super().__init__(parent)
        self._canvases = canvases  # [axial, sagittal, coronal]
        self._app = parent  # MSLesionApp — for slice scrolling
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._phase = "idle"
        self._elapsed = 0
        self._seg_vol = None
        self._last_sweep_idx = 0   # track which sweep was last active
        self._last_t = 0.5         # track last scan position (0..1)

    # -- public API ---------------------------------------------------
    def start_scan(self):
        self._phase = "scan"
        self._elapsed = 0
        self._seg_vol = None
        self._last_sweep_idx = 0
        self._last_t = 0.0
        self._clear_canvases()
        self._timer.start(self.SCAN_TICK)  # slow tick during inference

    def prediction_done(self, seg_vol):
        self._seg_vol = seg_vol

    def stop(self):
        self._timer.stop()
        self._phase = "idle"
        self._clear_canvases()

    def _clear_canvases(self):
        for c in self._canvases:
            c._scan_line_h = None
            c._scan_line_v = None
            c._scan_glow = False
            c._scan_glow_h = None
            c._scan_glow_v = None
            c._scan_phase = None
            c._lesion_flash_rects = []
            c._scan_vibrate = 0.0
            c._lesion_rows = {}
            c.update()

    def _scroll_primary(self, view_idx, t):
        """Update the slider position and render the view during scan."""
        try:
            app = self._app
            sl = app._sliders[view_idx]
            max_val = sl.maximum()
            if max_val > 0:
                new_val = int(t * max_val)
                if new_val != app._slice_idx[view_idx]:
                    app._slice_idx[view_idx] = new_val
                    sl.blockSignals(True)
                    sl.setValue(new_val)
                    sl.blockSignals(False)
                    app._update_view(view_idx)
        except (AttributeError, IndexError):
            pass

    def _set_sliders_to(self, positions):
        """Set all sliders to specific positions. positions = [t0, t1, t2] (0..1)."""
        try:
            for vi, t in enumerate(positions):
                sl = self._app._sliders[vi]
                max_val = sl.maximum()
                if max_val > 0:
                    val = int(t * max_val)
                    self._app._slice_idx[vi] = val
                    sl.blockSignals(True)
                    sl.setValue(val)
                    sl.blockSignals(False)
            # Batch-update all views once
            for vi in range(3):
                self._app._update_view(vi)
        except (AttributeError, IndexError):
            pass

    # -- animation loop -----------------------------------------------
    def _tick(self):
        tick = self.TICK if self._phase == "reveal" else self.SCAN_TICK
        self._elapsed += tick

        if self._phase == "scan":
            self._tick_scan()
        elif self._phase == "wait":
            if self._elapsed > 200:
                self._start_reveal()
        elif self._phase == "reveal":
            self._tick_reveal()

    def _tick_scan(self):
        total_ms = self.SWEEP_MS * 3
        loop_t = self._elapsed % total_ms
        sweep_idx = min(int(loop_t / self.SWEEP_MS), 2)
        local_ms = loop_t - sweep_idx * self.SWEEP_MS

        # Two sub-phases: forward scan (0→1), return scan (1→0.5)
        forward_ms = self.SWEEP_MS * 0.60
        return_ms = self.SWEEP_MS * 0.40

        # Clear all canvases first
        for c in self._canvases:
            c._scan_line_h = None
            c._scan_line_v = None
            c._scan_glow = False
            c._scan_glow_h = None
            c._scan_glow_v = None
            c._scan_phase = "scan"

        ax, sag, cor = self._canvases

        if local_ms <= forward_ms:
            # ── Forward: 0 → 1 ──
            raw = local_ms / forward_ms
            t = raw * raw * (3 - 2 * raw)  # smoothstep
            trail = max(0, t - 0.12)
        else:
            # ── Return: 1 → 0.5 (scan lines active, reversed trail) ──
            raw = (local_ms - forward_ms) / return_ms
            s = raw * raw * (3 - 2 * raw)  # smoothstep
            t = 1.0 - s * 0.5              # 1.0 → 0.5
            trail = min(1.0, t + 0.12)     # trail behind in return direction

        # Track position for seamless reveal transition
        self._last_sweep_idx = sweep_idx
        self._last_t = t

        # Apply scan lines — same code for both phases, just different t/trail
        if sweep_idx == 0:
            ax._scan_glow = True
            sag._scan_line_h = 1.0 - t
            sag._scan_glow_h = 1.0 - trail
            cor._scan_line_h = 1.0 - t
            cor._scan_glow_h = 1.0 - trail
            self._scroll_primary(0, t)
        elif sweep_idx == 1:
            sag._scan_glow = True
            ax._scan_line_v = t
            ax._scan_glow_v = trail
            cor._scan_line_v = t
            cor._scan_glow_v = trail
            self._scroll_primary(1, t)
        else:
            cor._scan_glow = True
            ax._scan_line_h = 1.0 - t
            ax._scan_glow_h = 1.0 - trail
            sag._scan_line_v = t
            sag._scan_glow_v = trail
            self._scroll_primary(2, t)

        for c in self._canvases:
            c.update()

        # Transition to reveal when prediction is done (at end of current sweep)
        at_sweep_end = local_ms >= (forward_ms + return_ms - self.SCAN_TICK * 2)
        if self._seg_vol is not None and at_sweep_end:
            self._phase = "wait"
            self._elapsed = 0

    def _start_reveal(self):
        self._phase = "reveal"
        self._elapsed = 0
        # Switch to faster tick for smooth reveal (inference is done, no GIL concern)
        self._timer.start(self.TICK)
        # Remember starting slice positions for smooth interpolation to center
        try:
            self._reveal_start_pos = [self._app._slice_idx[vi] for vi in range(3)]
            self._reveal_end_pos = [self._app._sliders[vi].maximum() // 2 for vi in range(3)]
        except (AttributeError, IndexError):
            self._reveal_start_pos = [0, 0, 0]
            self._reveal_end_pos = [0, 0, 0]
        for c in self._canvases:
            c._scan_phase = "reveal"
            c._scan_line_h = 0.0
            c._scan_line_v = None
            c._scan_glow = False
            c._scan_glow_h = None
            c._scan_glow_v = None
            c._lesion_flash_rects = []
            c._scan_vibrate = 0.0
            self._precompute_lesion_rows(c)

    def _tick_reveal(self):
        import random
        t = min(self._elapsed / self.REVEAL_DURATION_MS, 1.0)
        # Smoothly scroll sliders from scan-end position to center
        ease = t * t * (3 - 2 * t)  # smoothstep
        for vi in range(3):
            try:
                start = self._reveal_start_pos[vi]
                end = self._reveal_end_pos[vi]
                target = int(start + (end - start) * ease)
                self._app._slice_idx[vi] = target
                sl = self._app._sliders[vi]
                sl.blockSignals(True)
                sl.setValue(target)
                sl.blockSignals(False)
            except (AttributeError, IndexError):
                pass
        for c in self._canvases:
            c._scan_line_h = t
            c._scan_glow_h = max(0, t - 0.1)
            self._check_lesion_hit(c, t, random)
            c.update()
        # Update views periodically during reveal for smooth scroll-to-center
        if self._elapsed % (self.TICK * 3) < self.TICK:
            for vi in range(3):
                try:
                    self._app._update_view(vi)
                except (AttributeError, IndexError):
                    pass
        if t >= 1.0:
            self._phase = "idle"
            self._clear_canvases()
            # Ensure final position is center
            self._set_sliders_to([0.5, 0.5, 0.5])
            self._timer.stop()
            self.finished.emit()

    # -- lesion detection for reveal -----------------------------------
    def _check_lesion_hit(self, canvas, t, random):
        if canvas._pixmap is None:
            return
        img_h = canvas._pixmap.height()
        scan_row = int(t * img_h)
        if scan_row in canvas._lesion_rows:
            canvas._scan_vibrate = random.uniform(-3, 3)
            for x0, x1 in canvas._lesion_rows[scan_row]:
                rect = QRectF(x0, scan_row - 1, x1 - x0, 3)
                canvas._lesion_flash_rects.append((rect, 180))
        else:
            canvas._scan_vibrate = 0.0
        canvas._lesion_flash_rects = [
            (r, a - 6) for r, a in canvas._lesion_flash_rects if a > 6
        ]

    def _precompute_lesion_rows(self, canvas):
        """Fast numpy-based detection of which rows have lesion overlay pixels."""
        canvas._lesion_rows = {}
        if canvas._overlay_pixmap is None:
            return
        img = canvas._overlay_pixmap.toImage().convertToFormat(
            QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        ptr = img.bits()
        if ptr is None:
            return
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        alpha = arr[:, :, 3]
        for y in range(h):
            row = alpha[y]
            mask = row > 20
            if not np.any(mask):
                continue
            diffs = np.diff(mask.astype(np.int8))
            starts = np.where(diffs == 1)[0] + 1
            ends = np.where(diffs == -1)[0] + 1
            if mask[0]:
                starts = np.concatenate([[0], starts])
            if mask[-1]:
                ends = np.concatenate([ends, [w]])
            canvas._lesion_rows[y] = list(zip(starts.tolist(), ends.tolist()))


class StatsAnimator(QObject):
    """Animates count-up numbers in the stats labels after scan reveal."""

    def __init__(self, lbl_stats, lbl_3d, parent=None):
        super().__init__(parent)
        self._lbl_stats = lbl_stats
        self._lbl_3d = lbl_3d
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._target_vox = 0
        self._target_vol = 0.0
        self._elapsed = 0
        self._duration = 1200  # ms

    def animate(self, total_voxels, volume_mm3):
        self._target_vox = total_voxels
        self._target_vol = volume_mm3
        self._elapsed = 0
        self._timer.start(16)

    def _tick(self):
        self._elapsed += 16
        t = min(self._elapsed / self._duration, 1.0)
        ease = t * t * (3 - 2 * t)  # smoothstep
        vox = int(self._target_vox * ease)
        vol = self._target_vol * ease
        self._lbl_stats.setText(f"Total lesion: {vox} vox ({vol:.1f} mm\u00b3)")
        self._lbl_3d.setText(f"Lesion Volume: {vol:.1f} mm\u00b3\nTotal voxels: {vox}")
        if t >= 1.0:
            self._timer.stop()


# ─────────────────────────────────────────────────────────────────────
# DrawingToolbar
# ─────────────────────────────────────────────────────────────────────
class DrawingToolbar(QWidget):
    drawingToggled = pyqtSignal(bool)
    labelChanged = pyqtSignal(int)
    brushSizeChanged = pyqtSignal(int)
    eraserToggled = pyqtSignal(bool)
    growModeToggled = pyqtSignal(bool)
    growThresholdChanged = pyqtSignal(int)
    classifyModeToggled = pyqtSignal(bool)
    classLabelChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Tools")
        self.setObjectName("toolbar_drawing")
        self.setFixedWidth(230)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # ── Drawing section ──
        sep1 = QLabel("── Drawing ──")
        lay.addWidget(sep1)

        self._chk_draw = QCheckBox("Enable Drawing")
        self._chk_draw.toggled.connect(self.drawingToggled.emit)
        lay.addWidget(self._chk_draw)

        tool_row = QHBoxLayout()
        self._btn_brush = QToolButton(); self._btn_brush.setText("🖌 Brush")
        self._btn_brush.setCheckable(True); self._btn_brush.setChecked(True)
        self._btn_eraser = QToolButton(); self._btn_eraser.setText("🧽 Eraser")
        self._btn_eraser.setCheckable(True)
        self._tool_grp = QButtonGroup(self)
        self._tool_grp.setExclusive(True)
        self._tool_grp.addButton(self._btn_brush, 0)
        self._tool_grp.addButton(self._btn_eraser, 1)
        self._tool_grp.idToggled.connect(lambda bid, c: c and self.eraserToggled.emit(bid == 1))
        tool_row.addWidget(self._btn_brush)
        tool_row.addWidget(self._btn_eraser)
        lay.addLayout(tool_row)

        lay.addWidget(QLabel("Paint label:"))
        self._label_btns = QButtonGroup(self)
        self._label_btns.setExclusive(True)
        grid = QGridLayout()
        for lbl_id in SEG_LABELS.keys():
            name, color = SEG_LABELS[lbl_id]
            btn = QToolButton(); btn.setText(name); btn.setCheckable(True)
            btn.setStyleSheet(
                f"QToolButton {{ border:2px solid rgb({color[0]},{color[1]},{color[2]}); padding:2px; }}"
                f" QToolButton:checked {{ background:rgb({color[0]},{color[1]},{color[2]});"
                f" color:{'#000' if sum(color) > 400 else '#fff'}; }}")
            if lbl_id == 1:
                btn.setChecked(True)
            self._label_btns.addButton(btn, lbl_id)
            grid.addWidget(btn, lbl_id // 2, lbl_id % 2)
        self._label_btns.idToggled.connect(lambda bid, c: c and self.labelChanged.emit(bid))
        lay.addLayout(grid)

        lay.addWidget(QLabel("Brush size:"))
        sz_row = QHBoxLayout()
        self._sl_brush = QSlider(Qt.Orientation.Horizontal)
        self._sl_brush.setRange(1, 30); self._sl_brush.setValue(3)
        self._sl_brush.valueChanged.connect(self._on_brush_sz)
        sz_row.addWidget(self._sl_brush, 1)
        self._lbl_sz = QLabel("3"); sz_row.addWidget(self._lbl_sz)
        lay.addLayout(sz_row)

        # ── Lesion Growth section ──
        sep2 = QLabel("── Lesion Growth ──")
        lay.addWidget(sep2)

        self._chk_grow = QCheckBox("Growth Mode (click to grow)")
        self._chk_grow.toggled.connect(self.growModeToggled.emit)
        lay.addWidget(self._chk_grow)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Threshold:"))
        self._sl_thr = QSlider(Qt.Orientation.Horizontal)
        self._sl_thr.setRange(5, 95); self._sl_thr.setValue(30)
        self._sl_thr.valueChanged.connect(self._on_thr)
        thr_row.addWidget(self._sl_thr, 1)
        self._lbl_thr = QLabel("0.30"); thr_row.addWidget(self._lbl_thr)
        lay.addLayout(thr_row)

        # ── Classification section ──
        sep3 = QLabel("── Classification ──")
        lay.addWidget(sep3)

        self._chk_classify = QCheckBox("Classify Mode (click lesion)")
        self._chk_classify.toggled.connect(self.classifyModeToggled.emit)
        lay.addWidget(self._chk_classify)

        lay.addWidget(QLabel("Lesion type:"))
        self._class_btns = QButtonGroup(self)
        self._class_btns.setExclusive(True)
        cgrid = QGridLayout()
        for cid in range(1, 7):
            name, color = CLASS_LABELS[cid]
            btn = QToolButton(); btn.setText(name); btn.setCheckable(True)
            btn.setStyleSheet(
                f"QToolButton {{ border:2px solid rgb({color[0]},{color[1]},{color[2]}); padding:2px; font-size:10px; }}"
                f" QToolButton:checked {{ background:rgb({color[0]},{color[1]},{color[2]});"
                f" color:#000; }}")
            if cid == 1:
                btn.setChecked(True)
            self._class_btns.addButton(btn, cid)
            cgrid.addWidget(btn, (cid - 1) // 2, (cid - 1) % 2)
        self._class_btns.idToggled.connect(lambda bid, c: c and self.classLabelChanged.emit(bid))
        lay.addLayout(cgrid)

        lay.addStretch()

    def _on_brush_sz(self, v):
        self._lbl_sz.setText(str(v)); self.brushSizeChanged.emit(v)

    def _on_thr(self, v):
        self._lbl_thr.setText(f"{v/100:.2f}"); self.growThresholdChanged.emit(v)

    def set_brush_size(self, v): self._sl_brush.setValue(v)
    def select_brush(self): self._btn_brush.setChecked(True)
    def select_eraser(self): self._btn_eraser.setChecked(True)
    def set_drawing_enabled(self, e):
        if self._chk_draw.isChecked() != e: self._chk_draw.setChecked(e)
    def set_grow_enabled(self, e):
        if self._chk_grow.isChecked() != e: self._chk_grow.setChecked(e)
    def set_classify_enabled(self, e):
        if self._chk_classify.isChecked() != e: self._chk_classify.setChecked(e)


# ─────────────────────────────────────────────────────────────────────
# nnUNet 3D predictor
# ─────────────────────────────────────────────────────────────────────
def _create_predictor_3d(model_dir, device_str, folds=(0,), use_mirroring=True):
    if device_str == "cuda":
        try:
            # Verify CUDA actually works (forward compat handles newer GPUs)
            a = torch.ones(2, 2, device="cuda")
            _ = (a @ a).sum().item(); del a
        except Exception:
            device_str = "cpu"

    device = torch.device(device_str)
    on_device = device_str != "cpu"

    if device_str == "cpu":
        import multiprocessing
        # Maximize CPU utilization
        torch.set_num_threads(multiprocessing.cpu_count())
        # Increase step size to reduce number of patches (massive CPU speedup)
        step_sz = 0.8
    else:
        # Standard accuracy for GPU
        step_sz = 0.5

    predictor = nnUNetPredictor(
        tile_step_size=step_sz, use_gaussian=True, use_mirroring=use_mirroring,
        perform_everything_on_device=on_device,
        device=device, verbose=False, verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=tuple(folds), checkpoint_name="checkpoint_best.pth",
    )
    return predictor, device_str


# ─────────────────────────────────────────────────────────────────────
# nnUNet 2.5D predictor (slice-by-slice with K-context)
# ─────────────────────────────────────────────────────────────────────
_SLICE_AXIS_PERM = {
    0: None,            # Axial:    D is already axis 1
    1: (0, 2, 1, 3),    # Coronal:  swap D↔H
    2: (0, 3, 2, 1),    # Sagittal: swap D↔W
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


def _create_predictor_25d(model_dir, device_str, folds=(0,), K=7, slice_axis=0, use_mirroring=True):
    """Create a 2.5D predictor for slice-by-slice inference with K-context."""
    if device_str == "cuda":
        try:
            a = torch.ones(2, 2, device="cuda")
            _ = (a @ a).sum().item(); del a
        except Exception:
            device_str = "cpu"

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
        tile_step_size=step_sz, use_gaussian=True, use_mirroring=use_mirroring,
        perform_everything_on_device=on_device,
        device=device, verbose=False, verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=tuple(folds), checkpoint_name="checkpoint_best.pth",
    )
    return predictor, device_str


def _create_predictor(arch_key, model_dir, device_str, folds=(0,), use_mirroring=True):
    """Dispatcher: create the right predictor type for the architecture.
    If device_str starts with 'onnx', uses ONNX Runtime backend."""
    if device_str.startswith("onnx"):
        from msseg.inference_ort import create_ort_predictor, check_onnx_available
        fold = folds[0]
        if not check_onnx_available(model_dir, fold):
            _logger.warning("ONNX model not found for %s fold %d, falling back to PyTorch", arch_key, fold)
            # Fall back to PyTorch with the correct device
            device_str = "cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu"
        else:
            return create_ort_predictor(arch_key, model_dir, folds=folds,
                                        device_str=device_str, use_mirroring=use_mirroring)
    pred_type = ARCHITECTURES[arch_key][1]
    if pred_type == "25d":
        predictor, dev = _create_predictor_25d(model_dir, device_str, folds, use_mirroring=use_mirroring)
    else:
        predictor, dev = _create_predictor_3d(model_dir, device_str, folds, use_mirroring=use_mirroring)
    # Models saved with mixed precision have float16 weights — cast for CPU
    if dev == "cpu":
        predictor.network = predictor.network.float()
    elif dev == "cuda":
        # fp16 weights on GPU + eval mode — ready to predict immediately
        predictor.network = predictor.network.half().to(torch.device('cuda'))
        predictor.network.eval()
    return predictor, dev


class PreloadModelThread(QThread):
    progress = pyqtSignal(str)

    def __init__(self, ensemble_config, device_str, predictors_cache=None, parent=None):
        super().__init__(parent)
        self._ensemble_config = ensemble_config  # {arch_key: (fold_tuple,)}
        self._device_str = device_str
        self._predictors = predictors_cache if predictors_cache is not None else {}

    def run(self):
        try:
            # ORT GPU sessions are heavyweight — skip preloading to avoid VRAM exhaustion
            if self._device_str in ("onnx-cuda", "onnx-trt"):
                self.progress.emit("ORT GPU mode: models will be loaded on demand during inference.")
                return
            total = sum(len(folds) for folds in self._ensemble_config.values())
            loaded = 0
            for arch_key, folds in self._ensemble_config.items():
                subdir = ARCHITECTURES[arch_key][0]
                display = ARCHITECTURES[arch_key][2]
                model_dir = _resolve_model_dir(subdir)
                if model_dir is None:
                    self.progress.emit(f"WARNING: {display} not found, skipping")
                    continue
                arch_cache = self._predictors.setdefault(arch_key, {})
                for fold in folds:
                    if fold in arch_cache:
                        loaded += 1
                        continue
                    self.progress.emit(f"Preloading [{loaded+1}/{total}] {display} fold {fold}...")
                    predictor, _ = _create_predictor(arch_key, model_dir, self._device_str, folds=(fold,))
                    arch_cache[fold] = predictor
                    loaded += 1
            self.progress.emit(f"All {loaded} models preloaded and ready.")
        except Exception as e:
            self.progress.emit(f"Preload failed: {e}")


class MeshBuilderThread(QThread):
    finished = pyqtSignal(object, object, object)  # (verts, faces, colors)

    def __init__(self, mask_vol, class_vol, spacing, parent=None):
        super().__init__(parent)
        self.mask_vol = mask_vol.copy()
        self.class_vol = class_vol.copy() if class_vol is not None else None
        # spacing comes in as (sx,sy,sz)=(x,y,z); convert to array axis order (z,y,x)
        self.spacing = spacing
        self.mc_spacing = (spacing[2], spacing[1], spacing[0])  # (sz,sy,sx) for marching_cubes

    def run(self):
        if not _HAS_SKM or not np.any(self.mask_vol):
            self.finished.emit(None, None, None)
            return
        try:
            verts, faces, _, _ = marching_cubes(self.mask_vol, level=0.5, spacing=self.mc_spacing)

            colors = np.zeros((len(verts), 4), dtype=np.float32)
            colors[:, :] = (1.0, 0.3, 0.3, 0.8)  # Default reddish

            if self.class_vol is not None and np.any(self.class_vol):
                from scipy.ndimage import maximum_filter
                dilated_class = maximum_filter(self.class_vol, size=3)

                # Map physical vertices back to voxel indices using mc_spacing
                vox_coords = np.clip(np.round(verts / np.array(self.mc_spacing)).astype(int),
                                     0, np.array(self.class_vol.shape) - 1)
                v_classes = dilated_class[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]]
                
                for cid in range(1, 6):
                    mask = v_classes == cid
                    if np.any(mask):
                        r, g, b = CLASS_LABELS[cid][1]
                        colors[mask, 0] = r / 255.0
                        colors[mask, 1] = g / 255.0
                        colors[mask, 2] = b / 255.0
                        colors[mask, 3] = 0.9

            # Rotate actual final 3D scene so Axial (D) runs along PyQTGraph's native Z-up axis!
            verts = verts[:, [2, 1, 0]]
            self.finished.emit(verts, faces, colors)
        except Exception as e:
            _logger.error("Mesh error: %s", e)
            self.finished.emit(None, None, None)


# ─────────────────────────────────────────────────────────────────────
# Segmentation Thread (3D volume)
# ─────────────────────────────────────────────────────────────────────
class SegmentationThread(QThread):
    progress = pyqtSignal(str)
    progressPercent = pyqtSignal(int)   # 0-100
    finished = pyqtSignal(bool, str)
    warmedUp = pyqtSignal()  # emitted after cuDNN warmup, safe to start animation

    def __init__(self, ensemble_config, device_str, flair_arr, t1_arr,
                 spacing, predictors_cache=None, use_mirroring=True, parent=None):
        super().__init__(parent)
        self._ensemble_config = ensemble_config  # {arch_key: (fold_tuple,)}
        self._device_str = device_str
        self._flair = flair_arr
        self._t1 = t1_arr
        self._spacing = spacing
        self._predictors = predictors_cache if predictors_cache is not None else {}
        self._use_mirroring = use_mirroring
        self.fold_probs = {}  # {(arch_key, fold): prob_array}
        self.device_used = device_str

    def _ensure_predictor(self, arch_key, fold, model_dir, label):
        """Return a predictor for the given arch/fold, loading if needed."""
        arch_cache = self._predictors.setdefault(arch_key, {})
        display_name = ARCHITECTURES[arch_key][2]
        if fold not in arch_cache:
            self.progress.emit(f"{label} {display_name} fold {fold} -- loading...")
            predictor, dev = _create_predictor(
                arch_key, model_dir, self._device_str, folds=(fold,),
                use_mirroring=self._use_mirroring)
            arch_cache[fold] = predictor
            self.device_used = dev
        else:
            predictor = arch_cache[fold]
        predictor.use_mirroring = self._use_mirroring
        return predictor

    def _preprocess_once(self, predictor, img, props):
        """Run nnUNet preprocessing — returns (data_tensor, properties_dict)."""
        from nnunetv2.inference.predict_from_raw_data import PreprocessAdapterFromNpy
        ppa = PreprocessAdapterFromNpy(
            [img], [None], [props], [None],
            predictor.plans_manager, predictor.dataset_json,
            predictor.configuration_manager,
            num_threads_in_multithreaded=1, verbose=False)
        dct = next(ppa)
        return dct['data'], dct['data_properties']

    def _predict_from_preprocessed(self, predictor, data_tensor, data_properties):
        """Run prediction on already-preprocessed data — returns (seg, probs)."""
        from nnunetv2.inference.predict_from_raw_data import (
            convert_predicted_logits_to_segmentation_with_correct_shape)
        # Log predictor state for debugging
        try:
            net_dev = next(predictor.network.parameters()).device
        except Exception:
            net_dev = "?"
        _logger.info("    predictor: device=%s net_device=%s perform_on_device=%s "
                      "mirroring=%s n_params_sets=%d tile_step=%.2f",
                      predictor.device, net_dev,
                      predictor.perform_everything_on_device,
                      predictor.use_mirroring,
                      len(predictor.list_of_parameters),
                      predictor.tile_step_size)
        t0 = time.time()
        logits = predictor.predict_logits_from_preprocessed_data(data_tensor)
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        logits_cpu = logits.cpu()
        t2 = time.time()
        result = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits_cpu, predictor.plans_manager, predictor.configuration_manager,
            predictor.label_manager, data_properties, return_probabilities=True)
        t3 = time.time()
        _logger.info("    predict_logits: %.3fs, .cpu(): %.3fs, postproc: %.3fs",
                      t1 - t0, t2 - t1, t3 - t2)
        return result

    def run(self):
        try:
            t0 = time.time()
            _logger.info("=== SegmentationThread.run() START ===")
            _logger.info("  device=%s, use_mirroring=%s", self._device_str, self._use_mirroring)

            img = np.stack([self._flair, self._t1], axis=0).astype(np.float32)
            props = {'spacing': list(self._spacing)}
            _logger.info("  img stack: %.3fs", time.time() - t0)

            if self._device_str == "cpu" and _HAS_TORCH:
                import multiprocessing
                physical_threads = max(1, multiprocessing.cpu_count() // 2)
                torch.set_num_threads(physical_threads)
                self.progress.emit(f"CPU mode: {physical_threads} threads")
            elif self._device_str.startswith("onnx"):
                self.progress.emit(f"ONNX Runtime mode: {self._device_str}")

            # Collect all (arch_key, fold, model_dir) tasks
            tasks = []
            for arch_key, folds in self._ensemble_config.items():
                subdir, pred_type, display_name = ARCHITECTURES[arch_key]
                model_dir = _resolve_model_dir(subdir)
                if model_dir is None:
                    self.progress.emit(f"WARNING: {display_name} not found, skipping")
                    continue
                for fold in folds:
                    tasks.append((arch_key, fold, model_dir))

            total = len(tasks)
            if total == 0:
                self.finished.emit(False, "No models found.")
                return

            # ── Disable cuDNN benchmark auto-tuning ─────────────────────
            # nnUNet sets cudnn.benchmark=True, which triggers expensive
            # per-shape algorithm trials on the first forward pass.
            # These trials need frequent GIL grabs, and the Qt event loop
            # causes massive GIL contention (6-9s per arch instead of <1s).
            # Since cuDNN handles are thread_local in PyTorch, any tuning
            # done in PreloadModelThread is lost anyway.
            # With benchmark=False, cuDNN uses fast heuristic selection
            # (~same speed for our patch sizes, zero warmup cost).
            if _HAS_TORCH:
                torch.backends.cudnn.benchmark = False
            # Signal main thread: safe to start animation
            self.warmedUp.emit()

            # Log predictor cache state and CUDA memory
            for ak in self._predictors:
                cached_folds = list(self._predictors[ak].keys())
                if cached_folds:
                    p0 = self._predictors[ak][cached_folds[0]]
                    try:
                        dev = next(p0.network.parameters()).device
                    except (AttributeError, StopIteration):
                        dev = getattr(p0, '_device', '?')
                    on_dev = getattr(p0, 'perform_everything_on_device', '?')
                    step = getattr(p0, 'tile_step_size', '?')
                    mirror = getattr(p0, 'use_mirroring', '?')
                    n_params = len(p0.list_of_parameters) if p0.list_of_parameters else 0
                    _logger.info("  cache[%s] folds=%s device=%s on_device=%s step=%s mirror=%s n_params=%d",
                                 ak, cached_folds, dev, on_dev, step, mirror, n_params)
            if _HAS_TORCH and torch.cuda.is_available():
                _logger.info("  CUDA memory: allocated=%.0fMB reserved=%.0fMB",
                             torch.cuda.memory_allocated() / 1e6,
                             torch.cuda.memory_reserved() / 1e6)

            # Decide parallelism
            t_res = time.time()
            resources = detect_compute_resources()
            _logger.info("  detect_compute_resources: %.3fs", time.time() - t_res)
            is_ort = self._device_str.startswith("onnx")
            if is_ort and self._device_str != "onnx-cpu":
                vram = resources["vram_free_mb"]
            elif self._device_str == "cuda":
                vram = resources["vram_free_mb"]
            else:
                vram = 0  # CPU mode — use RAM-based planning
            max_workers = plan_parallel_execution(
                self._ensemble_config, vram, resources["ram_free_mb"])

            # Force sequential for GPU modes (CUDA stream conflicts / ORT OOM)
            if self._device_str in ("cuda", "onnx-cuda", "onnx-trt"):
                max_workers = 1

            gpu_info = resources['gpu_name'] or 'CPU'
            vram_info = f"{resources['vram_free_mb']}MB free" if resources['vram_free_mb'] > 0 else "N/A"
            ram_info = f"{resources['ram_free_mb']}MB free"
            self.progress.emit(
                f"Hardware: {gpu_info} (VRAM: {vram_info}, RAM: {ram_info}) → "
                f"{max_workers} parallel model(s)")
            _logger.info("  setup total: %.3fs", time.time() - t0)

            if max_workers > 1 and is_ort:
                # ── Parallel ORT execution ──
                self.progress.emit(f"Running {total} models with {max_workers} parallel workers...")

                # Pre-load all predictors sequentially (ORT session creation not thread-safe)
                for i, (arch_key, fold, model_dir) in enumerate(tasks):
                    arch_cache = self._predictors.setdefault(arch_key, {})
                    display_name = ARCHITECTURES[arch_key][2]
                    if fold not in arch_cache:
                        self.progress.emit(
                            f"[{i+1}/{total}] Loading {display_name} fold {fold}...")
                        predictor, dev = _create_predictor(
                            arch_key, model_dir, self._device_str, folds=(fold,),
                            use_mirroring=self._use_mirroring)
                        arch_cache[fold] = predictor
                        self.device_used = dev

                # Run predictions in parallel
                import concurrent.futures
                # Sync TTA on all cached predictors before parallel run
                for arch_key, fold, _model_dir in tasks:
                    self._predictors[arch_key][fold].use_mirroring = self._use_mirroring
                def _predict(task_info):
                    arch_key, fold, _model_dir = task_info
                    predictor = self._predictors[arch_key][fold]
                    _seg, probs = predictor.predict_single_npy_array(
                        img, props, output_file_truncated=None,
                        save_or_return_probabilities=True)
                    return (arch_key, fold), probs

                self.progress.emit(f"Predicting {total} models ({max_workers} parallel)...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_predict, t): t for t in tasks}
                    done_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        key, probs = future.result()
                        self.fold_probs[key] = probs
                        done_count += 1
                        arch_key, fold = key
                        display_name = ARCHITECTURES[arch_key][2]
                        self.progress.emit(
                            f"[{done_count}/{total}] {display_name} fold {fold} done")
                        self.progressPercent.emit(int(done_count / total * 100))
            else:
                # ── Sequential execution (preprocess once per architecture) ──
                from itertools import groupby
                done = 0
                _logger.info("  === Sequential path ===")
                # Group tasks by arch_key to share preprocessing
                for arch_key, arch_tasks in groupby(tasks, key=lambda t: t[0]):
                    arch_tasks = list(arch_tasks)
                    display_name = ARCHITECTURES[arch_key][2]
                    first_fold = arch_tasks[0][1]
                    model_dir = arch_tasks[0][2]

                    # Ensure first predictor is loaded (for preprocessing config)
                    t_ensure = time.time()
                    predictor = self._ensure_predictor(
                        arch_key, first_fold, model_dir, f"[{done+1}/{total}]")
                    _logger.info("  ensure_predictor(%s, fold %d): %.3fs (cache %s)",
                                 arch_key, first_fold, time.time() - t_ensure,
                                 "hit" if time.time() - t_ensure < 0.1 else "MISS")

                    # Preprocess once for this architecture
                    self.progress.emit(f"Preprocessing for {display_name}...")
                    t_pp = time.time()
                    data_tensor, data_props = self._preprocess_once(predictor, img, props)
                    dt_pp = time.time() - t_pp
                    _logger.info("  preprocess(%s): %.3fs, tensor shape=%s device=%s",
                                 arch_key, dt_pp,
                                 tuple(data_tensor.shape) if hasattr(data_tensor, 'shape') else '?',
                                 data_tensor.device if hasattr(data_tensor, 'device') else '?')

                    # Run all folds with cached preprocessed data
                    for arch_key_f, fold, model_dir_f in arch_tasks:
                        done += 1
                        t_ensure2 = time.time()
                        pred = self._ensure_predictor(
                            arch_key_f, fold, model_dir_f, f"[{done}/{total}]")
                        dt_ensure2 = time.time() - t_ensure2
                        self.progress.emit(
                            f"[{done}/{total}] {display_name} fold {fold} -- predicting...")

                        # Time predict_logits separately from postprocessing
                        t_pred = time.time()
                        seg, prb = self._predict_from_preprocessed(
                            pred, data_tensor, data_props)
                        dt = time.time() - t_pred
                        _logger.info("  predict(%s fold %d): %.3fs (ensure=%.3fs) prb shape=%s",
                                     arch_key_f, fold, dt, dt_ensure2,
                                     tuple(prb.shape) if hasattr(prb, 'shape') else '?')
                        self.progress.emit(
                            f"[{done}/{total}] {display_name} fold {fold} -- {dt:.1f}s (prep {dt_pp:.1f}s)")
                        self.fold_probs[(arch_key_f, fold)] = prb
                        self.progressPercent.emit(int(done / total * 100))

                    # ORT GPU: release sessions after each arch to free VRAM
                    if is_ort and self._device_str != "onnx-cpu":
                        arch_cache = self._predictors.get(arch_key, {})
                        for fold_key in list(arch_cache.keys()):
                            del arch_cache[fold_key]
                        if _HAS_TORCH and torch.cuda.is_available():
                            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            _logger.info("=== SegmentationThread.run() DONE: %.3fs ===", elapsed)
            mode = f"parallel({max_workers})" if max_workers > 1 and is_ort else "sequential"
            self.finished.emit(True, f"Predicted {total} models in {elapsed:.1f}s ({mode})")
        except Exception as exc:
            import traceback
            _logger.error("Segmentation error:\n%s", traceback.format_exc())
            self.finished.emit(False, str(exc))

# ─────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────
if _HAS_PG:
    import OpenGL.GL as native_gl
    
    class FullScreenGLView(gl.GLViewWidget):
        doubleClicked = pyqtSignal()
        def mouseDoubleClickEvent(self, ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                self.doubleClicked.emit()
            super().mouseDoubleClickEvent(ev)

    class UnlitGLImageItem(gl.GLImageItem):
        def paint(self):
            native_gl.glPushAttrib(native_gl.GL_ENABLE_BIT)
            native_gl.glDisable(native_gl.GL_LIGHTING)
            native_gl.glDisable(native_gl.GL_CULL_FACE)
            super().paint()
            native_gl.glPopAttrib()


# ─────────────────────────────────────────────────────────────────────
# Custom Frameless Title Bar
# ─────────────────────────────────────────────────────────────────────
_TITLE_H = 32
_RESIZE_GRIP = 5


class TitleBar(QWidget):
    """Custom HUD-style title bar with bracket accents and window controls."""

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self._win = parent_window
        self._drag_pos = None
        self.setFixedHeight(_TITLE_H)
        self.setMouseTracking(True)
        # hover state for buttons
        self._hover_btn = None  # "min" | "max" | "close" | None

    # -- layout helpers ------------------------------------------------
    def _btn_rects(self):
        w = self.width()
        bw = 36
        y = 0
        h = _TITLE_H
        close = QRectF(w - bw, y, bw, h)
        maxi = QRectF(w - bw * 2, y, bw, h)
        mini = QRectF(w - bw * 3, y, bw, h)
        return {"min": mini, "max": maxi, "close": close}

    # -- painting ------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(self.rect(), QColor(4, 8, 16))
        # Bottom border line
        p.setPen(QPen(QColor(74, 240, 255, 40), 1))
        p.drawLine(0, h - 1, w, h - 1)

        # Left bracket accent
        p.setPen(QPen(QColor(74, 240, 255, 120), 1.5))
        p.drawLine(6, 6, 6, h - 8)
        p.drawLine(6, 6, 14, 6)
        p.drawLine(6, h - 8, 14, h - 8)

        # Title text
        p.setPen(QColor(74, 240, 255, 200))
        p.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        p.drawText(22, 0, w - 140, h, Qt.AlignmentFlag.AlignVCenter,
                   "MSLesionTool")
        # Subtitle
        p.setPen(QColor(74, 240, 255, 80))
        p.setFont(QFont("Courier New", 8))
        p.drawText(155, 0, w - 280, h, Qt.AlignmentFlag.AlignVCenter,
                   "MS Lesion Segmentation")

        # Right bracket accent (placed right after subtitle)
        p.setPen(QPen(QColor(74, 240, 255, 120), 1.5))
        sub_font = QFont("Courier New", 8)
        sub_w = QFontMetricsF(sub_font).horizontalAdvance("MS Lesion Segmentation")
        rr = int(155 + sub_w + 10)
        p.drawLine(rr, 6, rr, h - 8)
        p.drawLine(rr, 6, rr - 8, 6)
        p.drawLine(rr, h - 8, rr - 8, h - 8)

        # Window buttons
        rects = self._btn_rects()
        for key in ("min", "max", "close"):
            r = rects[key]
            if self._hover_btn == key:
                bg = QColor(255, 51, 102, 60) if key == "close" else QColor(74, 240, 255, 30)
                p.fillRect(r, bg)
            color = QColor(255, 51, 102, 220) if (key == "close" and self._hover_btn == "close") \
                else QColor(74, 240, 255, 160)
            p.setPen(QPen(color, 1.5))
            p.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
            cx, cy = r.center().x(), r.center().y()
            if key == "min":
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, "\u2500")
            elif key == "max":
                if self._win.isMaximized():
                    p.drawText(r, Qt.AlignmentFlag.AlignCenter, "\u25a3")
                else:
                    p.drawText(r, Qt.AlignmentFlag.AlignCenter, "\u25a1")
            else:
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, "\u00d7")

        p.end()

    # -- mouse interaction ---------------------------------------------
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            rects = self._btn_rects()
            pos = ev.position()
            for key in ("min", "max", "close"):
                if rects[key].contains(pos):
                    if key == "min":
                        self._win.showMinimized()
                    elif key == "max":
                        if self._win.isMaximized():
                            self._win.showNormal()
                        else:
                            self._win.showMaximized()
                    else:
                        self._win.close()
                    return
            self._drag_pos = ev.globalPosition().toPoint()

    def mouseMoveEvent(self, ev):
        # Drag
        if self._drag_pos is not None and ev.buttons() & Qt.MouseButton.LeftButton:
            if self._win.isMaximized():
                self._win.showNormal()
                # Re-center on cursor after unmaximize
                geo = self._win.geometry()
                self._drag_pos = ev.globalPosition().toPoint()
                self._win.move(self._drag_pos.x() - geo.width() // 2,
                               self._drag_pos.y() - _TITLE_H // 2)
                self._drag_pos = ev.globalPosition().toPoint()
            else:
                delta = ev.globalPosition().toPoint() - self._drag_pos
                self._win.move(self._win.pos() + delta)
                self._drag_pos = ev.globalPosition().toPoint()
        # Hover detection
        rects = self._btn_rects()
        pos = ev.position()
        new_hover = None
        for key in ("min", "max", "close"):
            if rects[key].contains(pos):
                new_hover = key
                break
        if new_hover != self._hover_btn:
            self._hover_btn = new_hover
            self.update()

    def mouseReleaseEvent(self, ev):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            rects = self._btn_rects()
            pos = ev.position()
            for key in ("min", "max", "close"):
                if rects[key].contains(pos):
                    return
            if self._win.isMaximized():
                self._win.showNormal()
            else:
                self._win.showMaximized()

    def leaveEvent(self, _ev):
        if self._hover_btn is not None:
            self._hover_btn = None
            self.update()


class MSLesionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("MSLesionTool")
        self.resize(1600, 900)

        self._qsettings = QSettings("MSLesionTool", "MSLesionTool")
        self._device = detect_best_device()

        # ── volume data ──
        self._flair_vol = None   # (D, H, W) numpy
        self._t1_vol = None
        self._t2_vol = None
        self._mask_vol = None    # ground truth if available
        self._spacing = (1.0, 1.0, 1.0)
        self._sitk_ref = None    # reference sitk image for export
        self._patient_folder = None

        # ── segmentation results ──
        self._seg_vol = None     # (D, H, W) uint8 label volume
        self._prob_vol = None    # (C, D, H, W) float32 probability volume
        self._fold_probs = {}    # {(arch_key, fold): prob_array}
        self._predictors_cache = {k: {} for k in ARCHITECTURES}
        self._seg_thread = None
        self._ensemble_mode = "best2"  # "best2", "all5", "custom"
        self._custom_config = {k: list(DEFAULT_BEST2[k]) for k in ARCHITECTURES}
        self._arch_enabled = {k: True for k in ARCHITECTURES}

        # ── classification ──
        self._class_vol = None   # (D, H, W) uint8 classification labels

        # ── view state ──
        self._display_vol = None  # which modality to display (reference to one of flair/t1/t2)
        self._display_name = "FLAIR"
        # slice indices for each axis: 0=Axial, 1=Sagittal, 2=Coronal
        self._slice_idx = [0, 0, 0]
        self._vol_shape = (0, 0, 0)  # (D, H, W)

        # ── drawing state ──
        self._draw_label = 1
        self._erase_mode = False
        self._brush_size = 3
        self._grow_threshold = 0.30
        self._class_label = 1
        self._undo_stack = collections.deque(maxlen=20)
        self._stroke_undo_pushed = False
        self._last_brush_img = None
        self._preload_thread = None
        self._mesh_thread = None
        self._mesh_pending = False
        self._heat_threshold = 0.05

        self._build_ui()
        self._start_preloading()

    # -- Frameless window edge-resize via mouse events --------------------
    def _edge_zone(self, pos):
        """Return resize edges for a position near the window border."""
        g = _RESIZE_GRIP
        r = self.rect()
        left = pos.x() <= g
        right = pos.x() >= r.width() - g
        top = pos.y() <= g
        bottom = pos.y() >= r.height() - g
        if top and left:     return Qt.Edge.TopEdge | Qt.Edge.LeftEdge
        if top and right:    return Qt.Edge.TopEdge | Qt.Edge.RightEdge
        if bottom and left:  return Qt.Edge.BottomEdge | Qt.Edge.LeftEdge
        if bottom and right: return Qt.Edge.BottomEdge | Qt.Edge.RightEdge
        if left:   return Qt.Edge.LeftEdge
        if right:  return Qt.Edge.RightEdge
        if top:    return Qt.Edge.TopEdge
        if bottom: return Qt.Edge.BottomEdge
        return None

    _EDGE_CURSORS = {
        Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.TopEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.BottomEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.TopEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeBDiagCursor,
        Qt.Edge.BottomEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeBDiagCursor,
    }

    def mouseMoveEvent(self, ev):
        if not (self.windowFlags() & Qt.WindowType.FramelessWindowHint):
            return super().mouseMoveEvent(ev)
        edges = self._edge_zone(ev.pos())
        if edges and edges in self._EDGE_CURSORS:
            self.setCursor(self._EDGE_CURSORS[edges])
        else:
            self.unsetCursor()
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        if not (self.windowFlags() & Qt.WindowType.FramelessWindowHint):
            return super().mousePressEvent(ev)
        if ev.button() == Qt.MouseButton.LeftButton:
            edges = self._edge_zone(ev.pos())
            if edges:
                if hasattr(self.windowHandle(), 'startSystemResize'):
                    self.windowHandle().startSystemResize(edges)
                    return
        super().mousePressEvent(ev)

    def _on_device_changed(self, new_device):
        """Handle device dropdown change — clear cached models and re-preload."""
        old = self._device
        self._device = new_device
        if old == new_device:
            return
        _logger.info("Device changed: %s → %s, clearing predictor cache", old, new_device)
        # Clear all cached predictors (they're on the wrong device)
        for arch_key in self._predictors_cache:
            self._predictors_cache[arch_key].clear()
        # Re-preload on the new device
        self._start_preloading()

    def _start_preloading(self):
        """Silently preload nnUNet models in the background."""
        ensemble_config = self._get_ensemble_config()
        if not ensemble_config:
            return
        # Only preload architectures that have model dirs present
        valid_config = {}
        for arch_key, folds in ensemble_config.items():
            if _resolve_model_dir(ARCHITECTURES[arch_key][0]) is not None:
                valid_config[arch_key] = folds
        if not valid_config:
            return
        self._preload_thread = PreloadModelThread(
            valid_config, self._device,
            predictors_cache=self._predictors_cache, parent=self)
        self._preload_thread.progress.connect(lambda msg: self._status.showMessage(msg))
        self._preload_thread.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(1, 0, 1, 1)
        root.setSpacing(2)

        # ── Custom title bar ──
        self._title_bar = TitleBar(self)
        root.addWidget(self._title_bar)

        # ── Top bar ──
        top = QHBoxLayout()

        btn_open = QPushButton("📂 Open Patient")
        btn_open.setObjectName("btn_open")
        btn_open.setToolTip("Select a folder or files containing brain MRI (FLAIR + T1)")
        btn_open.clicked.connect(self._open_patient)
        top.addWidget(btn_open)

        self._lbl_patient = QLabel("(no patient loaded)")
        self._lbl_patient.setObjectName("lbl_patient")
        top.addWidget(self._lbl_patient, 1)

        top.addWidget(QLabel("Display:"))
        self._combo_display = QComboBox()
        self._combo_display.addItems(["FLAIR", "T1", "T2"])
        self._combo_display.currentTextChanged.connect(self._on_display_changed)
        top.addWidget(self._combo_display)

        top.addWidget(QLabel("Device:"))
        self._combo_device = QComboBox()
        devs = ["cpu"]
        if _HAS_TORCH and torch.cuda.is_available():
            devs.insert(0, "cuda")
        if _HAS_ORT:
            ort_provs = ort.get_available_providers()
            if 'CUDAExecutionProvider' in ort_provs:
                devs.append("onnx-cuda")
            if 'TensorrtExecutionProvider' in ort_provs:
                devs.append("onnx-trt")
            devs.append("onnx-cpu")
        self._combo_device.addItems(devs)
        self._combo_device.setCurrentText(self._device)
        self._combo_device.currentTextChanged.connect(self._on_device_changed)
        top.addWidget(self._combo_device)

        top.addWidget(QLabel("Ensemble:"))
        self._combo_ensemble = QComboBox()
        self._combo_ensemble.addItems(["Best-2/arch (Recommended)", "All 15 folds", "Custom..."])
        self._combo_ensemble.currentIndexChanged.connect(self._on_ensemble_mode_changed)
        top.addWidget(self._combo_ensemble)

        self._lbl_ensemble_info = QLabel("")
        self._lbl_ensemble_info.setObjectName("lbl_ensemble_info")
        top.addWidget(self._lbl_ensemble_info)
        self._update_ensemble_info()

        self._chk_tta = QCheckBox("TTA")
        self._chk_tta.setChecked(True)
        self._chk_tta.setToolTip("Test-Time Augmentation: mirrors input along all axes\n"
                                 "and averages predictions. +0.2-0.8pp Dice, ~8x slower.")
        top.addWidget(self._chk_tta)

        self._chk_anim = QCheckBox("Scan FX")
        self._chk_anim.setChecked(True)
        self._chk_anim.setToolTip("Toggle scanning animation during segmentation")
        top.addWidget(self._chk_anim)

        root.addLayout(top)

        # ── 2x2 viewer grid ──
        view_grid = QGridLayout()
        view_grid.setSpacing(1)

        self._canvases = []  # [axial, sagittal, coronal]
        self._sliders = []
        self._slice_labels = []
        self._view_boxes = []

        for vi, vname in enumerate(VIEWS):
            box = QGroupBox(vname)
            self._view_boxes.append(box)
            vlay = QVBoxLayout(box)
            vlay.setContentsMargins(2, 14, 2, 2)

            canvas = SliceCanvas(title=vname)
            canvas.doubleClicked.connect(lambda v=vi: self._toggle_fullscreen(v))
            canvas.scrollRequested.connect(lambda d, v=vi: self._on_scroll(v, d))
            canvas.crosshairMoved.connect(lambda x, y, v=vi: self._on_crosshair(v, x, y))
            canvas.brushStroke.connect(lambda x, y, f, v=vi: self._on_brush(v, x, y, f))
            canvas.brushReleased.connect(self._on_brush_released)
            canvas.lesionPicked.connect(lambda x, y, v=vi: self._on_lesion_pick(v, x, y))
            self._canvases.append(canvas)
            vlay.addWidget(canvas, 1)

            sl_row = QHBoxLayout()
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setMinimum(0); sl.setMaximum(0)
            sl.valueChanged.connect(lambda val, v=vi: self._on_slider(v, val))
            self._sliders.append(sl)
            sl_row.addWidget(sl, 1)
            lbl = QLabel("0/0")
            self._slice_labels.append(lbl)
            sl_row.addWidget(lbl)
            vlay.addLayout(sl_row)

            row, col = vi // 2, vi % 2
            view_grid.addWidget(box, row, col)

        # ── 3D view placeholder ──
        box3d = QGroupBox("3D View")
        self._view_boxes.append(box3d)
        lay3d = QVBoxLayout(box3d)
        
        if _HAS_PG:
            self._gl_view = FullScreenGLView()
            self._gl_view.doubleClicked.connect(lambda: self._toggle_fullscreen(3))
            self._gl_view.setBackgroundColor((4, 8, 16))
            self._gl_view.setCameraPosition(distance=300)
            
            # Dummy triangle wrapped explicitly in MeshData to prevent internal PyQtGraph NoneType crash
            d_verts = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
            d_faces = np.array([[0,1,2]], dtype=int)
            md = gl.MeshData(vertexes=d_verts, faces=d_faces)
            self._mesh_item = gl.GLMeshItem(meshdata=md, smooth=True, color=(1.0, 0.3, 0.3, 0.0), shader='shaded')
            self._gl_view.addItem(self._mesh_item)
            
            self._brain_mesh_item = gl.GLMeshItem(
                meshdata=md, smooth=True, 
                color=(0.7, 0.7, 0.7, 0.04), 
                shader='shaded', glOptions='translucent'
            )
            self._gl_view.addItem(self._brain_mesh_item)

            self._gl_images = [UnlitGLImageItem(np.zeros((1, 1, 4), dtype=np.uint8)) for _ in range(3)]
            for gi in self._gl_images:
                self._gl_view.addItem(gi)
                
            lay3d.addWidget(self._gl_view, 1)
        else:
            self._gl_view = None

        self._lbl_3d = QLabel("Loading 3D...") if _HAS_PG else QLabel("3D rendering\n(pyqtgraph not available)")
        self._lbl_3d.setObjectName("lbl_3d")
        self._lbl_3d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay3d.addWidget(self._lbl_3d)
        view_grid.addWidget(box3d, 1, 1)

        root.addLayout(view_grid, 1)

        # ── Bottom controls ──
        bot = QHBoxLayout()

        self._btn_seg = QPushButton("▶  Run Segmentation")
        self._btn_seg.setObjectName("btn_seg")
        self._btn_seg.setEnabled(False)
        self._btn_seg.clicked.connect(self._run_segmentation)
        bot.addWidget(self._btn_seg)

        self._btn_tools = QPushButton("\u270f Tools")
        self._btn_tools.setObjectName("btn_tools")
        self._btn_tools.clicked.connect(self._toggle_toolbar)
        bot.addWidget(self._btn_tools)

        bot.addWidget(QLabel("Overlay Type:"))
        self._combo_overlay = QComboBox()
        self._combo_overlay.addItems(["Mask", "Probability Heatmap"])
        self._combo_overlay.currentTextChanged.connect(lambda _: self._update_all_views())
        bot.addWidget(self._combo_overlay)

        bot.addWidget(QLabel("Heat Thresh:"))
        self._lbl_thresh_val = QLabel("5%")
        self._lbl_thresh_val.setFixedWidth(35)
        self._lbl_thresh_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bot.addWidget(self._lbl_thresh_val)
        
        self._sl_thresh = QSlider(Qt.Orientation.Horizontal)
        self._sl_thresh.setRange(0, 100); self._sl_thresh.setValue(5)
        self._sl_thresh.setFixedWidth(80)
        self._sl_thresh.valueChanged.connect(self._on_thresh)
        self._sl_thresh.setToolTip("Probability threshold for heatmap rendering")
        bot.addWidget(self._sl_thresh)

        bot.addWidget(QLabel("Alpha:"))
        self._sl_alpha = QSlider(Qt.Orientation.Horizontal)
        self._sl_alpha.setRange(0, 100); self._sl_alpha.setValue(45)
        self._sl_alpha.setFixedWidth(80)
        self._sl_alpha.valueChanged.connect(self._on_alpha)
        bot.addWidget(self._sl_alpha)

        bot.addWidget(QLabel("3D Brain:"))
        self._sl_brain_alpha = QSlider(Qt.Orientation.Horizontal)
        self._sl_brain_alpha.setRange(0, 100); self._sl_brain_alpha.setValue(4)
        self._sl_brain_alpha.setFixedWidth(80)
        self._sl_brain_alpha.valueChanged.connect(self._on_brain_alpha)
        bot.addWidget(self._sl_brain_alpha)

        self._btn_save = QPushButton("\U0001f4be Save Results")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_results)
        bot.addWidget(self._btn_save)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setObjectName("btn_reset")
        self._btn_reset.clicked.connect(self._reset)
        bot.addWidget(self._btn_reset)

        bot.addStretch()
        self._lbl_ai = QLabel("")
        self._lbl_ai.setObjectName("lbl_ai")
        self._update_ai_label()
        bot.addWidget(self._lbl_ai)

        root.addLayout(bot)

        # ── Legend ──
        legend = QHBoxLayout()
        legend.addWidget(QLabel("Seg:"))
        for lid, (name, color) in SEG_LABELS.items():
            if lid == 0: continue
            sq = QLabel("  "); sq.setFixedSize(12, 12); sq.setAutoFillBackground(True)
            pal = sq.palette(); pal.setColor(QPalette.ColorRole.Window, QColor(*color))
            sq.setPalette(pal)
            legend.addWidget(sq); legend.addWidget(QLabel(name))
        legend.addWidget(QLabel("  |  Class:"))
        for cid in range(1, 6):
            name, color = CLASS_LABELS[cid]
            sq = QLabel("  "); sq.setFixedSize(12, 12); sq.setAutoFillBackground(True)
            pal = sq.palette(); pal.setColor(QPalette.ColorRole.Window, QColor(*color))
            sq.setPalette(pal)
            legend.addWidget(sq); legend.addWidget(QLabel(name))
        legend.addStretch()
        root.addLayout(legend)

        # ── Stats label ──
        self._lbl_stats = QLabel("")
        self._lbl_stats.setObjectName("lbl_stats")
        root.addWidget(self._lbl_stats)

        # ── Bottom progress bar ──
        prog_row = QHBoxLayout()
        prog_row.setContentsMargins(4, 0, 4, 2)
        self._lbl_progress_pct = QLabel("")
        self._lbl_progress_pct.setObjectName("lbl_progress_pct")
        self._lbl_progress_pct.setFixedWidth(40)
        prog_row.addWidget(self._lbl_progress_pct)
        self._pbar = QProgressBar()
        self._pbar.setObjectName("main_pbar")
        self._pbar.setRange(0, 100)
        self._pbar.setValue(0)
        self._pbar.setTextVisible(False)
        self._pbar.setFixedHeight(6)
        self._pbar.hide()
        prog_row.addWidget(self._pbar, 1)
        self._lbl_progress_pct.hide()
        root.addLayout(prog_row)

        # ── Scan animation ──
        self._scan_animator = ScanAnimator(self._canvases, self)
        self._scan_animator.finished.connect(self._on_scan_animation_done)
        self._stats_animator = StatsAnimator(self._lbl_stats, self._lbl_3d, self)

        # ── Toolbar ──
        self._toolbar = DrawingToolbar(self)
        self._toolbar.drawingToggled.connect(self._on_drawing_toggled)
        self._toolbar.labelChanged.connect(lambda l: setattr(self, '_draw_label', l))
        self._toolbar.brushSizeChanged.connect(self._on_brush_size_changed)
        self._toolbar.eraserToggled.connect(lambda e: setattr(self, '_erase_mode', e))
        self._toolbar.growModeToggled.connect(self._on_grow_toggled)
        self._toolbar.growThresholdChanged.connect(lambda v: setattr(self, '_grow_threshold', v / 100.0))
        self._toolbar.classifyModeToggled.connect(self._on_classify_toggled)
        self._toolbar.classLabelChanged.connect(lambda c: setattr(self, '_class_label', c))
        self._toolbar.hide()

        # ── Shortcuts ──
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("B"), self, lambda: self._toolbar.select_brush())
        QShortcut(QKeySequence("E"), self, lambda: self._toolbar.select_eraser())
        QShortcut(QKeySequence("["), self, lambda: self._toolbar.set_brush_size(max(1, self._brush_size - 1)))
        QShortcut(QKeySequence("]"), self, lambda: self._toolbar.set_brush_size(min(30, self._brush_size + 1)))

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Load a patient folder to begin.")

    def _update_ai_label(self):
        if _HAS_NNUNET and _HAS_TORCH:
            ort_str = " + ORT" if _HAS_ORT else ""
            self._lbl_ai.setText(f"AI: nnUNet ready ({self._device}){ort_str}")
            self._lbl_ai.setProperty("aiStatus", "ready")
        elif _HAS_ORT:
            self._lbl_ai.setText(f"AI: ONNX Runtime ready")
            self._lbl_ai.setProperty("aiStatus", "ort_ready")
        else:
            self._lbl_ai.setText("AI: unavailable (install torch + nnunetv2)")
            self._lbl_ai.setProperty("aiStatus", "unavailable")
        self._lbl_ai.style().unpolish(self._lbl_ai)
        self._lbl_ai.style().polish(self._lbl_ai)

    # ── Data loading ─────────────────────────────────────────────────
    def _open_patient(self):
        """Pick FLAIR file, then T1 file. T2/Mask auto-detected from same folder."""
        nii_filter = "NIfTI Files (*.nii *.nii.gz);;All Files (*)"

        flair_path, _ = QFileDialog.getOpenFileName(
            self, "Select FLAIR Volume", "", nii_filter)
        if not flair_path:
            return

        t1_path, _ = QFileDialog.getOpenFileName(
            self, "Select T1 Volume", os.path.dirname(flair_path), nii_filter)
        if not t1_path:
            return

        files = {"FLAIR": flair_path, "T1": t1_path, "T2": None, "MASK": None}

        # Auto-detect T2 and ground truth from the same folder
        folder = os.path.dirname(flair_path)
        selected = {os.path.normpath(flair_path), os.path.normpath(t1_path)}
        for nifti in find_nifti_files_recursive(folder):
            if os.path.normpath(nifti) in selected:
                continue
            up = os.path.basename(nifti).upper()
            if files["T2"] is None and "T2" in up:
                files["T2"] = nifti
            elif files["MASK"] is None and any(k in up for k in ("MASK", "SEG", "LABEL", "WMH")):
                files["MASK"] = nifti

        self._load_from_files(files, folder)

    def _load_from_files(self, files, patient_folder):
        self._patient_folder = patient_folder
        errors = []
        _logger.info("Loading patient from: %s", patient_folder)
        _logger.info("Files: %s", {k: v for k, v in files.items() if v})

        if files["FLAIR"]:
            try:
                _logger.info("Loading FLAIR: %s", files["FLAIR"])
                arr, aff, sp, ref = load_nifti(files["FLAIR"])
                self._flair_vol = arr.astype(np.float32)
                self._spacing = sp
                self._sitk_ref = ref
                _logger.info("FLAIR loaded: shape=%s spacing=%s", arr.shape, sp)
                self._status.showMessage(f"FLAIR: {arr.shape}")
                QApplication.processEvents()
            except Exception as e:
                errors.append(f"FLAIR: {e}")
                _logger.error("Failed to load FLAIR: %s", e, exc_info=True)

        if files["T1"]:
            try:
                _logger.info("Loading T1: %s", files["T1"])
                arr, _, _, _ = load_nifti(files["T1"])
                self._t1_vol = arr.astype(np.float32)
                _logger.info("T1 loaded: shape=%s", arr.shape)
            except Exception as e:
                errors.append(f"T1: {e}")
                _logger.error("Failed to load T1: %s", e, exc_info=True)

        if files.get("T2"):
            try:
                arr, _, _, _ = load_nifti(files["T2"])
                self._t2_vol = arr.astype(np.float32)
            except Exception as e:
                errors.append(f"T2: {e}")

        if files.get("MASK"):
            try:
                arr, _, _, _ = load_nifti(files["MASK"])
                self._mask_vol = arr.astype(np.uint8)
            except Exception as e:
                errors.append(f"MASK: {e}")

        if errors:
            QMessageBox.warning(self, "Load Warning",
                                "Some files failed to load:\n" + "\n".join(errors))

        if self._flair_vol is None:
            QMessageBox.critical(self, "Load Error", "FLAIR volume could not be loaded.")
            return

        # Set display volume
        self._display_vol = self._flair_vol
        self._display_name = "FLAIR"
        self._combo_display.setCurrentText("FLAIR")
        # Compute global windowing (percentile-based for robustness to outliers)
        p1, p99 = np.percentile(self._display_vol[self._display_vol > 0], [1, 99]) if np.any(self._display_vol > 0) else (float(self._display_vol.min()), float(self._display_vol.max()))
        self._global_wc = (p1 + p99) / 2.0
        self._global_ww = p99 - p1

        # Determine volume shape from first available
        ref_vol = self._flair_vol if self._flair_vol is not None else (
            self._t1_vol if self._t1_vol is not None else self._t2_vol)
        if ref_vol is None:
            return
        self._vol_shape = ref_vol.shape  # (D, H, W)

        # Init segmentation and classification volumes
        self._seg_vol = None
        self._prob_vol = None
        self._class_vol = None
        self._fold_probs = {}

        # Setup sliders
        D, H, W = self._vol_shape
        _logger.info("Volume shape: D=%d H=%d W=%d", D, H, W)
        maxes = [D, W, H]  # Axial=D, Sagittal=W, Coronal=H
        for vi in range(3):
            self._sliders[vi].setMaximum(max(0, maxes[vi] - 1))
            self._slice_idx[vi] = maxes[vi] // 2
            self._sliders[vi].blockSignals(True)
            self._sliders[vi].setValue(self._slice_idx[vi])
            self._sliders[vi].blockSignals(False)

        # Set pixel aspect ratios from voxel spacing
        # spacing = (x, y, z) from SimpleITK; array is (z, y, x)
        sx, sy, sz = self._spacing
        self._canvases[0].set_aspect(sx, sy)  # Axial:    cols=W(x), rows=H(y)
        self._canvases[1].set_aspect(sy, sz)  # Sagittal: cols=H(y), rows=D(z)
        self._canvases[2].set_aspect(sx, sz)  # Coronal:  cols=W(x), rows=D(z)

        _logger.info("Updating all views...")
        self._update_all_views()
        self._lbl_patient.setText(f"Patient: {os.path.basename(patient_folder)}  |  "
                                  f"Shape: {D}×{H}×{W}  |  "
                                  f"Spacing: {self._spacing[0]:.2f}×{self._spacing[1]:.2f}×{self._spacing[2]:.2f}")

        # Check if segmentation can run
        has_ai = (_HAS_NNUNET and _HAS_TORCH) or _HAS_ORT
        ready = (self._flair_vol is not None and self._t1_vol is not None and has_ai)
        self._btn_seg.setEnabled(ready)
        self._status.showMessage(f"Loaded patient from: {os.path.basename(patient_folder)}")
        
        # Build Contextual Brain Mesh
        if getattr(self, '_gl_view', None) is not None:
            # Pin the exact physical center of rotation for the orbit camera!
            D, H, W = self._vol_shape
            # spacing=(sx,sy,sz)=(x,y,z); array=(D,H,W)=(z,y,x)
            sx, sy, sz = self._spacing
            # Physical extents: D*sz, H*sy, W*sx
            cx, cy, cz = (W * sx) / 2, (H * sy) / 2, (D * sz) / 2

            # After [2,1,0] vertex reorder: GL_X=x(sx), GL_Y=y(sy), GL_Z=z(sz)
            self._gl_view.opts['center'] = pg.Vector(cx, cy, cz)
            self._gl_view.setCameraPosition(distance=max(cx, cy, cz) * 3.5)
            self._mesh_centered = True
            try:
                import skimage.measure
                mask = self._flair_vol > (np.max(self._flair_vol) * 0.15)
                ds = 4
                verts, faces, _, _ = skimage.measure.marching_cubes(
                    mask[::ds, ::ds, ::ds], level=0.5,
                    spacing=(sz*ds, sy*ds, sx*ds))  # (z,y,x) to match array axes
                
                # Rotate the mesh array so Axial maps natively to Z!
                verts = verts[:, [2, 1, 0]]
                
                md = gl.MeshData(vertexes=verts, faces=faces)
                self._brain_mesh_item.setMeshData(meshdata=md)
            except Exception as e:
                _logger.warning("Brain mesh generation failed: %s", e)

    def _on_display_changed(self, name):
        self._display_name = name
        if name == "FLAIR":
            self._display_vol = self._flair_vol
        elif name == "T1":
            self._display_vol = self._t1_vol
        elif name == "T2":
            self._display_vol = self._t2_vol
        # Recompute global windowing for the new volume (percentile-based)
        if self._display_vol is not None:
            nz = self._display_vol[self._display_vol > 0]
            if nz.size > 0:
                p1, p99 = np.percentile(nz, [1, 99])
            else:
                p1, p99 = float(self._display_vol.min()), float(self._display_vol.max())
            self._global_wc = (p1 + p99) / 2.0
            self._global_ww = p99 - p1
        else:
            self._global_wc = self._global_ww = None
        self._update_all_views()

    # ── View updates ─────────────────────────────────────────────────
    def _update_all_views(self):
        for vi in range(3):
            self._update_view(vi)

    def _update_view(self, view_idx):
        axis = view_idx
        idx = self._slice_idx[view_idx]
        vol = self._display_vol

        # Extract and display slice
        sl = extract_slice(vol, axis, idx)
        ww = getattr(self, '_global_ww', None)
        wc = getattr(self, '_global_wc', None)
        qimg = volume_to_qimage(sl, ww=ww, wc=wc) if sl is not None else None
        D, H, W = self._vol_shape if self._vol_shape[0] > 0 else (1, 1, 1)
        maxes = [D, W, H]
        title = f"{VIEWS[view_idx]}  {idx + 1}/{maxes[view_idx]}  ({self._display_name})"
        self._canvases[view_idx].set_image(qimg, title)
        self._slice_labels[view_idx].setText(f"{idx + 1}/{maxes[view_idx]}")

        # Base / Overlay rendering
        overlay_type = getattr(self, "_combo_overlay", None)
        is_heatmap = (overlay_type is not None and overlay_type.currentText() == "Probability Heatmap")

        if is_heatmap and self._prob_vol is not None:
            # _prob_vol is (2, D, H, W) where 1 is the lesion class
            prob_vol_1 = self._prob_vol[1]
            prob_sl = extract_slice(prob_vol_1, axis, idx)
            if prob_sl is not None:
                rgba = prob_to_heatmap_rgba(prob_sl, threshold=self._heat_threshold)
                self._canvases[view_idx].set_overlay(rgba)
            else:
                self._canvases[view_idx].set_overlay(None)
        elif not is_heatmap and self._seg_vol is not None:
            seg_sl = extract_slice(self._seg_vol, axis, idx)
            if seg_sl is not None:
                rgba = labels_to_rgba(seg_sl)
                self._canvases[view_idx].set_overlay(rgba)
            else:
                self._canvases[view_idx].set_overlay(None)
        else:
            self._canvases[view_idx].set_overlay(None)

        # Classification overlay
        if self._class_vol is not None:
            cls_sl = extract_slice(self._class_vol, axis, idx)
            if cls_sl is not None:
                rgba = classification_to_rgba(cls_sl)
                self._canvases[view_idx].set_class_overlay(rgba)
            else:
                self._canvases[view_idx].set_class_overlay(None)
        else:
            self._canvases[view_idx].set_class_overlay(None)

        # Crosshairs
        self._update_crosshairs()
        try:
            self._update_gl_planes(view_idx)
        except Exception as e:
            _logger.warning("GL planes update failed for view %d: %s", view_idx, e)

    def _update_gl_planes(self, view_idx):
        if not getattr(self, '_gl_images', None) or self._display_vol is None:
            return
        ix = self._slice_idx[view_idx]
        vol = self._display_vol
        
        # Grab raw unflipped arrays so OpenGL maps match walking coords identically
        if view_idx == 0:
            sl = vol[ix, :, :]
        elif view_idx == 1:
            sl = vol[:, :, ix]
        else:
            sl = vol[:, ix, :]

        # Build dynamically transparent grayscale plane using global windowing
        ww = getattr(self, '_global_ww', None)
        wc = getattr(self, '_global_wc', None)
        sl_min, sl_max = float(np.min(sl)), float(np.max(sl))
        if ww is not None and wc is not None and ww > 0:
            sl_clip = np.clip(sl, wc - ww / 2, wc + ww / 2)
            v = (sl_clip - (wc - ww / 2)) / ww
        else:
            v = sl if sl_max == sl_min else (sl - sl_min) / (sl_max - sl_min)
        rgba = np.empty((sl.shape[0], sl.shape[1], 4), dtype=np.uint8)
        val255 = (v * 255).astype(np.uint8)
        rgba[..., 0] = val255
        rgba[..., 1] = val255
        rgba[..., 2] = val255
        
        # Dynamically map absolute low-level scanner background noise completely to transparency
        # so the gray bounding boxes don't project past the actual skull!
        alpha_map = np.full(sl.shape, 90, dtype=np.uint8)
        alpha_map[sl < (sl_max * 0.05)] = 0  # pure empty void clipping
        rgba[..., 3] = alpha_map
        self._gl_images[view_idx].setData(rgba)

        # Morph exact 3D positioning using GL coords [X=x(sx), Y=y(sy), Z=z(sz)]
        import PyQt6.QtGui as QtGui
        tr = QtGui.QMatrix4x4()
        sx, sy, sz = float(self._spacing[0]), float(self._spacing[1]), float(self._spacing[2])

        # GLImageItem maps from [0,0] to [cols, rows] of the slice.
        # After [2,1,0] vert reorder: GL_X=x(W), GL_Y=y(H), GL_Z=z(D)
        if view_idx == 0:    # Axial at slice ix in D-axis → translate along GL_Z
            tr.translate(0.0, 0.0, float(ix * sz))
            # slice shape (H,W): rows=H(y)→GL_Y, cols=W(x)→GL_X
            tr.setColumn(0, QtGui.QVector4D(0.0, sy, 0.0, 0.0))  # row→GL_Y
            tr.setColumn(1, QtGui.QVector4D(sx, 0.0, 0.0, 0.0))  # col→GL_X
        elif view_idx == 1:  # Sagittal at slice ix in W-axis → translate along GL_X
            tr.translate(float(ix * sx), 0.0, 0.0)
            # slice shape (D,H): rows=D(z)→GL_Z, cols=H(y)→GL_Y
            tr.setColumn(0, QtGui.QVector4D(0.0, 0.0, sz, 0.0))  # row→GL_Z
            tr.setColumn(1, QtGui.QVector4D(0.0, sy, 0.0, 0.0))  # col→GL_Y
        elif view_idx == 2:  # Coronal at slice ix in H-axis → translate along GL_Y
            tr.translate(0.0, float(ix * sy), 0.0)
            # slice shape (D,W): rows=D(z)→GL_Z, cols=W(x)→GL_X
            tr.setColumn(0, QtGui.QVector4D(0.0, 0.0, sz, 0.0))  # row→GL_Z
            tr.setColumn(1, QtGui.QVector4D(sx, 0.0, 0.0, 0.0))  # col→GL_X
            
        self._gl_images[view_idx].setTransform(tr)

    def _update_crosshairs(self):
        D, H, W = self._vol_shape if self._vol_shape[0] > 0 else (1, 1, 1)
        ai, si, ci = self._slice_idx  # axial, sagittal, coronal indices

        # Crosshairs must be flipped to match the flipud display
        D, H, W = self._vol_shape
        # Axial view shows (H, W) → crosshair at (sagittal_idx, H-1-coronal_idx)
        self._canvases[0].set_crosshair(si, H - 1 - ci)
        # Sagittal view shows (D, H) → crosshair at (coronal_idx, D-1-axial_idx)
        self._canvases[1].set_crosshair(ci, D - 1 - ai)
        # Coronal view shows (D, W) → crosshair at (sagittal_idx, D-1-axial_idx)
        self._canvases[2].set_crosshair(si, D - 1 - ai)

    # ── Interaction slots ────────────────────────────────────────────
    def _toggle_fullscreen(self, view_idx):
        if getattr(self, '_fullscreen_view_idx', None) is None:
            self._fullscreen_view_idx = view_idx
            for i, box in enumerate(self._view_boxes):
                if i != view_idx:
                    box.hide()
        else:
            self._fullscreen_view_idx = None
            for box in self._view_boxes:
                box.show()

    def _on_scroll(self, view_idx, delta):
        mx = self._sliders[view_idx].maximum()
        v = max(0, min(self._slice_idx[view_idx] + delta, mx))
        self._sliders[view_idx].setValue(v)

    def _on_slider(self, view_idx, val):
        self._slice_idx[view_idx] = val
        self._update_view(view_idx)
        # Update crosshairs in other views
        for vi in range(3):
            if vi != view_idx:
                self._update_view(vi)

    def _unflip_y(self, view_idx, iy):
        """Convert displayed y back to volume y (undo flipud)."""
        sh, sw = slice_dims(self._vol_shape, view_idx)
        return sh - 1 - iy

    def _on_crosshair(self, view_idx, img_x, img_y):
        """User clicked in a view to set crosshair position."""
        ix, iy = int(round(img_x)), int(round(img_y))
        iy = self._unflip_y(view_idx, iy)  # undo display flip
        D, H, W = self._vol_shape if self._vol_shape[0] > 0 else (1, 1, 1)

        if view_idx == 0:  # Axial: (H, W) → x=sagittal, y=coronal
            self._slice_idx[1] = max(0, min(ix, W - 1))
            self._slice_idx[2] = max(0, min(iy, H - 1))
        elif view_idx == 1:  # Sagittal: (D, H) → x=coronal, y=axial
            self._slice_idx[2] = max(0, min(ix, H - 1))
            self._slice_idx[0] = max(0, min(iy, D - 1))
        elif view_idx == 2:  # Coronal: (D, W) → x=sagittal, y=axial
            self._slice_idx[1] = max(0, min(ix, W - 1))
            self._slice_idx[0] = max(0, min(iy, D - 1))

        for vi in range(3):
            self._sliders[vi].blockSignals(True)
            self._sliders[vi].setValue(self._slice_idx[vi])
            self._sliders[vi].blockSignals(False)
        self._update_all_views()

    def _on_thresh(self, v):
        self._heat_threshold = v / 100.0
        if hasattr(self, '_lbl_thresh_val'):
            self._lbl_thresh_val.setText(f"{v}%")
        overlay_type = getattr(self, "_combo_overlay", None)
        if overlay_type is not None and overlay_type.currentText() == "Probability Heatmap":
            self._update_all_views()

    def _on_alpha(self, v):
        a = v / 100.0
        for c in self._canvases:
            c.set_overlay_alpha(a)

    def _on_brain_alpha(self, v):
        if getattr(self, '_brain_mesh_item', None) is not None:
            self._brain_mesh_item.setColor((0.7, 0.7, 0.7, v / 100.0))

    def _on_folds_changed(self):
        if self._fold_probs:
            self._ensemble_predictions()

    def _get_ensemble_config(self):
        """Return {arch_key: (fold_tuple,)} for the current ensemble mode."""
        config = {}
        if self._ensemble_mode == "best2":
            for k in ARCHITECTURES:
                config[k] = DEFAULT_BEST2[k]
        elif self._ensemble_mode == "all5":
            for k in ARCHITECTURES:
                config[k] = (0, 1, 2, 3, 4)
        else:  # custom
            for k in ARCHITECTURES:
                if self._arch_enabled.get(k, True) and self._custom_config.get(k):
                    config[k] = tuple(self._custom_config[k])
        return config

    def _on_ensemble_mode_changed(self, idx):
        modes = ["best2", "all5", "custom"]
        self._ensemble_mode = modes[idx]
        if idx == 2:
            self._show_custom_ensemble_dialog()
        self._update_ensemble_info()
        if self._fold_probs:
            self._ensemble_predictions()

    def _update_ensemble_info(self):
        config = self._get_ensemble_config()
        n_models = sum(len(v) for v in config.values())
        archs = [ARCHITECTURES[k][2] for k in config]
        self._lbl_ensemble_info.setText(f"{n_models} models ({', '.join(archs)})")

    def _show_custom_ensemble_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Custom Ensemble Configuration")
        dlg.setMinimumWidth(450)
        lay = QVBoxLayout(dlg)

        arch_widgets = {}
        for arch_key in ARCHITECTURES:
            _, _, display = ARCHITECTURES[arch_key]
            row = QHBoxLayout()
            chk_arch = QCheckBox(display)
            chk_arch.setChecked(self._arch_enabled.get(arch_key, True))
            row.addWidget(chk_arch)
            fold_checks = []
            for f in range(5):
                fc = QCheckBox(str(f))
                fc.setChecked(f in self._custom_config.get(arch_key, []))
                row.addWidget(fc)
                fold_checks.append(fc)
            row.addStretch()
            lay.addLayout(row)
            arch_widgets[arch_key] = (chk_arch, fold_checks)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            for arch_key, (chk_arch, fold_checks) in arch_widgets.items():
                self._arch_enabled[arch_key] = chk_arch.isChecked()
                self._custom_config[arch_key] = [
                    i for i, fc in enumerate(fold_checks) if fc.isChecked()]
            self._update_ensemble_info()
            if self._fold_probs:
                self._ensemble_predictions()

    # ── Segmentation ─────────────────────────────────────────────────
    def _run_segmentation(self):
        if self._flair_vol is None or self._t1_vol is None:
            QMessageBox.warning(self, "Missing data", "Need FLAIR and T1 volumes.")
            return
        if not (_HAS_NNUNET and _HAS_TORCH) and not _HAS_ORT:
            QMessageBox.warning(self, "AI unavailable", "torch+nnunetv2 or onnxruntime required.")
            return

        ensemble_config = self._get_ensemble_config()
        if not ensemble_config:
            QMessageBox.warning(self, "No models", "Select at least one architecture and fold.")
            return

        # Verify at least one model dir exists
        found_any = False
        for arch_key in ensemble_config:
            if _resolve_model_dir(ARCHITECTURES[arch_key][0]) is not None:
                found_any = True
                break
        if not found_any:
            dirs = [ARCHITECTURES[k][0] for k in ensemble_config]
            QMessageBox.critical(self, "Model not found",
                f"No model directories found.\nLooked for:\n" + "\n".join(dirs))
            return

        total = sum(len(v) for v in ensemble_config.values())
        self._btn_seg.setEnabled(False)

        # Wait for preload thread to finish (avoids GPU contention)
        if self._preload_thread is not None and self._preload_thread.isRunning():
            self._status.showMessage("Waiting for model preloading to finish...")
            QApplication.processEvents()
            self._preload_thread.wait()

        self._status.showMessage(f"Running {total} models on {self._device}...")
        QApplication.processEvents()

        use_tta = self._chk_tta.isChecked()
        self._seg_thread = SegmentationThread(
            ensemble_config, self._device,
            self._flair_vol, self._t1_vol,
            self._spacing, predictors_cache=self._predictors_cache,
            use_mirroring=use_tta, parent=self)
        self._seg_thread.progress.connect(lambda m: self._status.showMessage(m))
        self._seg_thread.progressPercent.connect(self._on_seg_progress)
        self._seg_thread.finished.connect(self._on_seg_finished)
        # Start animation AFTER cuDNN warmup completes (avoids GIL contention
        # that would slow down cuDNN benchmark auto-tuning by 5-10x)
        if self._chk_anim.isChecked():
            self._seg_thread.warmedUp.connect(self._scan_animator.start_scan)
        self._pbar.setValue(0)
        self._lbl_progress_pct.setText("0%")
        self._pbar.show()
        self._lbl_progress_pct.show()
        self._seg_thread.start()

    def _on_seg_progress(self, pct):
        self._pbar.setValue(pct)
        self._lbl_progress_pct.setText(f"{pct}%")

    def _on_seg_finished(self, success, msg):
        self._pbar.hide()
        self._lbl_progress_pct.hide()
        self._btn_seg.setEnabled(True)
        if not success:
            self._scan_animator.stop()
            QMessageBox.critical(self, "Prediction Error", msg)
            self._status.showMessage("Segmentation failed.")
            return
        self._fold_probs = self._seg_thread.fold_probs
        self._device = self._seg_thread.device_used
        self._ensemble_predictions()
        self._btn_save.setEnabled(True)
        self._status.showMessage(msg)
        # Hand off to scan animator for reveal pass (stats shown after animation)
        if self._chk_anim.isChecked():
            self._scan_animator.prediction_done(self._seg_vol)
        else:
            self._update_stats()

    def _on_scan_animation_done(self):
        """Called when reveal sweep finishes — animate stats count-up."""
        if self._seg_vol is not None:
            total = int(np.sum(self._seg_vol > 0))
            sx, sy, sz = self._spacing
            vol_mm3 = total * sx * sy * sz
            self._stats_animator.animate(total, vol_mm3)
        self._update_stats()

    def _ensemble_predictions(self):
        if not self._fold_probs:
            return
        ensemble_config = self._get_ensemble_config()
        prob_stack = []
        for arch_key, folds in ensemble_config.items():
            for fold in folds:
                key = (arch_key, fold)
                if key in self._fold_probs:
                    prob_stack.append(self._fold_probs[key])
        if not prob_stack:
            return
        mean_prob = np.mean(prob_stack, axis=0)  # (C, D, H, W)
        # Safety net: merge multi-class into binary if needed
        if mean_prob.shape[0] > 2:
            bg_prob = mean_prob[0:1]
            fg_prob = np.sum(mean_prob[1:], axis=0, keepdims=True)
            mean_prob = np.concatenate([bg_prob, fg_prob], axis=0)

        self._prob_vol = mean_prob
        self._seg_vol = np.argmax(mean_prob, axis=0).astype(np.uint8)
        self._class_vol = np.zeros_like(self._seg_vol, dtype=np.uint8)
        self._update_all_views()
        self._update_stats()

    def _update_stats(self):
        if self._seg_vol is None:
            self._lbl_stats.setText("")
            self._lbl_3d.setText("No segmentation")
            return
        total_lesion = int(np.sum(self._seg_vol > 0))
        sx, sy, sz = self._spacing
        voxel_vol = sx * sy * sz  # mm³
        total_mm3 = total_lesion * voxel_vol

        stats_parts = [
            f"Total lesion: {total_lesion} vox ({total_mm3:.1f} mm³)"
        ]
        if self._class_vol is not None and np.any(self._class_vol > 0):
            for cid in range(1, 6):
                cnt = int(np.sum(self._class_vol == cid))
                if cnt > 0:
                    stats_parts.append(f"{CLASS_LABELS[cid][0]}: {cnt}vox")
        self._lbl_stats.setText("  |  ".join(stats_parts))

        # Update 3D placeholder with stats
        lines = [f"Lesion Volume: {total_mm3:.1f} mm³",
                 f"Total voxels: {total_lesion}"]
        if self._class_vol is not None:
            for cid in range(1, 6):
                cnt = int(np.sum(self._class_vol == cid))
                if cnt > 0:
                    lines.append(f"{CLASS_LABELS[cid][0]}: {cnt} vox")
        self._lbl_3d.setText("\n".join(lines))
        self._trigger_mesh_build()

    def _trigger_mesh_build(self):
        if not _HAS_PG or self._seg_vol is None:
            return
        if self._mesh_thread is not None and self._mesh_thread.isRunning():
            self._mesh_pending = True
            return
        
        self._mesh_pending = False
        self._mesh_thread = MeshBuilderThread(self._seg_vol > 0, self._class_vol, self._spacing, parent=self)
        self._mesh_thread.finished.connect(self._on_mesh_built)
        self._mesh_thread.start()

    def _on_mesh_built(self, verts, faces, colors):
        if verts is not None and faces is not None and self._gl_view is not None:
            self._mesh_item.show()
            md = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
            self._mesh_item.setMeshData(meshdata=md)
            self._lbl_3d.hide()
        elif self._gl_view is not None:
            self._mesh_item.hide()
            d_verts = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
            d_faces = np.array([[0,1,2]], dtype=int)
            md = gl.MeshData(vertexes=d_verts, faces=d_faces)
            self._mesh_item.setMeshData(meshdata=md)
            if self._seg_vol is None or np.sum(self._seg_vol) == 0:
                self._lbl_3d.setText("No lesions to render in 3D")
                self._lbl_3d.show()

        if self._mesh_pending:
            self._trigger_mesh_build()

    # ── Tool modes ───────────────────────────────────────────────────
    def _toggle_toolbar(self):
        if self._toolbar.isVisible():
            self._toolbar.hide()
        else:
            geo = self.geometry()
            self._toolbar.move(geo.right() - self._toolbar.width() - 20, geo.top() + 60)
            self._toolbar.show(); self._toolbar.raise_()

    def _on_drawing_toggled(self, enabled):
        if enabled:
            self._toolbar.set_grow_enabled(False)
            self._toolbar.set_classify_enabled(False)
        for c in self._canvases:
            c._draw_mode = enabled
            c._grow_mode = False
            c._classify_mode = False
            c.update()

    def _on_grow_toggled(self, enabled):
        if enabled:
            self._toolbar.set_drawing_enabled(False)
            self._toolbar.set_classify_enabled(False)
        for c in self._canvases:
            c._grow_mode = enabled
            c._draw_mode = False
            c._classify_mode = False
            c.update()

    def _on_classify_toggled(self, enabled):
        if enabled:
            self._toolbar.set_drawing_enabled(False)
            self._toolbar.set_grow_enabled(False)
        for c in self._canvases:
            c._classify_mode = enabled
            c._draw_mode = False
            c._grow_mode = False
            c.update()

    def _on_brush_size_changed(self, v):
        self._brush_size = v
        for c in self._canvases:
            c._brush_size = v; c.update()

    # ── Drawing logic ────────────────────────────────────────────────
    def _on_brush(self, view_idx, img_x, img_y, is_first):
        if self._seg_vol is None:
            return
        ix, iy = int(round(img_x)), int(round(img_y))
        if is_first or not self._stroke_undo_pushed:
            self._undo_stack.append(self._seg_vol.copy())
            self._stroke_undo_pushed = True
            self._last_brush_img = None

        if self._last_brush_img is not None:
            lx, ly = self._last_brush_img
            for px, py in _bresenham(lx, ly, ix, iy):
                self._paint_circle_3d(view_idx, px, py)
        else:
            self._paint_circle_3d(view_idx, ix, iy)
        self._last_brush_img = (ix, iy)
        self._update_view(view_idx)

    def _paint_circle_3d(self, view_idx, cx, cy):
        """Paint a circle on the segmentation volume in the given view plane."""
        r = self._brush_size
        label = 0 if self._erase_mode else self._draw_label
        axis = view_idx
        idx = self._slice_idx[view_idx]

        # Un-flip y for volume write
        cy = self._unflip_y(view_idx, cy)

        sh, sw = slice_dims(self._vol_shape, axis)
        y0, y1 = max(0, cy - r), min(sh, cy + r + 1)
        x0, x1 = max(0, cx - r), min(sw, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r

        # Write back into 3D volume
        if axis == 0:
            self._seg_vol[idx, y0:y1, x0:x1][mask] = label
        elif axis == 1:
            self._seg_vol[y0:y1, x0:x1, idx][mask] = label
        elif axis == 2:
            self._seg_vol[y0:y1, idx, x0:x1][mask] = label

    def _on_brush_released(self):
        self._stroke_undo_pushed = False
        self._last_brush_img = None

    # ── Lesion growth ────────────────────────────────────────────────
    def _on_lesion_pick(self, view_idx, img_x, img_y):
        ix, iy = int(round(img_x)), int(round(img_y))

        # Determine which mode
        if self._canvases[view_idx]._grow_mode:
            self._grow_lesion(view_idx, ix, iy)
        elif self._canvases[view_idx]._classify_mode:
            self._classify_lesion(view_idx, ix, iy)

    def _grow_lesion(self, view_idx, ix, iy):
        """Grow a lesion from a clicked point using probability maps + FLAIR intensity."""
        if self._prob_vol is None:
            self._status.showMessage("Run segmentation first to get probability maps.")
            return
        if self._seg_vol is None:
            return

        # Map 2D click to 3D voxel
        axis = view_idx
        slice_idx = self._slice_idx[view_idx]
        D, H, W = self._vol_shape

        # Un-flip y for volume coords
        iy = self._unflip_y(view_idx, iy)

        if axis == 0:  # Axial (H, W)
            vz, vy, vx = slice_idx, iy, ix
        elif axis == 1:  # Sagittal (D, H)
            vz, vy, vx = iy, ix, slice_idx
        elif axis == 2:  # Coronal (D, W)
            vz, vy, vx = iy, slice_idx, ix

        if not (0 <= vz < D and 0 <= vy < H and 0 <= vx < W):
            return

        # Save undo
        self._undo_stack.append(self._seg_vol.copy())

        # Probability-based growth: max lesion prob at each voxel
        lesion_prob = np.max(self._prob_vol[1:], axis=0)  # (D, H, W) max over lesion classes
        threshold = self._grow_threshold

        # Region growing: seed from clicked point
        prob_mask = lesion_prob >= threshold

        # To ensure the click always connects to the lesion even if the exact voxel
        # is slightly below threshold (edge click), we force the immediate 3x3x3 
        # neighbourhood around the click to True.
        seed_mask = np.zeros_like(prob_mask)
        z0, z1 = max(0, vz-1), min(D, vz+2)
        y0, y1 = max(0, vy-1), min(H, vy+2)
        x0, x1 = max(0, vx-1), min(W, vx+2)
        seed_mask[z0:z1, y0:y1, x0:x1] = True
        
        combined = prob_mask | seed_mask

        # Connected component from seed
        if _HAS_SCIPY and ndimage_label is not None:
            # generate_binary_structure(3, 3) allows fully diagonally connected components
            struc = generate_binary_structure(3, 3)
            labeled, n_comp = ndimage_label(combined, structure=struc)
            seed_label = labeled[vz, vy, vx]
            
            if seed_label > 0:
                grown = labeled == seed_label
            else:
                grown = seed_mask
        else:
            grown = seed_mask

        # Assign the best lesion label from probabilities
        best_label = np.argmax(self._prob_vol[:, vz, vy, vx])
        if best_label == 0:
            best_label = 1  # default to small lesion

        added = int(np.sum(grown & (self._seg_vol == 0)))
        self._seg_vol[grown] = best_label
        self._update_all_views()
        self._update_stats()
        self._status.showMessage(f"Lesion growth: added {added} voxels (label {best_label}, threshold {threshold:.2f})")

    # ── Classification ───────────────────────────────────────────────
    def _classify_lesion(self, view_idx, ix, iy):
        """Classify a lesion by clicking on it."""
        if self._seg_vol is None:
            return
        if self._class_vol is None:
            self._class_vol = np.zeros_like(self._seg_vol, dtype=np.uint8)

        axis = view_idx
        slice_idx = self._slice_idx[view_idx]
        D, H, W = self._vol_shape

        # Un-flip y for volume coords
        iy = self._unflip_y(view_idx, iy)

        if axis == 0:
            vz, vy, vx = slice_idx, iy, ix
        elif axis == 1:
            vz, vy, vx = iy, ix, slice_idx
        elif axis == 2:
            vz, vy, vx = iy, slice_idx, ix

        if not (0 <= vz < D and 0 <= vy < H and 0 <= vx < W):
            return

        if self._seg_vol[vz, vy, vx] == 0:
            self._status.showMessage("Click on a lesion to classify it.")
            return

        # Find connected component of this lesion in 3D
        lesion_mask = self._seg_vol > 0
        if _HAS_SCIPY and ndimage_label is not None:
            labeled, n_comp = ndimage_label(lesion_mask)
            comp_id = labeled[vz, vy, vx]
            if comp_id > 0:
                comp_mask = labeled == comp_id
                if self._class_label == 6:
                    self._seg_vol[comp_mask] = 0
                    self._class_vol[comp_mask] = 0
                    nvox = int(np.sum(comp_mask))
                    self._status.showMessage(f"Deleted entire lesion component ({nvox} voxels).")
                    self._trigger_mesh_build()
                else:
                    self._class_vol[comp_mask] = self._class_label
                    nvox = int(np.sum(comp_mask))
                    self._status.showMessage(
                        f"Classified {nvox} voxels as {CLASS_LABELS[self._class_label][0]}")
            else:
                self._status.showMessage("No lesion found at clicked location.")
        else:
            # Fallback: just classify the single voxel
            if self._class_label == 6:
                self._seg_vol[vz, vy, vx] = 0
                self._class_vol[vz, vy, vx] = 0
                self._status.showMessage("Deleted voxel.")
                self._trigger_mesh_build()
            else:
                self._class_vol[vz, vy, vx] = self._class_label
                self._status.showMessage(f"Classified voxel as {CLASS_LABELS[self._class_label][0]}")

        self._update_all_views()
        self._canvases[0].setFocus()

    # ── Actions ─────────────────────────────────────────────────────────
    def _undo(self):
        if not self._undo_stack:
            self._status.showMessage("Nothing to undo.")
            return
        self._seg_vol = self._undo_stack.pop()
        self._update_all_views()
        self._update_stats()
        self._status.showMessage(f"Undo (remaining: {len(self._undo_stack)})")

    # ── Save ─────────────────────────────────────────────────────────
    def _save_results(self):
        if self._seg_vol is None:
            QMessageBox.warning(self, "Nothing to save", "Run segmentation first.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not save_dir:
            return

        patient_name = os.path.basename(self._patient_folder) if self._patient_folder else "unknown"
        out_dir = os.path.join(save_dir, patient_name)
        os.makedirs(out_dir, exist_ok=True)
        saved = []

        if _HAS_SITK:
            def _write_nifti(arr3d, name, ref_img=None):
                img = sitk.GetImageFromArray(arr3d)
                if ref_img is not None and isinstance(ref_img, sitk.Image):
                    img.SetSpacing(ref_img.GetSpacing())
                    img.SetOrigin(ref_img.GetOrigin())
                    img.SetDirection(ref_img.GetDirection())
                else:
                    img.SetSpacing(self._spacing)
                path = os.path.join(out_dir, name)
                sitk.WriteImage(img, path, useCompression=True)
                return name

            saved.append(_write_nifti(self._seg_vol, f"{patient_name}_segmentation.nii.gz", self._sitk_ref))
            if self._class_vol is not None and np.any(self._class_vol > 0):
                saved.append(_write_nifti(self._class_vol, f"{patient_name}_classification.nii.gz", self._sitk_ref))

        # Statistics JSON
        stats = {
            "patient": patient_name,
            "total_lesion_voxels": int(np.sum(self._seg_vol > 0)),
            "voxel_volume_mm3": float(self._spacing[0] * self._spacing[1] * self._spacing[2]),
            "lesion_volume_mm3": float(np.sum(self._seg_vol > 0) * self._spacing[0] * self._spacing[1] * self._spacing[2]),
            "per_label": {},
            "classification": {},
        }
        for lid, (name, _) in SEG_LABELS.items():
            if lid == 0:
                continue
            stats["per_label"][name] = int(np.sum(self._seg_vol == lid))
        if self._class_vol is not None:
            for cid in range(1, 6):
                cnt = int(np.sum(self._class_vol == cid))
                if cnt > 0:
                    stats["classification"][CLASS_LABELS[cid][0]] = cnt

        json_path = os.path.join(out_dir, "statistics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        saved.append("statistics.json")

        self._status.showMessage(f"Saved {len(saved)} file(s) to {out_dir}")
        QMessageBox.information(self, "Save Complete",
            f"Saved {len(saved)} file(s) to:\n{out_dir}")

    # ── Reset ────────────────────────────────────────────────────────
    def _reset(self):
        self._flair_vol = self._t1_vol = self._t2_vol = self._mask_vol = None
        self._seg_vol = self._prob_vol = self._class_vol = None
        self._fold_probs = {}
        self._display_vol = None
        self._vol_shape = (0, 0, 0)
        self._slice_idx = [0, 0, 0]
        self._undo_stack.clear()
        self._patient_folder = None
        for c in self._canvases:
            c.clear()
        for sl in self._sliders:
            sl.setMaximum(0)
        self._lbl_patient.setText("(no patient loaded)")
        self._lbl_stats.setText("")
        self._lbl_3d.setText("3D rendering\n(no data)")
        self._btn_seg.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._toolbar.set_drawing_enabled(False)
        self._toolbar.set_grow_enabled(False)
        self._toolbar.set_classify_enabled(False)
        self._toolbar.hide()
        self._status.showMessage("Reset complete.")

# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
def main():
    global _splash_3d, _splash_start_time
    import multiprocessing
    multiprocessing.freeze_support()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette (fallback for native dialogs)
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(4, 8, 16))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(192, 232, 240))
    pal.setColor(QPalette.ColorRole.Base,            QColor(12, 20, 32))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(10, 14, 24))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(10, 14, 24))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(192, 232, 240))
    pal.setColor(QPalette.ColorRole.Text,            QColor(192, 232, 240))
    pal.setColor(QPalette.ColorRole.Button,          QColor(12, 20, 32))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(192, 232, 240))
    pal.setColor(QPalette.ColorRole.BrightText,      QColor(255, 51, 102))
    pal.setColor(QPalette.ColorRole.Link,            QColor(74, 240, 255))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(74, 240, 255))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(4, 8, 16))
    app.setPalette(pal)

    # Load global QSS theme
    _qss_path = os.path.join(os.path.dirname(__file__), "msseg", "theme.qss")
    if os.path.exists(_qss_path):
        with open(_qss_path) as _f:
            app.setStyleSheet(_f.read())

    _update_splash(98, "BUILDING INTERFACE...")
    win = MSLesionApp()

    screen = app.primaryScreen()
    if screen:
        sg = screen.availableGeometry()
        win.move(sg.center().x() - 800, sg.center().y() - 450)

    # Close splash, then show main window
    if _splash_3d:
        elapsed = time.time() - (_splash_start_time or time.time())
        remaining = max(0, 3.0 - elapsed)
        if remaining > 0:
            _splash_3d.set_progress(100, "READY")
            deadline = time.time() + remaining
            while time.time() < deadline:
                app.processEvents()
                time.sleep(0.02)
        try:
            _splash_3d.finish(win)
        except Exception:
            pass
        _splash_3d = None
        app.processEvents()

    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        crash_log = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), "msseg_crash.log")
        with open(crash_log, "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
