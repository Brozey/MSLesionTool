"""
msseg.mesh – 3D mesh building for lesion and brain surface rendering.
"""
import logging
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from .constants import CLASS_LABELS

_logger = logging.getLogger("MSLesionTool")

try:
    from skimage.measure import marching_cubes
    _HAS_SKM = True
except ImportError:
    marching_cubes = None
    _HAS_SKM = False


class MeshBuilderThread(QThread):
    """Background thread that builds 3D mesh from segmentation mask via marching cubes."""
    finished = pyqtSignal(object, object, object)  # (verts, faces, colors)

    def __init__(self, mask_vol, class_vol, spacing, parent=None):
        super().__init__(parent)
        self.mask_vol = mask_vol.copy()
        self.class_vol = class_vol.copy() if class_vol is not None else None
        # spacing comes in as (sx,sy,sz)=(x,y,z); convert to array axis order (z,y,x)
        self.spacing = spacing
        self.mc_spacing = (spacing[2], spacing[1], spacing[0])

    def run(self):
        if not _HAS_SKM or not np.any(self.mask_vol):
            self.finished.emit(None, None, None)
            return
        try:
            verts, faces, _, _ = marching_cubes(self.mask_vol, level=0.5, spacing=self.mc_spacing)

            colors = np.zeros((len(verts), 4), dtype=np.float32)
            colors[:, :] = (1.0, 0.3, 0.3, 0.8)

            if self.class_vol is not None and np.any(self.class_vol):
                from scipy.ndimage import maximum_filter
                dilated_class = maximum_filter(self.class_vol, size=3)
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

            # Reorder (z,y,x) → (x,y,z) for GL display with Z-up
            verts = verts[:, [2, 1, 0]]
            self.finished.emit(verts, faces, colors)
        except Exception as e:
            _logger.error("Mesh error: %s", e)
            self.finished.emit(None, None, None)
