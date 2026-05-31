"""
msseg.viewer – 2D slice viewer widgets and image conversion utilities.
"""
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QSizePolicy, QVBoxLayout, QHBoxLayout, QGridLayout,
    QCheckBox, QLabel, QSlider, QToolButton, QButtonGroup, QSpinBox,
)
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush

from .constants import SEG_LABELS, CLASS_LABELS

try:
    import matplotlib.cm as cm
    _HAS_CM = True
except ImportError:
    _HAS_CM = False


# ─── Utility functions ────────────────────────────────────────────

def extract_slice(volume, axis, index):
    """Extract a 2D slice from a 3D volume along the given axis.
    Flips vertically so display matches radiological convention."""
    if volume is None:
        return None
    D, H, W = volume.shape
    if axis == 0 and 0 <= index < D:
        return np.flipud(volume[index, :, :])
    elif axis == 1 and 0 <= index < W:
        return np.flipud(volume[:, :, index])
    elif axis == 2 and 0 <= index < H:
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
    mask = prob_2d >= threshold
    if not np.any(mask) or not _HAS_CM:
        return rgba
    vals = np.clip(prob_2d[mask], 0, 1)
    colors = cm.turbo(vals)
    rgba[mask, 0] = (colors[:, 0] * 255).astype(np.uint8)
    rgba[mask, 1] = (colors[:, 1] * 255).astype(np.uint8)
    rgba[mask, 2] = (colors[:, 2] * 255).astype(np.uint8)
    rgba[mask, 3] = (vals * 255).astype(np.uint8)
    return rgba


def classification_to_rgba(class_vol_2d):
    """Convert a 2D classification array to RGBA with CLASS_LABELS colors."""
    cmap = {k: v[1] for k, v in CLASS_LABELS.items() if k > 0}
    return labels_to_rgba(class_vol_2d, cmap)


def bresenham(x0, y0, x1, y1):
    """Generator yielding integer pixel coordinates along a Bresenham line."""
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


# ─── SliceCanvas ──────────────────────────────────────────────────

class SliceCanvas(QWidget):
    """Zoomable/pannable 2D slice viewer for one orthogonal plane."""
    scrollRequested = pyqtSignal(int)
    viewChanged = pyqtSignal(float, float, float)
    crosshairMoved = pyqtSignal(float, float)
    brushStroke = pyqtSignal(float, float, bool)
    brushReleased = pyqtSignal()
    lesionPicked = pyqtSignal(float, float)
    doubleClicked = pyqtSignal()

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._overlay_pixmap = None
        self._class_overlay_pixmap = None
        self._overlay_alpha = 0.45
        self._scale = 0.0
        self._aspect = (1.0, 1.0)
        self._offset = QPointF(0, 0)
        self._dragging = False
        self._last_mouse = QPointF()
        self._title = title
        self._crosshair = None
        self._draw_mode = False
        self._painting = False
        self._brush_size = 3
        self._grow_mode = False
        self._classify_mode = False
        self._cursor_pos = None
        self._show_class_overlay = True
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
        self._scale = 0.0

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
            p.fillRect(self.rect(), QColor(30, 30, 30))
            p.setPen(QColor(100, 100, 100))
            p.setFont(QFont("Arial", 11))
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
        p.resetTransform()

        # crosshair
        if self._crosshair is not None and self._pixmap is not None:
            cx, cy = self._crosshair
            pw, ph = self._pixmap.width(), self._pixmap.height()
            wl = self._img_to_widget(QPointF(0, cy))
            wr = self._img_to_widget(QPointF(pw, cy))
            wt = self._img_to_widget(QPointF(cx, 0))
            wb = self._img_to_widget(QPointF(cx, ph))
            p.setPen(QPen(QColor(0, 255, 0, 140), 1, Qt.PenStyle.DashLine))
            p.drawLine(wl, wr)
            p.drawLine(wt, wb)

        # title
        if self._title:
            p.setPen(QColor(220, 220, 220))
            p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
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
            if self._grow_mode or self._classify_mode:
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
        elif bool(ev.buttons() & Qt.MouseButton.LeftButton) and not (
                self._draw_mode or self._grow_mode or self._classify_mode):
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


# ─── DrawingToolbar ───────────────────────────────────────────────

class DrawingToolbar(QWidget):
    """Floating tool window for drawing, erasing, lesion growth, and classification."""
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
        self.setFixedWidth(230)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # Drawing section
        sep1 = QLabel("── Drawing ──")
        sep1.setStyleSheet("color:#aaa; font-weight:bold;")
        lay.addWidget(sep1)

        self._chk_draw = QCheckBox("Enable Drawing")
        self._chk_draw.toggled.connect(self.drawingToggled.emit)
        lay.addWidget(self._chk_draw)

        tool_row = QHBoxLayout()
        self._btn_brush = QToolButton(); self._btn_brush.setText("Brush")
        self._btn_brush.setCheckable(True); self._btn_brush.setChecked(True)
        self._btn_eraser = QToolButton(); self._btn_eraser.setText("Eraser")
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

        # Lesion Growth section
        sep2 = QLabel("── Lesion Growth ──")
        sep2.setStyleSheet("color:#aaa; font-weight:bold;")
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

        # Classification section
        sep3 = QLabel("── Classification ──")
        sep3.setStyleSheet("color:#aaa; font-weight:bold;")
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
