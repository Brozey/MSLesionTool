"""
HUD-style splash screen for MSLesionTool.

Native Qt implementation with QPainter — no external dependencies.
Renders a sci-fi/medical HUD with crosshair, corner brackets, and info panels.
"""

import os
import sys
import math
import time
from datetime import datetime


_CYAN = "#4af0ff"
_BG = "#040810"


class _HUDSplashWidget:
    """QPainter-based HUD splash matching the Three.js visual style."""

    def __init__(self):
        from PyQt6.QtWidgets import QWidget, QApplication
        from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
        from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QLinearGradient

        class _Canvas(QWidget):
            def __init__(self):
                super().__init__(None)
                self.setWindowFlags(
                    Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.WindowStaysOnTopHint
                    | Qt.WindowType.Tool
                )
                self.resize(940, 640)
                self._progress = 0
                self._progress_text = "INITIALIZING..."
                self._t0 = time.time()
                self._anim_timer = QTimer(self)
                self._anim_timer.timeout.connect(self.update)
                self._anim_timer.start(33)  # ~30 fps

                screen = QApplication.primaryScreen()
                if screen:
                    g = screen.availableGeometry()
                    self.move(g.center().x() - 470, g.center().y() - 320)

            def set_progress(self, val, text=""):
                self._progress = val
                if text:
                    self._progress_text = text
                self.update()

            def paintEvent(self, _ev):
                p = QPainter(self)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                W, H = self.width(), self.height()
                t = time.time() - self._t0  # animation time

                # ── Background ──
                p.fillRect(0, 0, W, H, QColor(4, 8, 16))

                # Subtle vignette
                for i in range(40):
                    a = int(3 * (40 - i) / 40)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(0, 0, 0, a))
                    p.drawRect(i, i, W - 2 * i, H - 2 * i)

                cyan = QColor(74, 240, 255)
                dim = QColor(74, 240, 255, 40)
                vdim = QColor(74, 240, 255, 15)

                # ── Grid lines (subtle) ──
                p.setPen(QPen(QColor(74, 240, 255, 8), 1))
                for x in range(0, W, 60):
                    p.drawLine(x, 0, x, H)
                for y in range(0, H, 60):
                    p.drawLine(0, y, W, y)

                # ── Corner brackets ──
                BL = 40  # bracket length
                p.setPen(QPen(QColor(74, 240, 255, 120), 2))
                # top-left
                p.drawLine(8, 8, 8 + BL, 8)
                p.drawLine(8, 8, 8, 8 + BL)
                # top-right
                p.drawLine(W - 8, 8, W - 8 - BL, 8)
                p.drawLine(W - 8, 8, W - 8, 8 + BL)
                # bottom-left
                p.drawLine(8, H - 8, 8 + BL, H - 8)
                p.drawLine(8, H - 8, 8, H - 8 - BL)
                # bottom-right
                p.drawLine(W - 8, H - 8, W - 8 - BL, H - 8)
                p.drawLine(W - 8, H - 8, W - 8, H - 8 - BL)

                # ── Center crosshair / targeting reticle ──
                cx, cy = W // 2, H // 2 - 30

                # Outer circle (rotating dashed)
                p.setPen(QPen(dim, 1, Qt.PenStyle.DashLine))
                r_outer = 120
                p.save()
                p.translate(cx, cy)
                p.rotate(t * 8)  # slow rotation
                p.drawEllipse(QPointF(0, 0), r_outer, r_outer)
                p.restore()

                # Middle circle
                r_mid = 80
                p.setPen(QPen(QColor(74, 240, 255, 60), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(cx, cy), r_mid, r_mid)

                # Inner circle (pulsing)
                pulse = 0.85 + 0.15 * math.sin(t * 3)
                r_inner = int(40 * pulse)
                p.setPen(QPen(QColor(74, 240, 255, 80), 1))
                p.drawEllipse(QPointF(cx, cy), r_inner, r_inner)

                # Crosshair lines
                gap = 20
                line_len = 50
                p.setPen(QPen(QColor(74, 240, 255, 100), 1))
                p.drawLine(cx - gap - line_len, cy, cx - gap, cy)
                p.drawLine(cx + gap, cy, cx + gap + line_len, cy)
                p.drawLine(cx, cy - gap - line_len, cx, cy - gap)
                p.drawLine(cx, cy + gap, cx, cy + gap + line_len)

                # Small center brackets
                sb = 12
                p.setPen(QPen(QColor(74, 240, 255, 140), 1.5))
                p.drawLine(cx - sb, cy - sb, cx - sb + 6, cy - sb)
                p.drawLine(cx - sb, cy - sb, cx - sb, cy - sb + 6)
                p.drawLine(cx + sb, cy - sb, cx + sb - 6, cy - sb)
                p.drawLine(cx + sb, cy - sb, cx + sb, cy - sb + 6)
                p.drawLine(cx - sb, cy + sb, cx - sb + 6, cy + sb)
                p.drawLine(cx - sb, cy + sb, cx - sb, cy + sb - 6)
                p.drawLine(cx + sb, cy + sb, cx + sb - 6, cy + sb)
                p.drawLine(cx + sb, cy + sb, cx + sb, cy + sb - 6)

                # Tick marks on circles
                p.setPen(QPen(QColor(74, 240, 255, 50), 1))
                for angle in range(0, 360, 15):
                    rad = math.radians(angle + t * 12)
                    x1 = cx + r_outer * math.cos(rad)
                    y1 = cy + r_outer * math.sin(rad)
                    x2 = cx + (r_outer + 6) * math.cos(rad)
                    y2 = cy + (r_outer + 6) * math.sin(rad)
                    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

                # ── Top bar ──
                p.fillRect(0, 0, W, 44, QColor(4, 8, 16, 220))
                p.setPen(QPen(QColor(74, 240, 255, 40), 1))
                p.drawLine(0, 44, W, 44)

                # Title
                p.setPen(QColor(74, 240, 255))
                fnt_title = QFont("Courier New", 9)
                fnt_title.setBold(True)
                fnt_title.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
                p.setFont(fnt_title)
                p.drawText(20, 18, "MS LESION SEGMENTATION TOOL")

                # Subtitle with date
                fnt_sub = QFont("Courier New", 7)
                fnt_sub.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
                p.setFont(fnt_sub)
                p.setPen(QColor(74, 240, 255, 100))
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                p.drawText(20, 34, f"v1.0 · nnUNet 3-ARCH ENSEMBLE · {now}")

                # Top-right status
                p.setPen(QColor(74, 240, 255))
                p.setFont(fnt_title)
                status_x = W - 20
                p.drawText(QRectF(status_x - 250, 4, 250, 16),
                           Qt.AlignmentFlag.AlignRight, "FLAIR + T1")
                p.setFont(fnt_sub)
                p.setPen(QColor(74, 240, 255, 100))

                # ── Left info panel ──
                lx, ly = 30, 180
                fnt_label = QFont("Courier New", 7)
                fnt_label.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.5)
                fnt_value = QFont("Courier New", 11)
                fnt_value.setBold(True)

                for label, value in [("ARCHITECTURE", "3-MODEL ENSEMBLE"),
                                     ("BEST DICE", "0.7179"),
                                     ("DATASET", "MSLesSeg-2024")]:
                    p.setFont(fnt_label)
                    p.setPen(QColor(74, 240, 255, 115))
                    p.drawText(lx, ly, label)
                    p.setFont(fnt_value)
                    p.setPen(cyan)
                    p.drawText(lx, ly + 18, value)
                    ly += 55

                # ── Right info panel ──
                rx, ry = W - 30, 180
                for label, value in [("MODELS", "CNN3D · ResEncL · 2.5D"),
                                     ("FOLDS", "BEST-2 (F1, F3)"),
                                     ("BACKEND", "PyTorch · ONNX-RT")]:
                    p.setFont(fnt_label)
                    p.setPen(QColor(74, 240, 255, 115))
                    p.drawText(QRectF(rx - 250, ry - 12, 250, 16),
                               Qt.AlignmentFlag.AlignRight, label)
                    p.setFont(fnt_value)
                    p.setPen(cyan)
                    p.drawText(QRectF(rx - 250, ry + 4, 250, 20),
                               Qt.AlignmentFlag.AlignRight, value)
                    ry += 55

                # ── Bottom-right: model cards ──
                card_x = W - 180
                card_y = H - 155
                fnt_card = QFont("Courier New", 8)
                fnt_card.setBold(True)
                for name, size in [("CNN 3D · 239 MB", cyan),
                                   ("ResEncL 3D · 782 MB", cyan),
                                   ("Conv 2.5D · 158 MB", cyan)]:
                    p.setPen(QPen(QColor(74, 240, 255, 60), 1))
                    p.setBrush(QColor(4, 8, 16, 180))
                    p.drawRect(QRectF(card_x, card_y, 160, 24))
                    p.setPen(cyan)
                    p.setFont(fnt_card)
                    p.drawText(QRectF(card_x, card_y, 160, 24),
                               Qt.AlignmentFlag.AlignCenter, name)
                    card_y += 32

                # ── Bottom-left: legend ──
                leg_x, leg_y = 30, H - 130
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 60, 100))
                p.drawEllipse(leg_x, leg_y, 8, 8)
                p.setPen(QColor(74, 240, 255, 160))
                p.setFont(QFont("Courier New", 7))
                p.drawText(leg_x + 14, leg_y + 8, "MS LESIONS (WM)")

                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(160, 160, 160))
                p.drawEllipse(leg_x, leg_y + 18, 8, 8)
                p.setPen(QColor(74, 240, 255, 160))
                p.drawText(leg_x + 14, leg_y + 26, "BRAIN SURFACE")

                # ── Bottom title ──
                p.setPen(cyan)
                fnt_big = QFont("Courier New", 22)
                fnt_big.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 10)
                p.setFont(fnt_big)
                p.drawText(QRectF(0, H - 90, W, 35),
                           Qt.AlignmentFlag.AlignCenter, "MSLESIONTOOL")

                p.setPen(QColor(74, 240, 255, 100))
                fnt_tagline = QFont("Courier New", 7)
                fnt_tagline.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4)
                p.setFont(fnt_tagline)
                p.drawText(QRectF(0, H - 60, W, 16),
                           Qt.AlignmentFlag.AlignCenter,
                           "AUTOMATED MS LESION SEGMENTATION & ANALYSIS")

                # ── Progress bar ──
                bar_w = 400
                bar_h = 16
                bar_x = (W - bar_w) // 2
                bar_y = H - 35
                # Background
                p.setPen(QPen(QColor(74, 240, 255, 40), 1))
                p.setBrush(QColor(74, 240, 255, 15))
                p.drawRect(QRectF(bar_x, bar_y, bar_w, bar_h))
                # Fill
                fill_w = bar_w * (self._progress / 100.0)
                if fill_w > 0:
                    grad = QLinearGradient(bar_x, 0, bar_x + fill_w, 0)
                    grad.setColorAt(0.0, QColor(74, 240, 255, 80))
                    grad.setColorAt(1.0, QColor(74, 240, 255, 180))
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(grad))
                    p.drawRect(QRectF(bar_x + 1, bar_y + 1, fill_w - 2, bar_h - 2))
                # Text
                p.setPen(cyan)
                fnt_prog = QFont("Courier New", 7)
                fnt_prog.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
                p.setFont(fnt_prog)
                p.drawText(QRectF(bar_x, bar_y, bar_w - 8, bar_h),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                           f"  {self._progress_text}")
                p.drawText(QRectF(bar_x, bar_y, bar_w - 8, bar_h),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                           f"{self._progress}%")

                # ── Outer border ──
                p.setPen(QPen(QColor(74, 240, 255, 25), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(0, 0, W - 1, H - 1)

                p.end()

        self._widget = _Canvas()

    @property
    def widget(self):
        return self._widget


class Splash3DWidget:
    """Splash screen wrapper — native QPainter HUD."""

    def __init__(self):
        self._mode = "qt"
        self._hud = _HUDSplashWidget()
        self._qt_widget = self._hud.widget

    def show(self):
        if self._qt_widget:
            self._qt_widget.show()

    def raise_(self):
        if self._qt_widget:
            self._qt_widget.raise_()

    def activateWindow(self):
        if self._qt_widget:
            self._qt_widget.activateWindow()

    def isVisible(self):
        if self._qt_widget:
            return self._qt_widget.isVisible()
        return False

    def set_progress(self, value, text=""):
        if self._qt_widget:
            try:
                self._qt_widget.set_progress(int(value), text)
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().processEvents()
            except (RuntimeError, AttributeError):
                pass

    def finish(self, _window=None):
        if self._qt_widget:
            try:
                self._qt_widget._anim_timer.stop()
                self._qt_widget.close()
                self._qt_widget.deleteLater()
            except RuntimeError:
                pass
            self._qt_widget = None

    def close(self):
        self.finish()
