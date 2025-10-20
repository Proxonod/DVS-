# app/ui/main_window.py
# GUI with toolbar, RAW/camera open, play/pause, speed, view-FPS cap,
# timeline, color settings, and live filters. Fixes:
# - dynamic resize of canvas to true sensor size (no top-left cropping)
# - duration estimation when SDK does not provide it

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QColor
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QLabel,
    QToolBar,
    QComboBox,
    QHBoxLayout,
    QSlider,
    QStyle,
    QSizePolicy,
    QColorDialog,
    QDockWidget,
)

from ..io.metavision_reader import MetavisionReader, is_metavision_raw
from ..filters.baf import BackgroundActivityFilter
from ..filters.refractory import RefractoryFilter
from .filter_panel import FilterPanel


@dataclass
class ViewColors:
    pos: tuple[int, int, int] = (0, 255, 170)
    neg: tuple[int, int, int] = (255, 51, 102)


class MainWindow(QMainWindow):
    def __init__(self, initial_file: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("DVS Viewer")
        self.resize(1200, 750)

        # Core state
        self.reader: Optional[MetavisionReader] = None
        self._current_raw_path: Optional[str] = None
        self.playing = False
        self.playback_speed = 1.0
        self.view_fps_cap = 60
        self.colors = ViewColors()
        self.meta_width = 640
        self.meta_height = 480
        self.current_time_us = 0
        self.duration_us: Optional[int] = None
        self.last_frame_ts = time.perf_counter()
        self.decay_per_frame = 0.90
        self.canvas: Optional[np.ndarray] = None

        # Filters
        self.filter_baf = BackgroundActivityFilter()
        self.filter_refractory = RefractoryFilter()
        self.enable_baf = False
        self.enable_refractory = False

        # Central UI
        central = QWidget(self)
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        self.hud = QLabel("Ready.")
        self.hud.setStyleSheet("color:#444;")
        vbox.addWidget(self.hud)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setScaledContents(True)  # crucial to show full frame without cropping
        vbox.addWidget(self.view, 1)

        tl = QHBoxLayout()
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.sliderPressed.connect(self._pause_while_scrubbing)
        self.time_slider.sliderReleased.connect(self._seek_released)
        tl.addWidget(QLabel("0 ms"))
        tl.addWidget(self.time_slider, 1)
        self.end_label = QLabel("…")
        tl.addWidget(self.end_label)
        vbox.addLayout(tl)

        self._build_toolbar()
        self._build_filter_dock()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)
        self._apply_view_fps_cap()

        self._bind_shortcuts()

        if initial_file:
            try:
                self.open_file(initial_file)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    # ---------- UI building ----------

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        act_open = QAction(self.style().standardIcon(QStyle.SP_DialogOpenButton), "Open RAW", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.open_file_dialog)
        tb.addAction(act_open)

        act_cam = QAction("Camera", self)
        act_cam.triggered.connect(self.open_camera)
        tb.addAction(act_cam)

        tb.addSeparator()

        self.act_play = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Play/Pause", self)
        self.act_play.setShortcut(Qt.Key_Space)
        self.act_play.triggered.connect(self.toggle_playback)
        tb.addAction(self.act_play)

        tb.addWidget(QLabel("Speed"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "8x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        tb.addWidget(self.speed_combo)

        tb.addSeparator()

        tb.addWidget(QLabel("View FPS"))
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "30", "60", "120"])
        self.fps_combo.setCurrentText("60")
        self.fps_combo.currentTextChanged.connect(self._on_fps_changed)
        tb.addWidget(self.fps_combo)

        tb.addSeparator()
        act_colors = QAction("Colours", self)
        act_colors.triggered.connect(self._open_color_dialog)
        tb.addAction(act_colors)

    def _build_filter_dock(self):
        dock = QDockWidget("Filters", self)
        panel = FilterPanel(
            on_toggle_baf=self._on_toggle_baf,
            on_change_baf=self._on_change_baf,
            on_toggle_refractory=self._on_toggle_refractory,
            on_change_refractory=self._on_change_refractory,
            parent=dock,
        )
        dock.setWidget(panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _bind_shortcuts(self):
        for key, val in [(Qt.Key_J, 0.5), (Qt.Key_K, 1.0), (Qt.Key_L, 2.0)]:
            a = QAction(self); a.setShortcut(key)
            a.triggered.connect(lambda _, v=val: self._set_speed_value(v))
            self.addAction(a)
        for key, d in [(Qt.Key_Left, -5000), (Qt.Key_Right, +5000)]:
            a = QAction(self); a.setShortcut(key)
            a.triggered.connect(lambda _, delta=d: self._nudge_time(delta))
            self.addAction(a)
        act_f = QAction(self); act_f.setShortcut(Qt.Key_F)
        act_f.triggered.connect(self._toggle_fullscreen)
        self.addAction(act_f)

    # ---------- File/camera ----------

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open DVS RAW", str(Path.home()), "RAW Files (*.raw)")
        if path:
            self.open_file(path)

    def open_file(self, path: str):
        if not is_metavision_raw(path):
            QMessageBox.warning(self, "Unsupported", "Choose a .raw file.")
            return
        if self.reader:
            self.reader.close()
        self.reader = MetavisionReader.from_raw(path)
        meta = self.reader.metadata
        self.meta_width = meta.width or 640
        self.meta_height = meta.height or 480
        self.duration_us = meta.duration_us
        self._current_raw_path = path

        # If SDK did not provide duration, try to estimate once
        if not self.duration_us:
            try:
                dur = self.reader.estimate_duration_us(delta_t_us=100000)
                if dur:
                    self.duration_us = dur
            except Exception:
                pass

        self._ensure_canvas()
        self.filter_baf.reset(self.meta_width, self.meta_height)
        self.filter_refractory.reset(self.meta_width, self.meta_height)
        self.playing = False
        self.current_time_us = 0
        self.end_label.setText(f"{(self.duration_us or 0)/1000:.0f} ms" if self.duration_us else "unknown")
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)}")
        self._update_hud()

    def open_camera(self):
        try:
            self.reader = MetavisionReader.from_camera()
        except Exception as e:
            QMessageBox.warning(self, "Camera", str(e))
            return
        meta = self.reader.metadata
        self.meta_width = meta.width or 640
        self.meta_height = meta.height or 480
        self._ensure_canvas()
        self.duration_us = None
        self.filter_baf.reset(self.meta_width, self.meta_height)
        self.filter_refractory.reset(self.meta_width, self.meta_height)
        self.playing = True
        self._update_hud()

    # ---------- Filters & colors ----------

    def _on_toggle_baf(self, en: bool): self.enable_baf = en
    def _on_change_baf(self, w, c, r, s): self.filter_baf.set_params(window_ms=w, count_threshold=c, refractory_us=r, spatial_radius=s)
    def _on_toggle_refractory(self, en: bool): self.enable_refractory = en
    def _on_change_refractory(self, r): self.filter_refractory.set_params(refractory_us=r)

    def _open_color_dialog(self):
        c = QColorDialog.getColor(QColor(*self.colors.pos), self, "Positive Polarity Colour")
        if c.isValid(): self.colors.pos = (c.red(), c.green(), c.blue())
        c2 = QColorDialog.getColor(QColor(*self.colors.neg), self, "Negative Polarity Colour")
        if c2.isValid(): self.colors.neg = (c2.red(), c2.green(), c2.blue())

    # ---------- Playback ----------

    def toggle_playback(self):
        self.playing = not self.playing
        self._update_hud()

    def _on_speed_changed(self, txt): self.playback_speed = float(txt.replace("x", ""))
    def _set_speed_value(self, v): self.playback_speed = v; self.speed_combo.setCurrentText(f"{v:g}x"); self._update_hud()
    def _on_fps_changed(self, txt): self.view_fps_cap = int(txt); self._apply_view_fps_cap()
    def _apply_view_fps_cap(self): self.timer.start(max(1, int(1000 / max(1, self.view_fps_cap))))

    def _pause_while_scrubbing(self): self._scrub_was_playing = self.playing; self.playing = False
    def _seek_released(self):
        if self.duration_us:
            pos = self.time_slider.value() / 1000.0
            self.current_time_us = int(pos * self.duration_us)
        self.playing = getattr(self, "_scrub_was_playing", False)

    def _nudge_time(self, delta_us):
        self.current_time_us = max(0, self.current_time_us + delta_us)
        if self.duration_us:
            self.current_time_us = min(self.current_time_us, self.duration_us)
            slider = int(1000 * (self.current_time_us / self.duration_us))
            self.time_slider.setValue(slider)
        self._update_hud()

    # ---------- Frame loop ----------

    def _on_frame(self):
        self.last_frame_ts = time.perf_counter()
        if not self.reader:
            self.view.setText("No source loaded.")
            return
        if self.playing:
            try: events = next(self.reader)
            except StopIteration:
                if self._current_raw_path:
                    self.open_file(self._current_raw_path); self.playing = True; return
                self.playing = False; return

            # Filters
            frame_state = {}
            ev = events
            if self.enable_refractory: ev = self.filter_refractory.process(ev, frame_state).get("events", ev)
            if self.enable_baf: ev = self.filter_baf.process(ev, frame_state).get("events", ev)

            # Render
            self._ensure_canvas()
            self._composite_events(ev)

        # Show frame
        if self.canvas is not None:
            rgb8 = np.clip(self.canvas * 255.0, 0, 255).astype(np.uint8)
            h, w, _ = rgb8.shape
            qimg = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.view.setPixmap(QPixmap.fromImage(qimg))

        self._update_hud()

    # ---------- Rendering ----------

    def _ensure_canvas(self):
        if self.canvas is None or self.canvas.shape[:2] != (self.meta_height, self.meta_width):
            self.canvas = np.zeros((self.meta_height, self.meta_width, 3), np.float32)

    def _grow_canvas_if_needed(self, xs: np.ndarray, ys: np.ndarray):
        """If event coordinates exceed current size, grow canvas and reset filters."""
        max_x = int(xs.max()) + 1 if xs.size else self.meta_width
        max_y = int(ys.max()) + 1 if ys.size else self.meta_height
        need_w = max_x > self.meta_width
        need_h = max_y > self.meta_height
        if need_w or need_h:
            self.meta_width = max(self.meta_width, max_x)
            self.meta_height = max(self.meta_height, max_y)
            self._ensure_canvas()
            self.filter_baf.reset(self.meta_width, self.meta_height)
            self.filter_refractory.reset(self.meta_width, self.meta_height)

    def _composite_events(self, ev: np.ndarray):
        if self.canvas is None:
            return
        self.canvas *= self.decay_per_frame
        if ev is None or ev.size == 0:
            return

        xs = ev["x"].astype(int, copy=False)
        ys = ev["y"].astype(int, copy=False)
        ps = ev["p"].astype(int, copy=False)

        # Dynamic resize if SDK did not give correct sensor size
        self._grow_canvas_if_needed(xs, ys)

        # Safe clipping after possible grow
        xs = np.clip(xs, 0, self.meta_width - 1)
        ys = np.clip(ys, 0, self.meta_height - 1)

        pos = np.array(self.colors.pos, dtype=np.float32) / 255.0
        neg = np.array(self.colors.neg, dtype=np.float32) / 255.0

        if np.any(ps == 1):
            idx = (ys[ps == 1], xs[ps == 1])
            self.canvas[idx] = np.minimum(1.0, self.canvas[idx] + pos)
        if np.any(ps == 0):
            idx = (ys[ps == 0], xs[ps == 0])
            self.canvas[idx] = np.minimum(1.0, self.canvas[idx] + neg)

        try:
            self.current_time_us = int(ev["t"][-1])
        except Exception:
            pass

    # ---------- HUD ----------

    def _update_hud(self):
        fps = f"{self.view_fps_cap} fps"; spd = f"{self.playback_speed:g}x"
        t_ms = self.current_time_us / 1000.0
        if self.duration_us:
            dur_ms = self.duration_us / 1000.0
            slider_pos = int(1000 * (self.current_time_us / self.duration_us))
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(max(0, min(1000, slider_pos)))
            self.time_slider.blockSignals(False)
            end_txt = f"{dur_ms:.0f} ms"
        else:
            end_txt = "unknown"
        self.end_label.setText(end_txt)
        state = "PLAY" if self.playing else "PAUSE"
        self.hud.setText(f"{state} | {t_ms:.1f} / {end_txt} | view {fps} | speed {spd}")

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()
