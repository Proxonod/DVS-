# app/ui/main_window.py
# GUI with toolbar, RAW/camera open, play/pause, speed control with slice accumulator,
# view FPS cap, timeline, color settings, live filters, canvas grow, and duration estimation.

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize, QObject, QThread, Signal
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


class VideoExportWorker(QObject):
    """Background worker that renders a RAW stream to a video file."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        raw_path: str,
        out_path: str,
        codec: str,
        width: int,
        height: int,
        fps: float,
        playback_speed: float,
        decay: float,
        pos_colour: tuple[int, int, int],
        neg_colour: tuple[int, int, int],
        enable_baf: bool,
        baf_params: dict,
        enable_refractory: bool,
        refractory_params: dict,
    ) -> None:
        super().__init__()
        self.raw_path = raw_path
        self.out_path = out_path
        self.codec = codec
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1.0, float(fps))
        self.playback_speed = max(0.01, float(playback_speed))
        self.decay = float(decay)
        self.pos_colour = np.array(pos_colour, dtype=np.float32) / 255.0
        self.neg_colour = np.array(neg_colour, dtype=np.float32) / 255.0
        self.enable_baf = enable_baf
        self.baf_params = dict(baf_params)
        self.enable_refractory = enable_refractory
        self.refractory_params = dict(refractory_params)

    def _composite_frame(self, canvas: np.ndarray, events: np.ndarray) -> None:
        canvas *= self.decay
        if events is None or events.size == 0:
            return

        xs = events["x"].astype(int, copy=False)
        ys = events["y"].astype(int, copy=False)
        ps = events["p"].astype(int, copy=False)

        h, w, _ = canvas.shape
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        pos_mask = ps == 1
        if np.any(pos_mask):
            y_pos = ys[pos_mask]
            x_pos = xs[pos_mask]
            np.add.at(canvas[..., 0], (y_pos, x_pos), self.pos_colour[0])
            np.add.at(canvas[..., 1], (y_pos, x_pos), self.pos_colour[1])
            np.add.at(canvas[..., 2], (y_pos, x_pos), self.pos_colour[2])

        neg_mask = ps == 0
        if np.any(neg_mask):
            y_neg = ys[neg_mask]
            x_neg = xs[neg_mask]
            np.add.at(canvas[..., 0], (y_neg, x_neg), self.neg_colour[0])
            np.add.at(canvas[..., 1], (y_neg, x_neg), self.neg_colour[1])
            np.add.at(canvas[..., 2], (y_neg, x_neg), self.neg_colour[2])

        np.clip(canvas, 0.0, 1.0, out=canvas)

    def _write_frame(self, writer, canvas: np.ndarray) -> None:
        rgb8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
        writer.write(rgb8[..., ::-1])

    def run(self) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - OpenCV optional in tests
            self.error.emit(f"OpenCV VideoWriter not available: {exc}")
            return

        try:
            reader = MetavisionReader.from_raw(self.raw_path)
        except Exception as exc:
            self.error.emit(str(exc))
            return

        meta = reader.metadata
        width = meta.width or self.width
        height = meta.height or self.height
        if not width or not height:
            reader.close()
            self.error.emit("Cannot determine sensor dimensions for export.")
            return

        frame_size = (int(width), int(height))
        fourcc = cv2.VideoWriter_fourcc(*("FFV1" if self.codec == "ffv1" else "mp4v"))
        writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, frame_size)
        if not writer.isOpened():
            reader.close()
            self.error.emit(f"Unable to open video writer for {self.out_path}")
            return

        baf = None
        refractory = None
        if self.enable_baf:
            baf = BackgroundActivityFilter()
            try:
                baf.set_params(**self.baf_params)
            except Exception:
                pass
            baf.reset(frame_size[0], frame_size[1])
        if self.enable_refractory:
            refractory = RefractoryFilter()
            try:
                refractory.set_params(**self.refractory_params)
            except Exception:
                pass
            refractory.reset(frame_size[0], frame_size[1])

        frame_interval_us = 1e6 / self.fps
        effective_interval_us = frame_interval_us * self.playback_speed
        canvas = np.zeros((frame_size[1], frame_size[0], 3), np.float32)
        buffers: deque[np.ndarray] = deque()
        empty_events: Optional[np.ndarray] = None
        last_frame_time: float | None = None

        try:
            for events in reader:
                if empty_events is None:
                    empty_events = np.empty(0, dtype=events.dtype)
                if events.size == 0:
                    continue

                if refractory is not None:
                    state: dict[str, object] = {}
                    events = refractory.process(events, state).get("events", events)
                if baf is not None:
                    state = {}
                    events = baf.process(events, state).get("events", events)

                buffers.append(events)
                slice_end = float(events["t"][-1])
                if last_frame_time is None:
                    last_frame_time = slice_end - effective_interval_us

                while last_frame_time is not None and slice_end - last_frame_time >= effective_interval_us:
                    cutoff = last_frame_time + effective_interval_us
                    frame_chunks: list[np.ndarray] = []
                    while buffers and buffers[0]["t"][-1] < cutoff:
                        frame_chunks.append(buffers.popleft())

                    if buffers:
                        current = buffers[0]
                        mask = current["t"] < cutoff
                        if np.any(mask):
                            frame_chunks.append(current[mask])
                            remaining = current[~mask]
                            if remaining.size:
                                buffers[0] = remaining
                            else:
                                buffers.popleft()

                    if frame_chunks:
                        frame_events = frame_chunks[0] if len(frame_chunks) == 1 else np.concatenate(frame_chunks)
                    else:
                        frame_events = empty_events if empty_events is not None else np.empty(0, dtype=events.dtype)

                    self._composite_frame(canvas, frame_events)
                    self._write_frame(writer, canvas)
                    last_frame_time = cutoff

            if buffers:
                remaining_events = list(buffers)
                buffers.clear()
                combined = remaining_events[0] if len(remaining_events) == 1 else np.concatenate(remaining_events)
                self._composite_frame(canvas, combined)
                self._write_frame(writer, canvas)
        except Exception as exc:
            self.error.emit(str(exc))
            writer.release()
            reader.close()
            return

        writer.release()
        reader.close()
        self.finished.emit(self.out_path)


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
        self._pos_colour_vec = np.zeros(3, dtype=np.float32)
        self._neg_colour_vec = np.zeros(3, dtype=np.float32)
        self.meta_width = 640
        self.meta_height = 480
        self.current_time_us = 0
        self.duration_us: Optional[int] = None
        self.last_frame_ts = time.perf_counter()
        self.decay_per_frame = 0.90
        self.canvas: Optional[np.ndarray] = None

        # Speed control via slice accumulator
        self._slice_accum: float = 0.0
        self._frame_interval_us: float = 1e6 / max(1, self.view_fps_cap)

        # Filters
        self.filter_baf = BackgroundActivityFilter()
        self.filter_refractory = RefractoryFilter()
        self.enable_baf = False
        self.enable_refractory = False
        self._export_thread: Optional[QThread] = None
        self._export_worker: Optional[VideoExportWorker] = None

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
        self.view.setScaledContents(True)  # show full image without cropping
        vbox.addWidget(self.view, 1)

        tl = QHBoxLayout()
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.sliderPressed.connect(self._pause_while_scrubbing)
        self.time_slider.sliderReleased.connect(self._seek_released)
        tl.addWidget(QLabel("0 ms"))
        tl.addWidget(self.time_slider, 1)
        self.end_label = QLabel("unknown")
        tl.addWidget(self.end_label)
        vbox.addLayout(tl)

        self._build_toolbar()
        self._build_filter_dock()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)
        self._apply_view_fps_cap()

        self._bind_shortcuts()

        self._update_color_cache()

        if initial_file:
            try:
                self.open_file(initial_file)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    # ---------- UI ----------

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

        tb.addSeparator()
        act_export = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Export Video", self)
        act_export.triggered.connect(self.export_video_dialog)
        tb.addAction(act_export)

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
        self._slice_accum = 0.0  # reset speed accumulator
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
        self._slice_accum = 0.0
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
        self._update_color_cache()

    def _update_color_cache(self) -> None:
        self._pos_colour_vec = np.array(self.colors.pos, dtype=np.float32) / 255.0
        self._neg_colour_vec = np.array(self.colors.neg, dtype=np.float32) / 255.0

    # ---------- Playback ----------

    def toggle_playback(self):
        self.playing = not self.playing
        self._update_hud()

    def _on_speed_changed(self, txt):
        self.playback_speed = float(txt.replace("x", ""))
        self._slice_accum = 0.0  # optional reset for immediate response

    def _set_speed_value(self, v):
        self.playback_speed = v
        self.speed_combo.setCurrentText(f"{v:g}x")
        self._slice_accum = 0.0
        self._update_hud()

    def _on_fps_changed(self, txt):
        self.view_fps_cap = int(txt)
        self._apply_view_fps_cap()

    def _apply_view_fps_cap(self):
        interval_ms = max(1, int(1000 / max(1, self.view_fps_cap)))
        self._frame_interval_us = 1e6 / max(1, self.view_fps_cap)
        self.timer.start(interval_ms)

    def _pause_while_scrubbing(self):
        self._scrub_was_playing = self.playing
        self.playing = False

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
            # How many slices should we consume this frame?
            dt_us = getattr(self.reader, "delta_t_us", 5000) or 5000
            slices_per_frame_float = (self._frame_interval_us / dt_us) * max(0.01, self.playback_speed)
            self._slice_accum += slices_per_frame_float
            n_slices = int(self._slice_accum)

            if n_slices > 0:
                self._slice_accum -= n_slices
                for _ in range(n_slices):
                    try:
                        ev = next(self.reader)
                    except StopIteration:
                        if self._current_raw_path:
                            self.open_file(self._current_raw_path)
                            self.playing = True
                            break
                        self.playing = False
                        break

                    # Filters
                    frame_state = {}
                    if self.enable_refractory:
                        ev = self.filter_refractory.process(ev, frame_state).get("events", ev)
                    if self.enable_baf:
                        ev = self.filter_baf.process(ev, frame_state).get("events", ev)

                    # Render
                    self._ensure_canvas()
                    self._composite_events(ev)
            else:
                # No new slice due yet: still decay for smoothness
                if self.canvas is not None:
                    self.canvas *= self.decay_per_frame

        # Present
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

        # Dynamic resize if needed
        self._grow_canvas_if_needed(xs, ys)

        xs = np.clip(xs, 0, self.meta_width - 1)
        ys = np.clip(ys, 0, self.meta_height - 1)

        pos_mask = ps == 1
        if np.any(pos_mask):
            y_pos = ys[pos_mask]
            x_pos = xs[pos_mask]
            np.add.at(self.canvas[..., 0], (y_pos, x_pos), self._pos_colour_vec[0])
            np.add.at(self.canvas[..., 1], (y_pos, x_pos), self._pos_colour_vec[1])
            np.add.at(self.canvas[..., 2], (y_pos, x_pos), self._pos_colour_vec[2])

        neg_mask = ps == 0
        if np.any(neg_mask):
            y_neg = ys[neg_mask]
            x_neg = xs[neg_mask]
            np.add.at(self.canvas[..., 0], (y_neg, x_neg), self._neg_colour_vec[0])
            np.add.at(self.canvas[..., 1], (y_neg, x_neg), self._neg_colour_vec[1])
            np.add.at(self.canvas[..., 2], (y_neg, x_neg), self._neg_colour_vec[2])

        np.clip(self.canvas, 0.0, 1.0, out=self.canvas)

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

    # ---------- Export ----------

    def export_video_dialog(self):
        if self._export_thread and self._export_thread.isRunning():
            QMessageBox.information(self, "Export", "A video export is already in progress.")
            return
        if not self._current_raw_path:
            QMessageBox.information(self, "Export", "Load a RAW file before exporting.")
            return

        base = Path(self._current_raw_path).with_suffix("")
        default_path = str(base.with_suffix(".mp4"))
        filters = "MP4 Video (*.mp4);;Lossless FFV1 (*.mkv)"
        out_path, selected_filter = QFileDialog.getSaveFileName(self, "Export Video", default_path, filters)
        if not out_path:
            return

        out_path = str(out_path)
        sel = (selected_filter or "").lower()
        if out_path.lower().endswith(".mkv") or "ffv1" in sel:
            codec = "ffv1"
            if not out_path.lower().endswith(".mkv"):
                out_path += ".mkv"
        else:
            codec = "mp4"
            if not out_path.lower().endswith(".mp4"):
                out_path += ".mp4"

        self._start_video_export(out_path, codec)

    def _start_video_export(self, out_path: str, codec: str) -> None:
        if not self._current_raw_path:
            return

        worker = VideoExportWorker(
            raw_path=self._current_raw_path,
            out_path=out_path,
            codec=codec,
            width=self.meta_width,
            height=self.meta_height,
            fps=self.view_fps_cap,
            playback_speed=self.playback_speed,
            decay=self.decay_per_frame,
            pos_colour=self.colors.pos,
            neg_colour=self.colors.neg,
            enable_baf=self.enable_baf,
            baf_params=self.filter_baf.params(),
            enable_refractory=self.enable_refractory,
            refractory_params=self.filter_refractory.params(),
        )

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_export_finished)
        worker.error.connect(self._on_export_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_export_refs)
        thread.start()

        self._export_thread = thread
        self._export_worker = worker
        self.statusBar().showMessage(f"Exporting to {out_path}…")

    def _on_export_finished(self, out_path: str) -> None:
        self.statusBar().showMessage(f"Export complete: {os.path.basename(out_path)}", 5000)
        QMessageBox.information(self, "Export complete", f"Video saved to:\n{out_path}")

    def _on_export_error(self, message: str) -> None:
        self.statusBar().showMessage("Export failed", 5000)
        QMessageBox.critical(self, "Export failed", message)

    def _clear_export_refs(self) -> None:
        self._export_thread = None
        self._export_worker = None
