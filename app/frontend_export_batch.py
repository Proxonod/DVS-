"""Batch GUI exporter for converting all RAW files in a directory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QWidget,
)

from app.export import export_stream
from app.io.metavision_reader import MetavisionReader, is_metavision_raw
from app.simple_export import _build_pipeline, _parse_colour


class BatchExportWorker(QThread):
    """Export all RAW files inside a directory sequentially."""

    progress = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        directory: str,
        speed: float,
        fps: float,
        pos_colour: str | None,
        neg_colour: str | None,
        export_codecs: Iterable[str],
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.speed = speed
        self.fps = fps
        self.pos_colour = pos_colour
        self.neg_colour = neg_colour
        self.export_codecs = tuple(export_codecs)

    def _iter_raw_files(self) -> Iterable[Path]:
        for path in sorted(self.directory.glob("*.raw")):
            if path.is_file():
                yield path

    def run(self) -> None:  # pragma: no cover - Qt thread / IO heavy
        try:
            raw_files = list(self._iter_raw_files())
            if not raw_files:
                raise RuntimeError("Im gewählten Verzeichnis wurden keine RAW-Dateien gefunden.")
            total = len(raw_files)
            for index, raw_path in enumerate(raw_files, start=1):
                self.progress.emit(f"[{index}/{total}] {raw_path.name} wird exportiert…")
                self._export_single(raw_path)
            self.finished.emit()
        except Exception as exc:  # noqa: BLE001 - surface message to UI
            self.failed.emit(str(exc))

    def _export_single(self, raw_path: Path) -> None:
        # Determine dimensions/duration once and reuse for both exports
        reader = MetavisionReader.from_raw(str(raw_path))
        try:
            dims = reader.ensure_sensor_size()
            if dims is None:
                raise RuntimeError(
                    f"Sensorauflösung der Datei konnte nicht ermittelt werden: {raw_path.name}"
                )
            width, height = dims

            duration_us = reader.metadata.duration_us
            if duration_us is None or duration_us <= 0:
                duration_us = reader.estimate_duration_us()
            if duration_us is None or duration_us <= 0:
                raise RuntimeError(f"Dauer des RAW-Streams unbekannt: {raw_path.name}")
            duration_s = duration_us / 1e6
        finally:
            try:
                reader.close()
            except Exception:
                pass

        targets = {
            "mp4": raw_path.with_suffix(".mp4"),
            "ffv1": raw_path.with_suffix(".mkv"),
        }

        for codec in self.export_codecs:
            out_path = targets[codec]
            self.progress.emit(f"→ {raw_path.name} -> {out_path.name}")
            reader = MetavisionReader.from_raw(str(raw_path))
            try:
                pipeline = _build_pipeline(width, height)
                pipeline.pos_colour = _parse_colour(self.pos_colour, pipeline.pos_colour)
                pipeline.neg_colour = _parse_colour(self.neg_colour, pipeline.neg_colour)
                export_stream(
                    reader,
                    pipeline,
                    duration_s,
                    self.fps,
                    str(out_path),
                    codec,
                    self.speed,
                )
            finally:
                try:
                    reader.close()
                except Exception:
                    pass


class BatchExportWindow(QWidget):
    """Small window to batch-export all RAW files in a directory."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Batch Export GUI")
        self.worker: BatchExportWorker | None = None

        self.dir_edit = QLineEdit()
        dir_btn = QPushButton("Verzeichnis…")
        dir_btn.clicked.connect(self._select_directory)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 240.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setValue(60.0)

        self.pos_colour_edit = QLineEdit("#00ffaa")
        pos_colour_btn = QPushButton("Wählen…")
        pos_colour_btn.clicked.connect(lambda: self._choose_colour(self.pos_colour_edit))

        self.neg_colour_edit = QLineEdit("#ff3366")
        neg_colour_btn = QPushButton("Wählen…")
        neg_colour_btn.clicked.connect(lambda: self._choose_colour(self.neg_colour_edit))

        self.mp4_check = QCheckBox("MP4")
        self.mp4_check.setChecked(True)
        self.ffv1_check = QCheckBox("FFV1")
        self.ffv1_check.setChecked(True)

        self.status_label = QLabel("Bereit")

        self.export_btn = QPushButton("Batch Export starten")
        self.export_btn.clicked.connect(self._start_export)

        layout = QFormLayout()
        layout.addRow("RAW-Verzeichnis", self._with_button(self.dir_edit, dir_btn))
        layout.addRow("Geschwindigkeit (x)", self.speed_spin)
        layout.addRow("FPS", self.fps_spin)
        layout.addRow("Pos. Farbe", self._with_button(self.pos_colour_edit, pos_colour_btn))
        layout.addRow("Neg. Farbe", self._with_button(self.neg_colour_edit, neg_colour_btn))
        layout.addRow("Formate", self._format_selector())
        layout.addRow(self.export_btn)
        layout.addRow(self.status_label)

        self.setLayout(layout)

    def _with_button(self, widget: QWidget, button: QPushButton) -> QWidget:
        container = QWidget()
        box = QHBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(widget)
        box.addWidget(button)
        return container

    def _choose_colour(self, target: QLineEdit) -> None:
        current = QColor(target.text())
        colour = QColorDialog.getColor(current, self, "Farbe wählen")
        if colour.isValid():
            target.setText(colour.name(QColor.NameFormat.HexRgb))

    def _select_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "RAW-Verzeichnis wählen")
        if directory:
            self.dir_edit.setText(directory)

    def _start_export(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        directory = self.dir_edit.text().strip()
        if not directory:
            QMessageBox.warning(self, "Fehlende Eingabe", "Bitte ein Verzeichnis auswählen.")
            return
        dir_path = Path(directory)
        if not dir_path.is_dir():
            QMessageBox.warning(self, "Ungültiges Verzeichnis", "Der angegebene Pfad ist kein Verzeichnis.")
            return
        raw_candidates = [path for path in dir_path.glob("*.raw") if path.is_file()]
        if not raw_candidates:
            QMessageBox.warning(
                self,
                "Keine RAW-Dateien",
                "Im ausgewählten Verzeichnis wurden keine RAW-Dateien gefunden.",
            )
            return
        invalid = [p.name for p in raw_candidates if not is_metavision_raw(str(p))]
        if invalid:
            QMessageBox.warning(
                self,
                "Ungültige Dateien",
                "Folgende Dateien scheinen keine Metavision RAWs zu sein:\n" + "\n".join(invalid),
            )
            return

        speed = self.speed_spin.value()
        fps = self.fps_spin.value()
        pos_colour = self.pos_colour_edit.text().strip() or None
        neg_colour = self.neg_colour_edit.text().strip() or None

        export_codecs = []
        if self.mp4_check.isChecked():
            export_codecs.append("mp4")
        if self.ffv1_check.isChecked():
            export_codecs.append("ffv1")
        if not export_codecs:
            QMessageBox.warning(
                self,
                "Kein Format gewählt",
                "Bitte wählen Sie mindestens ein Zielformat aus.",
            )
            return

        self.export_btn.setEnabled(False)
        self.status_label.setText("Export läuft…")

        worker = BatchExportWorker(directory, speed, fps, pos_colour, neg_colour, export_codecs)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        self.worker = worker
        worker.start()

    def _format_selector(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mp4_check)
        layout.addWidget(self.ffv1_check)
        layout.addStretch(1)
        return container

    def _on_progress(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_finished(self) -> None:
        self.worker = None
        self.status_label.setText("Fertig")
        self.export_btn.setEnabled(True)
        QMessageBox.information(self, "Export abgeschlossen", "Alle Dateien wurden exportiert.")

    def _on_failed(self, message: str) -> None:
        self.worker = None
        self.status_label.setText("Fehler beim Export")
        self.export_btn.setEnabled(True)
        QMessageBox.critical(self, "Fehler", message)


def main() -> None:  # pragma: no cover - Qt entry point
    app = QApplication.instance() or QApplication([])
    window = BatchExportWindow()
    window.resize(520, 0)
    window.show()
    app.exec()


if __name__ == "__main__":  # pragma: no cover
    main()
