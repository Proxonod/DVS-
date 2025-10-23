"""Simple GUI wrapper for the ``simple_export`` pipeline.

This module provides a tiny Qt-based frontend that mirrors the
``simple_export`` command-line script but lets users enter all parameters
through input fields.  It reuses the same pipeline helpers to ensure the
rendered results match the CLI behaviour.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QComboBox,
    QWidget,
)

from app.io.metavision_reader import MetavisionReader, is_metavision_raw
from app.simple_export import _build_pipeline, _parse_colour
from app.export import export_stream


class ExportWorker(QThread):
    """Background thread that runs the export to keep the UI responsive."""

    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        raw_path: str,
        out_path: str,
        codec: str,
        speed: float,
        fps: float,
        pos_colour: str | None,
        neg_colour: str | None,
    ) -> None:
        super().__init__()
        self.raw_path = raw_path
        self.out_path = out_path
        self.codec = codec
        self.speed = speed
        self.fps = fps
        self.pos_colour = pos_colour
        self.neg_colour = neg_colour

    def run(self) -> None:  # pragma: no cover - involves GUI/threading
        reader: MetavisionReader | None = None
        try:
            reader = MetavisionReader.from_raw(self.raw_path)
            dims = reader.ensure_sensor_size()
            if dims is None:
                raise RuntimeError("Sensorauflösung konnte nicht ermittelt werden.")
            width, height = dims
            pipeline = _build_pipeline(width, height)
            pipeline.pos_colour = _parse_colour(self.pos_colour, pipeline.pos_colour)
            pipeline.neg_colour = _parse_colour(self.neg_colour, pipeline.neg_colour)
            codec = "ffv1" if self.codec == "ffv1" else "mp4"
            duration_us = reader.metadata.duration_us
            if duration_us is None or duration_us <= 0:
                duration_us = reader.estimate_duration_us()
            if duration_us is None or duration_us <= 0:
                raise RuntimeError("Dauer des RAW-Streams konnte nicht bestimmt werden.")
            duration_s = duration_us / 1e6
            export_stream(
                reader,
                pipeline,
                duration_s,
                self.fps,
                self.out_path,
                codec,
                self.speed,
            )
        except Exception as exc:  # noqa: BLE001 - surface message to UI
            self.failed.emit(str(exc))
            return
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
        self.finished.emit(self.out_path)


class ExportWindow(QWidget):
    """Small form window to configure and run an export."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Simple Export GUI")
        self.worker: ExportWorker | None = None
        self._output_custom = False
        self._last_auto_output: str | None = None

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()

        input_btn = QPushButton("Öffnen…")
        input_btn.clicked.connect(self._select_input)
        output_btn = QPushButton("Speichern unter…")
        output_btn.clicked.connect(self._select_output)

        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["mp4", "ffv1"])

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 240.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setValue(60.0)

        self.pos_colour_edit = QLineEdit("#00ffaa")
        self.neg_colour_edit = QLineEdit("#ff3366")

        pos_colour_btn = QPushButton("Wählen…")
        pos_colour_btn.clicked.connect(lambda: self._choose_colour(self.pos_colour_edit))
        neg_colour_btn = QPushButton("Wählen…")
        neg_colour_btn.clicked.connect(lambda: self._choose_colour(self.neg_colour_edit))

        self.status_label = QLabel("Bereit")

        self.export_btn = QPushButton("Export starten")
        self.export_btn.clicked.connect(self._start_export)

        layout = QFormLayout()
        layout.addRow("RAW-Datei", self._with_button(self.input_edit, input_btn))
        layout.addRow("Ausgabedatei", self._with_button(self.output_edit, output_btn))
        layout.addRow("Codec", self.codec_combo)
        layout.addRow("Geschwindigkeit (x)", self.speed_spin)
        layout.addRow("FPS", self.fps_spin)
        layout.addRow("Pos. Farbe", self._with_button(self.pos_colour_edit, pos_colour_btn))
        layout.addRow("Neg. Farbe", self._with_button(self.neg_colour_edit, neg_colour_btn))
        layout.addRow(self.export_btn)
        layout.addRow(self.status_label)

        self.setLayout(layout)
        self.input_edit.textChanged.connect(self._on_input_changed)
        self.output_edit.textEdited.connect(self._on_output_edited)
        self.codec_combo.currentTextChanged.connect(self._on_codec_changed)

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

    def _select_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "RAW-Datei wählen", "", "RAW Dateien (*.raw)")
        if path:
            self.input_edit.setText(path)
            self._update_default_output()

    def _select_output(self) -> None:
        default_path = self._default_output_for_codec(self.codec_combo.currentText())
        filters = "MP4 Video (*.mp4);;Lossless FFV1 (*.mkv)"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Ausgabedatei wählen",
            default_path,
            filters,
        )
        if not path:
            return

        sel = (selected_filter or "").lower()
        if path.lower().endswith(".mkv") or "ffv1" in sel:
            codec = "ffv1"
        else:
            codec = "mp4"

        normalised = self._ensure_extension(path, codec)
        if codec != self.codec_combo.currentText():
            self.codec_combo.setCurrentText(codec)

        self.output_edit.setText(normalised)

        default_for_codec = self._default_output_for_codec(codec)
        if normalised == default_for_codec:
            self._last_auto_output = normalised
            self._output_custom = False
        else:
            self._last_auto_output = None
            self._output_custom = True

    def _start_export(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Fehlende Eingabe", "Bitte eine RAW-Datei auswählen.")
            return
        if not Path(input_path).is_file():
            QMessageBox.warning(self, "Datei nicht gefunden", "Die angegebene RAW-Datei existiert nicht.")
            return
        if not is_metavision_raw(input_path):
            QMessageBox.warning(self, "Ungültige Datei", "Nur Metavision RAW-Dateien werden unterstützt.")
            return
        if not output_path:
            QMessageBox.warning(self, "Fehlende Ausgabe", "Bitte einen Zielpfad angeben.")
            return

        speed = self.speed_spin.value()
        fps = self.fps_spin.value()
        pos_colour = self.pos_colour_edit.text().strip() or None
        neg_colour = self.neg_colour_edit.text().strip() or None
        codec = self.codec_combo.currentText()
        output_path = self._ensure_extension(output_path, codec)
        if output_path != self.output_edit.text().strip():
            self.output_edit.setText(output_path)
        if not self._output_custom:
            self._last_auto_output = output_path

        self.export_btn.setEnabled(False)
        self.status_label.setText("Export läuft…")

        worker = ExportWorker(
            input_path,
            output_path,
            codec,
            speed,
            fps,
            pos_colour,
            neg_colour,
        )
        worker.finished.connect(self._on_finished)
        worker.failed.connect(partial(self._on_failed, worker))
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)

        self.worker = worker
        worker.start()

    def _on_input_changed(self, _: str) -> None:
        self._update_default_output()

    def _on_output_edited(self, _: str) -> None:
        self._output_custom = True
        self._last_auto_output = None

    def _on_codec_changed(self, _: str) -> None:
        self._update_default_output()

    def _default_output_for_codec(self, codec: str) -> str:
        input_path = self.input_edit.text().strip()
        if not input_path:
            return ""
        suffix = ".mkv" if codec == "ffv1" else ".mp4"
        return str(Path(input_path).with_suffix(suffix))

    def _default_output_path(self) -> str:
        return self._default_output_for_codec(self.codec_combo.currentText())

    def _ensure_extension(self, path: str, codec: str) -> str:
        suffix = ".mkv" if codec == "ffv1" else ".mp4"
        p = Path(path)
        if p.suffix.lower() != suffix:
            p = p.with_suffix(suffix)
        return str(p)

    def _update_default_output(self) -> None:
        default = self._default_output_path()
        if not default:
            return
        current = self.output_edit.text().strip()
        if (not current) or (not self._output_custom) or current == (self._last_auto_output or ""):
            self.output_edit.setText(default)
            self._last_auto_output = default
            self._output_custom = False

    def _on_finished(self, out_path: str) -> None:
        self.worker = None
        self.status_label.setText(f"Fertig: {out_path}")
        self.export_btn.setEnabled(True)
        QMessageBox.information(self, "Export abgeschlossen", f"Video gespeichert unter:\n{out_path}")

    def _on_failed(self, worker: ExportWorker, message: str) -> None:
        if self.worker is worker:
            self.worker = None
        worker.deleteLater()
        self.status_label.setText("Fehler beim Export")
        self.export_btn.setEnabled(True)
        QMessageBox.critical(self, "Fehler", message)


def main() -> None:  # pragma: no cover - Qt entry point
    app = QApplication.instance() or QApplication([])
    window = ExportWindow()
    window.resize(480, 0)
    window.show()
    app.exec()


if __name__ == "__main__":  # pragma: no cover
    main()
