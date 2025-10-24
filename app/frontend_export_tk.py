"""Tkinter-based GUI for the ``simple_export`` pipeline.

This module mirrors :mod:`app.frontend_export` but avoids the dependency
on PySide6 by building the interface with the standard-library
:mod:`tkinter` package.  The feature set is intentionally similar so that
users can pick whichever frontend suits their environment best.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox
from typing import Callable

from app.export import export_stream
from app.io.metavision_reader import MetavisionReader, is_metavision_raw
from app.simple_export import _build_pipeline, _parse_colour


class ExportWorker(threading.Thread):
    """Background worker that performs the export pipeline."""

    def __init__(
        self,
        tk_root: tk.Tk,
        raw_path: str,
        out_path: str,
        codec: str,
        speed: float,
        fps: float,
        pos_colour: str | None,
        neg_colour: str | None,
        on_finished: Callable[[str], None],
        on_failed: Callable[[str], None],
    ) -> None:
        super().__init__(daemon=True)
        self.root = tk_root
        self.raw_path = raw_path
        self.out_path = out_path
        self.codec = codec
        self.speed = speed
        self.fps = fps
        self.pos_colour = pos_colour
        self.neg_colour = neg_colour
        self.on_finished = on_finished
        self.on_failed = on_failed

    def _dispatch(self, callback: Callable[[str], None], arg: str) -> None:
        self.root.after(0, lambda: callback(arg))

    def run(self) -> None:  # pragma: no cover - involves threading/UI
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
        except Exception as exc:  # noqa: BLE001 - propagate to UI
            self._dispatch(self.on_failed, str(exc))
            return
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:  # pragma: no cover - best effort
                    pass
        self._dispatch(self.on_finished, self.out_path)


class ExportWindow(tk.Frame):
    """Simple Tkinter form that collects export parameters."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.master.title("Simple Export GUI (Tkinter)")

        self.worker: ExportWorker | None = None
        self._output_custom = False
        self._last_auto_output: str | None = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.codec_var = tk.StringVar(value="mp4")
        self.speed_var = tk.DoubleVar(value=1.0)
        self.fps_var = tk.DoubleVar(value=60.0)
        self.pos_colour_var = tk.StringVar(value="#00ffaa")
        self.neg_colour_var = tk.StringVar(value="#ff3366")

        self._build_form()
        self.pack(padx=16, pady=16, fill=tk.BOTH, expand=True)

        self.input_var.trace_add("write", lambda *_: self._update_default_output())
        self.codec_var.trace_add("write", lambda *_: self._update_default_output())

    def _build_form(self) -> None:
        grid = tk.Frame(self)
        grid.pack(fill=tk.BOTH, expand=True)

        def add_row(row: int, label: str, widget: tk.Widget) -> None:
            tk.Label(grid, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
            widget.grid(row=row, column=1, sticky=tk.EW, pady=4)

        grid.columnconfigure(1, weight=1)

        input_entry = tk.Entry(grid, textvariable=self.input_var)
        input_btn = tk.Button(grid, text="Öffnen…", command=self._select_input)
        add_row(0, "RAW-Datei", input_entry)
        input_btn.grid(row=0, column=2, padx=(6, 0))

        output_entry = tk.Entry(grid, textvariable=self.output_var)
        output_entry.bind("<KeyRelease>", self._on_output_edited)
        output_btn = tk.Button(grid, text="Speichern unter…", command=self._select_output)
        add_row(1, "Ausgabedatei", output_entry)
        output_btn.grid(row=1, column=2, padx=(6, 0))

        codec_menu = tk.OptionMenu(grid, self.codec_var, "mp4", "ffv1")
        add_row(2, "Codec", codec_menu)

        speed_spin = tk.Spinbox(
            grid,
            textvariable=self.speed_var,
            from_=0.1,
            to=4.0,
            increment=0.1,
            format="%.2f",
        )
        add_row(3, "Geschwindigkeit (x)", speed_spin)

        fps_spin = tk.Spinbox(
            grid,
            textvariable=self.fps_var,
            from_=1.0,
            to=240.0,
            increment=1.0,
            format="%.1f",
        )
        add_row(4, "FPS", fps_spin)

        pos_entry = tk.Entry(grid, textvariable=self.pos_colour_var)
        pos_btn = tk.Button(grid, text="Wählen…", command=lambda: self._choose_colour(self.pos_colour_var))
        add_row(5, "Pos. Farbe", pos_entry)
        pos_btn.grid(row=5, column=2, padx=(6, 0))

        neg_entry = tk.Entry(grid, textvariable=self.neg_colour_var)
        neg_btn = tk.Button(grid, text="Wählen…", command=lambda: self._choose_colour(self.neg_colour_var))
        add_row(6, "Neg. Farbe", neg_entry)
        neg_btn.grid(row=6, column=2, padx=(6, 0))

        self.status_label = tk.Label(grid, text="Bereit")
        self.status_label.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(12, 0))

        self.export_btn = tk.Button(grid, text="Export starten", command=self._start_export)
        self.export_btn.grid(row=8, column=0, columnspan=3, pady=(12, 0), sticky=tk.EW)

    def _choose_colour(self, var: tk.StringVar) -> None:
        initial = var.get() or "#ffffff"
        colour, _ = colorchooser.askcolor(color=initial, parent=self.master)
        if colour:
            # ``askcolor`` returns RGB tuple; convert to hex string.
            r, g, b = map(int, colour)
            var.set(f"#{r:02x}{g:02x}{b:02x}")

    def _select_input(self) -> None:
        path = filedialog.askopenfilename(
            title="RAW-Datei wählen",
            filetypes=[("RAW Dateien", "*.raw"), ("Alle Dateien", "*.*")],
        )
        if path:
            self.input_var.set(path)

    def _select_output(self) -> None:
        default = self._default_output_for_codec(self.codec_var.get())
        path = filedialog.asksaveasfilename(
            title="Ausgabedatei wählen",
            initialfile=self._initial_filename(default),
            defaultextension=".mp4" if self.codec_var.get() == "mp4" else ".mkv",
            filetypes=[("MP4 Video", "*.mp4"), ("Lossless FFV1", "*.mkv"), ("Alle Dateien", "*.*")],
        )
        if not path:
            return
        codec = "ffv1" if path.lower().endswith(".mkv") else self.codec_var.get()
        normalised = self._ensure_extension(path, codec)
        if codec != self.codec_var.get():
            self.codec_var.set(codec)
        self.output_var.set(normalised)
        default_for_codec = self._default_output_for_codec(codec)
        if normalised == default_for_codec:
            self._last_auto_output = normalised
            self._output_custom = False
        else:
            self._last_auto_output = None
            self._output_custom = True

    def _initial_filename(self, path: str) -> str:
        return path.split("/")[-1] if path else ""

    def _on_output_edited(self, _: tk.Event) -> None:
        self._output_custom = True
        self._last_auto_output = None

    def _start_export(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip()
        if not input_path:
            messagebox.showwarning("Fehlende Eingabe", "Bitte eine RAW-Datei auswählen.")
            return
        if not Path(input_path).is_file():
            messagebox.showwarning(
                "Datei nicht gefunden",
                "Die angegebene RAW-Datei existiert nicht.",
            )
            return
        if not is_metavision_raw(input_path):
            messagebox.showwarning("Ungültige Datei", "Nur Metavision RAW-Dateien werden unterstützt.")
            return
        if not output_path:
            messagebox.showwarning("Fehlende Ausgabe", "Bitte einen Zielpfad angeben.")
            return

        codec = self.codec_var.get()
        output_path = self._ensure_extension(output_path, codec)
        if output_path != self.output_var.get().strip():
            self.output_var.set(output_path)
        if not self._output_custom:
            self._last_auto_output = output_path

        speed = float(self.speed_var.get())
        fps = float(self.fps_var.get())
        pos_colour = self.pos_colour_var.get().strip() or None
        neg_colour = self.neg_colour_var.get().strip() or None

        self.export_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Export läuft…")

        self.worker = ExportWorker(
            self.master,
            input_path,
            output_path,
            codec,
            speed,
            fps,
            pos_colour,
            neg_colour,
            on_finished=self._on_finished,
            on_failed=self._on_failed,
        )
        self.worker.start()

    def _default_output_for_codec(self, codec: str) -> str:
        input_path = self.input_var.get().strip()
        if not input_path:
            return ""
        suffix = ".mkv" if codec == "ffv1" else ".mp4"
        return str(Path(input_path).with_suffix(suffix))

    def _ensure_extension(self, path: str, codec: str) -> str:
        suffix = ".mkv" if codec == "ffv1" else ".mp4"
        p = Path(path)
        if p.suffix.lower() != suffix:
            p = p.with_suffix(suffix)
        return str(p)

    def _update_default_output(self) -> None:
        default = self._default_output_for_codec(self.codec_var.get())
        if not default:
            return
        current = self.output_var.get().strip()
        if (not current) or (not self._output_custom) or current == (self._last_auto_output or ""):
            self.output_var.set(default)
            self._last_auto_output = default
            self._output_custom = False

    def _on_finished(self, out_path: str) -> None:
        self.worker = None
        self.export_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Fertig: {out_path}")
        messagebox.showinfo("Export abgeschlossen", f"Video gespeichert unter:\n{out_path}")

    def _on_failed(self, message: str) -> None:
        self.worker = None
        self.export_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Fehler beim Export")
        messagebox.showerror("Fehler", message)


def main() -> None:  # pragma: no cover - Tkinter entry point
    root = tk.Tk()
    ExportWindow(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
