# DVS Viewer

This project provides a desktop application for browsing and filtering recordings captured with a Dynamic Vision Sensor (DVS) camera. The GUI is built with PySide6 and relies on Prophesee's Metavision SDK to read event streams.

## Prerequisites

- **Metavision SDK** from Prophesee (install separately before setting up this project)
- Python 3.10 or newer
- A C++17-capable compiler (only needed when you build Metavision-native extensions)
- FFmpeg if you plan to export videos

## Installation

1. Install the Metavision SDK according to the official Prophesee documentation and make sure the required binaries and environment variables are available in your shell.
2. Clone this repository and move into the project directory.
3. (Optional) Create and activate a Python virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

4. Install the Python dependencies:

   ```bash
   pip install -e app[dev]
   ```

   The package metadata in `app/pyproject.toml` lists all runtime and development dependencies, so no additional `requirements.txt` is necessary. For a minimal runtime-only installation you can run `pip install -e app`.

## Usage

Start the Qt-based viewer after installation with:

```bash
python run.py
```

On start-up the application tries to load a sample RAW file from `app/samples`. If a sample exists it is opened automatically; otherwise the viewer starts without content and you can pick your own recordings via the file dialog.

## Export Utilities

In addition to the main viewer the repository ships several export helpers that all reuse the same rendering pipeline:

- `app/frontend_export.py` – a PySide6 GUI for exporting a single RAW file. It exposes every command-line parameter (codec, duration, FPS, colours) as form fields and runs the export in a background thread to keep the interface responsive.
- `app/simple_export.py` – a command-line variant that accepts `--input`, `--out`, `--codec`, `--duration`, `--fps`, `--pos-colour`, and `--neg-colour`. Any parameter left out falls back to a sensible default and the script interactively asks for the RAW path when `--input` is omitted.
- `app/frontend_export_batch.py` – a PySide6 batch exporter that processes every RAW file in a directory sequentially. Use this when you need to convert large collections without launching the main viewer.

If PySide6 causes trouble on your platform, `app/frontend_export_tk.py` provides the same single-file exporter using the standard-library `tkinter` toolkit.

## Tests

Run the available unit tests from the `app` directory with `pytest`:

```bash
cd app
pytest
```

More details about the project structure can be found directly in the source code under `app/`.
