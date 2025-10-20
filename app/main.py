"""Entry point for the DVS viewer GUI.

This module starts a Qt application, opens an initial RAW recording if
available and instantiates the main window.  It also defines some helper
functions to detect sample recordings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .io.metavision_reader import default_sample_path


def main() -> None:
    """Start the GUI application.

    If a default sample exists in the ``samples`` directory, it will be
    automatically opened and paused at the beginning of the stream.  The
    function sets up the Qt event loop and shows the main window.
    """
    app = QApplication(sys.argv)

    sample = default_sample_path()
    initial_file = sample if Path(sample).exists() else None

    window = MainWindow(initial_file)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()