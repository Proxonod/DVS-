"""Top‑level package for the DVS viewer application.

This package contains all modules for loading, processing and displaying
event streams from Prophesee sensors.  Subpackages include:

* **ui** - PySide6 widgets and windows
* **io** - classes for reading RAW files and interacting with live cameras
* **render** - conversion of events into images and OpenGL backends
* **filters** - implementation of real-time filters
* **core** - pipeline orchestration, state management and presets
* **utils** - logging and profiling utilities
"""

__all__ = [
    "main",
]