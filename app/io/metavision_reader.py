"""Utilities for reading Metavision RAW files and interfacing with live devices.

This module wraps the Metavision SDK (when installed) to provide a uniform
API for retrieving event slices from files or cameras. It includes helper
functions to detect file types and locate sample recordings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Iterator

import numpy as np


def is_metavision_raw(path: str) -> bool:
    return path.lower().endswith(".raw")


def is_metavision_bias(path: str) -> bool:
    return path.lower().endswith(".bias")


def default_sample_path() -> str:
    return "samples/recording_2025-10-17_18-28-11.raw"


@dataclass
class StreamMetadata:
    width: Optional[int] = None
    height: Optional[int] = None
    duration_us: Optional[int] = None
    event_count: Optional[int] = None
    has_polarity: Optional[bool] = True

class MetavisionReader:
    def __init__(self, iterator, metadata):
        try:
            self._iterator = iter(iterator)
        except TypeError:
            self._iterator = iterator
        self.metadata = metadata
        self._done = False
        self._source_path = None
        self.delta_t_us = 5000  # default, will be overwritten by constructors


    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._done:
            raise StopIteration
        try:
            events = next(self._iterator)
        except StopIteration:
            self._done = True
            raise
        return events

    def close(self) -> None:
        try:
            # If underlying iterator exposes close(), call it
            close = getattr(self._iterator, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    # ---- Factories ----
    @classmethod
    def from_raw(cls, path: str) -> "MetavisionReader":
        """Open a RAW file as an iterator of event slices."""

        mv_it = EventsIterator(path, mode="delta_t", delta_t=delta_t_us)
        try:
            from metavision_core.event_io import EventsIterator  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Metavision SDK not available. Install the Prophesee SDK to read .raw files."
            ) from exc

        # 5 ms chunks for responsive UI
        mv_iter = EventsIterator(path, mode="delta_t", delta_t=5000)

        width, height = _try_get_sensor_size(mv_iter)
        duration_us = _try_get_duration_us(mv_iter)

        md = StreamMetadata(width=width or None, height=height or None, duration_us=duration_us)
        return cls(mv_iter, md)

    @classmethod
    def from_camera(cls) -> "MetavisionReader":
        """Open a live camera (if available)."""
        try:
            from metavision_core.event_io import EventsIterator  # type: ignore
        except Exception as exc:
            raise RuntimeError("Metavision SDK not available for camera.") from exc

        mv_iter = EventsIterator("")  # empty string selects default camera
        width, height = _try_get_sensor_size(mv_iter)
        duration_us = None

        md = StreamMetadata(width=width or None, height=height or None, duration_us=duration_us)
        return cls(mv_iter, md)


# ---- helpers ----

def _try_get_sensor_size(mv_iter) -> tuple[int | None, int | None]:
    """Try multiple SDK accessors to get width/height."""
    for w_attr, h_attr in [
        ("get_sensor_width", "get_sensor_height"),
        ("width", "height"),
        ("get_width", "get_height"),
    ]:
        try:
            w_fun = getattr(mv_iter, w_attr, None)
            h_fun = getattr(mv_iter, h_attr, None)
            w = w_fun() if callable(w_fun) else (w_fun if isinstance(w_fun, int) else None)
            h = h_fun() if callable(h_fun) else (h_fun if isinstance(h_fun, int) else None)
            if w and h and w > 0 and h > 0:
                return int(w), int(h)
        except Exception:
            pass
    return None, None


def _try_get_duration_us(mv_iter) -> int | None:
    """Try to obtain total duration from various SDKs."""
    for attr in ("get_seek_end_time", "get_duration", "duration_us", "duration"):
        try:
            fn = getattr(mv_iter, attr, None)
            if callable(fn):
                val = fn()
            else:
                val = fn
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
        except Exception:
            continue
    return None
