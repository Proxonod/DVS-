"""Metavision RAW and camera reader with metadata helpers.

This module wraps Metavision's EventsIterator to provide a simple
iterator interface and basic metadata (width, height, duration).
If the SDK does not expose dimensions or duration, the reader can
infer width/height from events and estimate duration by scanning once.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

# Try to import Metavision SDK
try:
    from metavision_core.event_io import EventsIterator  # type: ignore
except Exception:
    EventsIterator = None  # type: ignore


@dataclass
class StreamMetadata:
    width: Optional[int] = None
    height: Optional[int] = None
    duration_us: Optional[int] = None
    has_polarity: bool = True


def is_metavision_raw(path: str) -> bool:
    return path.lower().endswith(".raw")


def is_metavision_bias(path: str) -> bool:
    return path.lower().endswith(".bias")


def default_sample_path() -> str:
    return "samples/recording_2025-10-17_18-28-11.raw"


class MetavisionReader:
    """Uniform reader for RAW files or live cameras."""

    def __init__(self, iterator, metadata: StreamMetadata) -> None:
        # EventsIterator is iterable, wrap into a real iterator
        self._iterator_source = iterator
        try:
            self._iterator = iter(iterator)
        except TypeError:
            self._iterator = iterator
        self.metadata = metadata
        self._done = False
        self._source_path: Optional[str] = None  # set for RAW
        self.delta_t_us: int = 5000  # default; constructors override
        # Prefetched chunks that were consumed for metadata inspection.
        # They are replayed before pulling new data from the underlying iterator
        # so no events are lost when we probe the stream for dimensions.
        self._prefetched: Deque[np.ndarray] = deque()

    # -------- constructors --------

    @classmethod
    def from_raw(cls, path: str, delta_t_us: int = 5000) -> "MetavisionReader":
        if EventsIterator is None:
            raise RuntimeError("Metavision SDK not available. Please install the Metavision SDK.")

        mv_it = EventsIterator(path, mode="delta_t", delta_t=delta_t_us)

        # width/height best effort
        width = None
        height = None
        for getter in ("get_sensor_width", "width"):
            try:
                val = getattr(mv_it, getter)
                width = int(val() if callable(val) else val)
                break
            except Exception:
                pass
        for getter in ("get_sensor_height", "height"):
            try:
                val = getattr(mv_it, getter)
                height = int(val() if callable(val) else val)
                break
            except Exception:
                pass

        # duration best effort
        duration_us = None
        for getter in ("get_duration", "get_seek_end_time", "duration_us"):
            try:
                val = getattr(mv_it, getter)
                duration_us = int(val() if callable(val) else val)
                break
            except Exception:
                pass

        md = StreamMetadata(width=width, height=height, duration_us=duration_us, has_polarity=True)
        rdr = cls(mv_it, md)
        rdr._source_path = path
        rdr.delta_t_us = int(delta_t_us)
        return rdr

    @classmethod
    def from_camera(cls) -> "MetavisionReader":
        if EventsIterator is None:
            raise RuntimeError("Metavision SDK not available. Please install the Metavision SDK.")
        mv_it = EventsIterator("", mode="delta_t", delta_t=5000)  # empty path means live device
        width = None
        height = None
        try:
            width = int(getattr(mv_it, "get_sensor_width")())
            height = int(getattr(mv_it, "get_sensor_height")())
        except Exception:
            pass
        md = StreamMetadata(width=width, height=height, duration_us=None, has_polarity=True)
        rdr = cls(mv_it, md)
        rdr.delta_t_us = 5000
        return rdr

    # -------- iterator protocol --------

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._done:
            raise StopIteration
        if self._prefetched:
            return self._prefetched.popleft()
        try:
            events = next(self._iterator)
        except StopIteration:
            self._done = True
            raise
        return events

    # -------- helpers --------

    def close(self) -> None:
        try:
            it = getattr(self, "_iterator", None)
            if hasattr(it, "close"):
                it.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    def ensure_sensor_size(self, max_chunks: int = 1000) -> Optional[tuple[int, int]]:
        """Ensure ``metadata.width``/``height`` are populated.

        When RAW metadata does not expose the sensor geometry the export
        pipeline crashes later when attempting to convert ``None`` to
        integers.  To avoid this we try to recover the width/height by
        peeking into a couple of event chunks from the existing iterator.
        The consumed chunks are queued internally so callers still receive
        the full stream.
        """

        width = self.metadata.width
        height = self.metadata.height
        if width is not None and height is not None:
            return int(width), int(height)

        width_guess = width
        height_guess = height
        source = getattr(self, "_iterator_source", None)
        if source is not None:
            for getter in ("get_sensor_width", "width"):
                if width_guess is not None:
                    break
                try:
                    val = getattr(source, getter)
                except AttributeError:
                    continue
                try:
                    width_guess = int(val() if callable(val) else val)
                except Exception:
                    pass
            for getter in ("get_sensor_height", "height"):
                if height_guess is not None:
                    break
                try:
                    val = getattr(source, getter)
                except AttributeError:
                    continue
                try:
                    height_guess = int(val() if callable(val) else val)
                except Exception:
                    pass
        inspected = 0
        while inspected < max_chunks:
            if width_guess is not None and height_guess is not None:
                break
            try:
                events = next(self._iterator)
            except StopIteration:
                break
            self._prefetched.append(events)
            inspected += 1
            if events.size == 0:
                continue
            dtype_names = getattr(events.dtype, "names", None)
            if width_guess is None and dtype_names and "x" in dtype_names:
                try:
                    width_guess = int(events["x"].max()) + 1
                except Exception:
                    pass
            if height_guess is None and dtype_names and "y" in dtype_names:
                try:
                    height_guess = int(events["y"].max()) + 1
                except Exception:
                    pass

        if width_guess is not None and height_guess is not None:
            self.metadata.width = int(width_guess)
            self.metadata.height = int(height_guess)
            return int(width_guess), int(height_guess)

        return None

    def estimate_duration_us(self, max_chunks: int = 500000, delta_t_us: int = 100000) -> Optional[int]:
        """Estimate duration by scanning to the end once using a fresh iterator."""
        if self._source_path is None or EventsIterator is None:
            return None
        try:
            it = EventsIterator(self._source_path, mode="delta_t", delta_t=delta_t_us)
        except Exception:
            return None
        last_t: Optional[int] = None
        try:
            it2 = iter(it)
            for _ in range(max_chunks):
                try:
                    ev = next(it2)
                except StopIteration:
                    break
                if ev.size:
                    try:
                        last_t = int(ev["t"][-1])
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if hasattr(it, "close"):
                it.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        if last_t is not None:
            self.metadata.duration_us = last_t
        return last_t
