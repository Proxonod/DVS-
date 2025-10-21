"""Core pipeline for processing DVS event streams.

This module defines the :class:`Pipeline` class which maintains an
ordered list of filter instances and orchestrates the processing of
incoming event slices.  Each filter conforms to the interface defined
in :mod:`app.filters.base`; they may modify the event stream or
accumulate auxiliary state used for rendering.  The pipeline also
manages basic colour mapping for positive and negative polarities and
provides a simple method to composite the current state into an
RGB image suitable for display or export.

The implementation here is deliberately straightforward.  It does not
perform any multithreading or queueing - higher level components such
as the UI or export CLI are responsible for feeding event slices into
``process_events`` and retrieving the resulting frame via
``get_frame``.  Filters may maintain internal state between calls; the
pipeline will call ``reset`` on each filter whenever the sensor
geometry changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from app.filters.base import BaseFilter


def hex_to_rgb(col: str) -> Tuple[int, int, int]:
    """Convert a hex colour string (e.g. ``"#FF00AA"``) to an RGB tuple.

    Parameters
    ----------
    col:
        Hexadecimal string beginning with ``#``.  Only 6‑digit colours
        are supported.

    Returns
    -------
    tuple[int, int, int]
        Red, green and blue components in the range [0, 255].
    """
    col = col.lstrip("#")
    if len(col) != 6:
        raise ValueError(f"Invalid colour {col}: expected 6 hex digits")
    return tuple(int(col[i : i + 2], 16) for i in (0, 2, 4))


@dataclass
class Pipeline:
    """Pipeline managing a sequence of filters and rendering state.

    The pipeline holds a list of filter instances and applies them to
    incoming event arrays.  It also stores the current sensor
    dimensions and colour mapping for positive and negative events.
    """

    filters: List[BaseFilter] = field(default_factory=list)
    width: int = 0
    height: int = 0
    pos_colour: Tuple[int, int, int] = field(default_factory=lambda: hex_to_rgb("#00FFAA"))
    neg_colour: Tuple[int, int, int] = field(default_factory=lambda: hex_to_rgb("#FF3366"))
    _canvas: np.ndarray | None = field(default=None, init=False, repr=False)

    def reset(self, width: int, height: int) -> None:
        """Reset the pipeline for a new stream.

        This clears internal state and notifies each filter of the new
        sensor geometry via their ``reset`` method.  Colour mappings are
        left unchanged but may be modified through the public attributes
        ``pos_colour`` and ``neg_colour``.

        Parameters
        ----------
        width:
            Sensor width in pixels.
        height:
            Sensor height in pixels.
        """
        self.width = width
        self.height = height
        self._canvas = None
        for f in self.filters:
            # Many filters allocate arrays based on sensor dimensions.
            try:
                f.reset(width, height)
            except Exception:
                # Filters should handle errors internally; ignore
                pass

    # Filter management -----------------------------------------------------
    def add_filter(self, filt: BaseFilter) -> None:
        """Append a filter to the pipeline.

        The filter will be reset with the current sensor geometry.
        """
        self.filters.append(filt)
        try:
            filt.reset(self.width, self.height)
        except Exception:
            pass

    def remove_filter(self, filt: BaseFilter) -> None:
        """Remove a filter instance from the pipeline."""
        if filt in self.filters:
            self.filters.remove(filt)

    def clear_filters(self) -> None:
        """Remove all filters from the pipeline."""
        self.filters.clear()

    # Event processing ------------------------------------------------------
    def process_events(self, events: np.ndarray) -> dict:
        """Process a slice of events through the filter chain.

        Parameters
        ----------
        events:
            A NumPy structured array with fields ``x``, ``y``, ``t``
            (timestamp in microseconds) and ``p`` (polarity 0 or 1).

        Returns
        -------
        dict
            A state dictionary containing any auxiliary data provided by
            filters (e.g. time surfaces).  The final ``events`` field
            contains the event stream after all filters.
        """
        state = {}
        current_events = events
        for filt in self.filters:
            try:
                out = filt.process(current_events, state)
            except Exception:
                # Silently ignore exceptions in filters to avoid dropping
                # the entire pipeline; a robust implementation would log
                # and allow user control.
                out = None
            if isinstance(out, dict):
                events_out = out.get("events")
                if events_out is not None:
                    current_events = events_out
                if out:
                    for key, value in out.items():
                        if key != "events":
                            state[key] = value
        state["events"] = current_events
        return state

    # Rendering -------------------------------------------------------------
    def get_frame(self, state: dict) -> np.ndarray:
        """Compose an RGB frame from the current pipeline state.

        This method constructs an RGB image of size ``(height, width, 3)``
        using the filtered events and any auxiliary data present in
        ``state``.  The default implementation simply renders points
        for each event using the configured positive and negative
        colours.  More advanced renderers may override this method or
        interpret additional overlays (e.g. time surfaces) stored in
        ``state``.

        Parameters
        ----------
        state:
            The state dictionary returned by :meth:`process_events`.

        Returns
        -------
        numpy.ndarray
            A 3‑D array of unsigned 8‑bit integers representing an RGB
            image.
        """
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(
                "Pipeline dimensions are not set. Call reset() before rendering."
            )
        events = state.get("events")
        if events is None or len(events) == 0:
            canvas = self._ensure_canvas()
            canvas.fill(0)
            return canvas.astype(np.uint8)

        xs = events["x"].astype(np.intp, copy=False)
        ys = events["y"].astype(np.intp, copy=False)
        ps = events["p"].astype(np.uint8, copy=False)

        valid = (
            (xs >= 0)
            & (xs < self.width)
            & (ys >= 0)
            & (ys < self.height)
        )
        if not np.any(valid):
            canvas = self._ensure_canvas()
            canvas.fill(0)
            return canvas.astype(np.uint8)

        xs = xs[valid]
        ys = ys[valid]
        ps = ps[valid]

        canvas = self._ensure_canvas()
        canvas.fill(0)

        pos_colour = np.asarray(self.pos_colour, dtype=canvas.dtype)
        neg_colour = np.asarray(self.neg_colour, dtype=canvas.dtype)
        flat_size = self.width * self.height
        flat_indices = ys * self.width + xs

        pos_mask = ps == 1
        if np.any(pos_mask):
            counts = np.bincount(flat_indices[pos_mask], minlength=flat_size)
            if counts.any():
                counts = counts.astype(canvas.dtype, copy=False).reshape(self.height, self.width)
                canvas += counts[..., None] * pos_colour

        neg_mask = ps == 0
        if np.any(neg_mask):
            counts = np.bincount(flat_indices[neg_mask], minlength=flat_size)
            if counts.any():
                counts = counts.astype(canvas.dtype, copy=False).reshape(self.height, self.width)
                canvas += counts[..., None] * neg_colour

        np.clip(canvas, 0, 255, out=canvas)
        return canvas.astype(np.uint8)

    def _ensure_canvas(self) -> np.ndarray:
        if self._canvas is None or self._canvas.shape != (self.height, self.width, 3):
            self._canvas = np.zeros((self.height, self.width, 3), dtype=np.uint32)
        return self._canvas
