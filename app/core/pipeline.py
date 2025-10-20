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

from ..filters.base import BaseFilter


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
                out = {}
            # Allow filters to modify the event stream
            if isinstance(out, dict) and "events" in out:
                current_events = out["events"]
            # Merge any auxiliary outputs into state
            if isinstance(out, dict):
                for k, v in out.items():
                    if k == "events":
                        continue
                    state[k] = v
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
        # Start with a black frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        events = state.get("events")
        if events is None or len(events) == 0:
            return frame
        # Draw positive events
        pos_mask = events["p"] == 1
        if np.any(pos_mask):
            ys = events["y"][pos_mask].astype(np.int32)
            xs = events["x"][pos_mask].astype(np.int32)
            frame[ys, xs, 0] = self.pos_colour[0]
            frame[ys, xs, 1] = self.pos_colour[1]
            frame[ys, xs, 2] = self.pos_colour[2]
        # Draw negative events
        neg_mask = events["p"] == 0
        if np.any(neg_mask):
            ys = events["y"][neg_mask].astype(np.int32)
            xs = events["x"][neg_mask].astype(np.int32)
            frame[ys, xs, 0] = self.neg_colour[0]
            frame[ys, xs, 1] = self.neg_colour[1]
            frame[ys, xs, 2] = self.neg_colour[2]
        return frame
