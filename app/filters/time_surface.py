"""Time surface filter.

This filter constructs a time surface representation of recent events.
At each call to :meth:`process`, it decays the stored intensity
according to an exponential kernel and then boosts the intensity at
pixels where new events occurred.  The result is two floating point
images (for positive and negative polarities) that represent how
recently each pixel fired.  These images may be used to render
fading trails or heatmaps.

The decay is governed by ``tau_ms``, the time constant in
milliseconds.  Intensities are decayed by a factor of
``exp(-dt / tau)`` between successive calls, where ``dt`` is the
elapsed time in microseconds.  Setting ``polarity_separate`` to
``False`` causes both positive and negative events to be accumulated
in the same surface.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseFilter


class TimeSurfaceFilter(BaseFilter):
    """Compute exponentially decayed time surfaces for recent events."""

    name = "TimeSurface"

    def __init__(self, tau_ms: float = 50.0, polarity_separate: bool = True) -> None:
        # Time constant in microseconds
        self.tau_us = float(tau_ms) * 1000.0
        self.polarity_separate = bool(polarity_separate)
        self.surface_pos: np.ndarray
        self.surface_neg: np.ndarray
        self.last_timestamp: float = 0.0
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, object]:
        return {"tau_ms": self.tau_us / 1000.0, "polarity_separate": self.polarity_separate}

    def set_params(self, **kwargs: object) -> None:
        if "tau_ms" in kwargs:
            self.tau_us = float(kwargs["tau_ms"]) * 1000.0
        if "polarity_separate" in kwargs:
            self.polarity_separate = bool(kwargs["polarity_separate"])

    def reset(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # Allocate surfaces; use float32 for efficiency
        self.surface_pos = np.zeros((height, width), dtype=np.float32)
        self.surface_neg = np.zeros((height, width), dtype=np.float32)
        self.last_timestamp = 0.0

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if events.size == 0:
            # Even without events, we still decay surfaces by an arbitrary small amount
            return {}
        # Determine time difference since last update
        current_time = float(events["t"][-1])  # use last event timestamp
        if self.last_timestamp == 0.0:
            dt = 0.0
        else:
            dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        if dt < 0:
            dt = 0.0
        # Compute decay factor; protect against overflow when dt is large
        if self.tau_us > 0.0:
            decay = np.exp(-dt / self.tau_us)
        else:
            decay = 0.0
        # Apply decay to surfaces
        self.surface_pos *= decay
        self.surface_neg *= decay
        # Update surfaces with new events
        xs = events["x"].astype(np.int64)
        ys = events["y"].astype(np.int64)
        ps = events["p"].astype(np.int64)
        if self.polarity_separate:
            for i in range(len(events)):
                x = xs[i]
                y = ys[i]
                if ps[i] == 1:
                    self.surface_pos[y, x] = 1.0
                else:
                    self.surface_neg[y, x] = 1.0
        else:
            for i in range(len(events)):
                x = xs[i]
                y = ys[i]
                # Combine both polarities into the positive surface
                self.surface_pos[y, x] = 1.0
        # Store surfaces into state for downstream consumers
        # We copy the surfaces to avoid accidental mutation by the caller
        # Note: copying floats to Python overhead is acceptable for small images but
        # could be optimised by returning a view; the pipeline should treat
        # these as read‑only.
        state["time_surface_pos"] = self.surface_pos.copy()
        state["time_surface_neg"] = self.surface_neg.copy()
        return {}
