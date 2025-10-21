"""Time surface filter with optional CUDA acceleration.

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

When CuPy is available, the filter can keep its internal state on the
GPU and use CUDA kernels for the decay and update operations.  This is
particularly beneficial for high‑resolution sensors where the per-pixel
operations become a bottleneck on the CPU.  GPU mode can be enabled via
``use_cuda=True`` or by setting the environment variable
``DVS_USE_CUDA=1``.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

try:  # pragma: no cover - optional dependency
    import cupy as cp
except Exception:  # pragma: no cover - no CUDA runtime in tests
    cp = None

from app.filters.base import BaseFilter


class TimeSurfaceFilter(BaseFilter):
    """Compute exponentially decayed time surfaces for recent events."""

    name = "TimeSurface"

    def __init__(
        self,
        tau_ms: float = 50.0,
        polarity_separate: bool = True,
        use_cuda: bool | None = None,
    ) -> None:
        # Time constant in microseconds
        self.tau_us = float(tau_ms) * 1000.0
        self.polarity_separate = bool(polarity_separate)
        self._prefer_cuda = self._env_wants_cuda() if use_cuda is None else bool(use_cuda)
        self._xp = np
        self._use_cuda = False
        self.surface_pos: np.ndarray | "cp.ndarray"
        self.surface_neg: np.ndarray | "cp.ndarray"
        self.last_timestamp: float | None = None
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, object]:
        return {
            "tau_ms": self.tau_us / 1000.0,
            "polarity_separate": self.polarity_separate,
            "device": "cuda" if self._use_cuda else "cpu",
        }

    def set_params(self, **kwargs: object) -> None:
        if "tau_ms" in kwargs:
            self.tau_us = float(kwargs["tau_ms"]) * 1000.0
        if "polarity_separate" in kwargs:
            self.polarity_separate = bool(kwargs["polarity_separate"])
        if "use_cuda" in kwargs:
            self._prefer_cuda = bool(kwargs["use_cuda"])
            self._configure_backend()
            # Changing device requires reallocation on next reset
            if self.width and self.height:
                self.reset(self.width, self.height)

    def reset(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._configure_backend()
        xp = self._xp
        # Allocate surfaces; use float32 for efficiency
        self.surface_pos = xp.zeros((height, width), dtype=xp.float32)
        self.surface_neg = xp.zeros((height, width), dtype=xp.float32)
        self.last_timestamp = None

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if events.size == 0:
            # Even without events, we still decay surfaces by an arbitrary small amount
            return {}
        # Determine time difference since last update
        current_time = float(events["t"][-1])  # use last event timestamp
        if self.last_timestamp is None:
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

        if self._use_cuda:
            xp = self._xp
            xs = xp.asarray(events["x"], dtype=xp.int32)
            ys = xp.asarray(events["y"], dtype=xp.int32)
            ps = xp.asarray(events["p"], dtype=xp.int8)
            valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
            if xp.any(valid):
                xs = xs[valid]
                ys = ys[valid]
                ps = ps[valid]
                if self.polarity_separate:
                    pos_mask = ps == 1
                    neg_mask = ps == 0
                    if xp.any(pos_mask):
                        self.surface_pos[ys[pos_mask], xs[pos_mask]] = 1.0
                    if xp.any(neg_mask):
                        self.surface_neg[ys[neg_mask], xs[neg_mask]] = 1.0
                else:
                    if xs.size:
                        self.surface_pos[ys, xs] = 1.0
        else:
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
        state["time_surface_pos"] = self._to_numpy(self.surface_pos)
        state["time_surface_neg"] = self._to_numpy(self.surface_neg)
        return {}

    @staticmethod
    def _env_wants_cuda() -> bool:
        flag = os.environ.get("DVS_USE_CUDA", "")
        return flag.lower() in {"1", "true", "yes", "on"}

    def _configure_backend(self) -> None:
        if self._prefer_cuda and cp is not None:
            self._xp = cp
            self._use_cuda = True
        else:
            self._xp = np
            self._use_cuda = False

    def _to_numpy(self, arr):
        if self._use_cuda:
            return cp.asnumpy(arr)
        return np.array(arr, copy=True)

    @property
    def device(self) -> str:
        """Return the device that backs the internal buffers."""

        return "cuda" if self._use_cuda else "cpu"
