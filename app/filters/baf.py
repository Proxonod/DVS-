"""Background Activity Filter (BAF)."""

from __future__ import annotations
from typing import Dict
import numpy as np
from app.filters.base import BaseFilter

try:
    from numba import njit
except Exception:
    njit = None  # type: ignore


class BackgroundActivityFilter(BaseFilter):
    """Remove isolated noise events by enforcing local activity."""

    name = "BackgroundActivityFilter"

    def __init__(
        self,
        window_ms: float = 50.0,
        count_threshold: int = 1,
        refractory_us: int = 500,
        spatial_radius: int = 1,
    ) -> None:
        self.window_us = int(window_ms * 1000)
        self.count_threshold = max(1, int(count_threshold))
        self.refractory_us = int(refractory_us)
        self.spatial_radius = max(0, int(spatial_radius))
        self.last_times: np.ndarray
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, object]:
        return {
            "window_ms": self.window_us / 1000,
            "count_threshold": self.count_threshold,
            "refractory_us": self.refractory_us,
            "spatial_radius": self.spatial_radius,
        }

    def set_params(self, **kwargs: object) -> None:
        if "window_ms" in kwargs:
            self.window_us = int(float(kwargs["window_ms"]) * 1000)
        if "count_threshold" in kwargs:
            self.count_threshold = max(1, int(kwargs["count_threshold"]))
        if "refractory_us" in kwargs:
            self.refractory_us = int(kwargs["refractory_us"])
        if "spatial_radius" in kwargs:
            self.spatial_radius = max(0, int(kwargs["spatial_radius"]))

    def reset(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        safe_neg = -int(max(self.window_us, self.refractory_us) + 1)
        self.last_times = np.full((height, width), safe_neg, dtype=np.int64)

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if events.size == 0:
            return {"events": events}
        if njit is not None:
            mask = _baf_kernel_numba(
                events["x"],
                events["y"],
                events["t"],
                self.last_times,
                self.window_us,
                self.refractory_us,
                self.count_threshold,
                self.spatial_radius,
            )
        else:
            mask = _baf_kernel_python(
                events["x"],
                events["y"],
                events["t"],
                self.last_times,
                self.window_us,
                self.refractory_us,
                self.count_threshold,
                self.spatial_radius,
            )
        return {"events": events[mask]}


def _baf_kernel_python(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    last_times: np.ndarray,
    window_us: int,
    refractory_us: int,
    count_threshold: int,
    spatial_radius: int,
) -> np.ndarray:
    n = len(ts)
    mask = np.zeros(n, dtype=bool)
    height, width = last_times.shape
    for i in range(n):
        x = int(xs[i]); y = int(ys[i]); t = int(ts[i])
        if not (0 <= x < width and 0 <= y < height):
            continue
        # per-pixel refractory
        last_val = float(last_times[y, x])
        if t - last_val < refractory_us:
            # still update history to reflect activity
            last_times[y, x] = float(t)
            continue
        x0 = max(0, x - spatial_radius); y0 = max(0, y - spatial_radius)
        x1 = min(width - 1, x + spatial_radius); y1 = min(height - 1, y + spatial_radius)
        count = 0
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if t - float(last_times[yy, xx]) <= window_us:
                    count += 1
                    if count >= count_threshold:
                        break
            if count >= count_threshold:
                break
        if count >= count_threshold:
            mask[i] = True
        # Always update history
        last_times[y, x] = float(t)
    return mask


if njit is not None:

    @njit
    def _baf_kernel_numba(
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        last_times: np.ndarray,
        window_us: int,
        refractory_us: int,
        count_threshold: int,
        spatial_radius: int,
    ) -> np.ndarray:
        n = ts.shape[0]
        mask = np.zeros(n, dtype=np.bool_)
        height, width = last_times.shape
        for i in range(n):
            x = int(xs[i]); y = int(ys[i]); t = int(ts[i])
            if not (0 <= x < width and 0 <= y < height):
                continue
            last_val = last_times[y, x]
            if t - last_val < refractory_us:
                last_times[y, x] = float(t)
                continue
            x0 = 0 if x - spatial_radius < 0 else x - spatial_radius
            y0 = 0 if y - spatial_radius < 0 else y - spatial_radius
            x1 = width - 1 if x + spatial_radius >= width else x + spatial_radius
            y1 = height - 1 if y + spatial_radius >= height else y + spatial_radius
            count = 0
            yy = y0
            while yy <= y1:
                xx = x0
                while xx <= x1:
                    if t - last_times[yy, xx] <= window_us:
                        count += 1
                        if count >= count_threshold:
                            break
                    xx += 1
                if count >= count_threshold:
                    break
                yy += 1
            if count >= count_threshold:
                mask[i] = True
            last_times[y, x] = float(t)
        return mask
