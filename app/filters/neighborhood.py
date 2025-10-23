"""Neighbourhood consistency filter for event streams."""

from __future__ import annotations

from typing import Dict

import numpy as np

from app.filters.base import BaseFilter

try:
    from numba import njit
except Exception:  # pragma: no cover - numba optional
    njit = None  # type: ignore


class NeighborhoodActivityFilter(BaseFilter):
    """Reject events that lack spatio-temporal neighbours."""

    name = "NeighborhoodActivityFilter"

    def __init__(
        self,
        radius: int = 2,
        time_step_us: int = 2000,
        time_steps: int = 5,
        min_neighbours: int = 1,
    ) -> None:
        self.radius = max(0, int(radius))
        self.time_step_us = max(1, int(time_step_us))
        self.time_steps = max(1, int(time_steps))
        self.min_neighbours = max(1, int(min_neighbours))
        self.max_window = self.time_step_us * self.time_steps
        self.width = 0
        self.height = 0
        self.last_times: np.ndarray

    def params(self) -> Dict[str, object]:
        return {
            "radius": self.radius,
            "time_step_us": self.time_step_us,
            "time_steps": self.time_steps,
            "min_neighbours": self.min_neighbours,
        }

    def set_params(self, **kwargs: object) -> None:
        if "radius" in kwargs:
            self.radius = max(0, int(kwargs["radius"]))
        if "time_step_us" in kwargs:
            self.time_step_us = max(1, int(kwargs["time_step_us"]))
        if "time_steps" in kwargs:
            self.time_steps = max(1, int(kwargs["time_steps"]))
        if "min_neighbours" in kwargs:
            self.min_neighbours = max(1, int(kwargs["min_neighbours"]))
        self.max_window = self.time_step_us * self.time_steps

    def reset(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        safe_neg = -int(self.max_window + 1)
        self.last_times = np.full((self.height, self.width), safe_neg, dtype=np.int64)

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if events.size == 0 or self.width <= 0 or self.height <= 0:
            return {"events": events}

        if njit is not None:
            mask = _neighbourhood_kernel_numba(
                events["x"],
                events["y"],
                events["t"],
                self.last_times,
                self.radius,
                self.time_step_us,
                self.time_steps,
                self.min_neighbours,
            )
        else:
            mask = _neighbourhood_kernel_python(
                events["x"],
                events["y"],
                events["t"],
                self.last_times,
                self.radius,
                self.time_step_us,
                self.time_steps,
                self.min_neighbours,
            )
        return {"events": events[mask]}


def _neighbourhood_kernel_python(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    last_times: np.ndarray,
    radius: int,
    time_step_us: int,
    time_steps: int,
    min_neighbours: int,
) -> np.ndarray:
    n = len(ts)
    mask = np.zeros(n, dtype=bool)
    height, width = last_times.shape
    max_window = time_step_us * time_steps
    for i in range(n):
        x = int(xs[i])
        y = int(ys[i])
        t = int(ts[i])
        if not (0 <= x < width and 0 <= y < height):
            continue

        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(width - 1, x + radius)
        y1 = min(height - 1, y + radius)
        neighbours = 0
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if xx == x and yy == y:
                    continue
                dt = t - int(last_times[yy, xx])
                if dt <= 0 or dt > max_window:
                    continue
                step = (dt - 1) // time_step_us
                if step < time_steps:
                    neighbours += 1
                    if neighbours >= min_neighbours:
                        break
            if neighbours >= min_neighbours:
                break
        if neighbours >= min_neighbours:
            mask[i] = True
        last_times[y, x] = t
    return mask


if njit is not None:

    @njit
    def _neighbourhood_kernel_numba(
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        last_times: np.ndarray,
        radius: int,
        time_step_us: int,
        time_steps: int,
        min_neighbours: int,
    ) -> np.ndarray:
        n = ts.shape[0]
        mask = np.zeros(n, dtype=np.bool_)
        height, width = last_times.shape
        max_window = time_step_us * time_steps
        for i in range(n):
            x = int(xs[i])
            y = int(ys[i])
            t = int(ts[i])
            if not (0 <= x < width and 0 <= y < height):
                continue
            x0 = 0 if x - radius < 0 else x - radius
            y0 = 0 if y - radius < 0 else y - radius
            x1 = width - 1 if x + radius >= width else x + radius
            y1 = height - 1 if y + radius >= height else y + radius
            neighbours = 0
            yy = y0
            while yy <= y1:
                xx = x0
                while xx <= x1:
                    if xx == x and yy == y:
                        xx += 1
                        continue
                    dt = t - int(last_times[yy, xx])
                    if 0 < dt <= max_window:
                        step = (dt - 1) // time_step_us
                        if step < time_steps:
                            neighbours += 1
                            if neighbours >= min_neighbours:
                                break
                    xx += 1
                if neighbours >= min_neighbours:
                    break
                yy += 1
            if neighbours >= min_neighbours:
                mask[i] = True
            last_times[y, x] = t
        return mask

