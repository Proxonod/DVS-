"""Event-based optical flow filter."""

from __future__ import annotations

from typing import Dict

import numpy as np

from app.filters.base import BaseFilter


class EventOpticalFlowFilter(BaseFilter):
    """Estimate optical flow vectors from local event motion."""

    name = "Optical Flow (events)"

    def __init__(
        self,
        neighbor_radius: int = 1,
        history_us: float = 10000.0,
        decay_ms: float = 50.0,
        max_flow: float = 800.0,
        alpha: float = 0.8,
    ) -> None:
        self.radius = max(1, int(neighbor_radius))
        self.history_us = max(1.0, float(history_us))
        self.decay_us = max(1.0, float(decay_ms) * 1000.0)
        self.max_flow = max(1.0, float(max_flow))
        self.alpha = float(alpha)
        self.flow_x: np.ndarray | None = None
        self.flow_y: np.ndarray | None = None
        self.last_timestamp: float = 0.0
        self.last_event_time: np.ndarray | None = None
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, float]:
        return {
            "neighbor_radius": float(self.radius),
            "history_us": self.history_us,
            "decay_ms": self.decay_us / 1000.0,
            "max_flow": self.max_flow,
            "alpha": self.alpha,
        }

    def set_params(self, **kwargs: object) -> None:
        if "neighbor_radius" in kwargs:
            self.radius = max(1, int(kwargs["neighbor_radius"]))
        if "history_us" in kwargs:
            self.history_us = max(1.0, float(kwargs["history_us"]))
        if "decay_ms" in kwargs:
            self.decay_us = max(1.0, float(kwargs["decay_ms"]) * 1000.0)
        if "max_flow" in kwargs:
            self.max_flow = max(1.0, float(kwargs["max_flow"]))
        if "alpha" in kwargs:
            self.alpha = float(kwargs["alpha"])

    def reset(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.flow_x = np.zeros((self.height, self.width), dtype=np.float32)
        self.flow_y = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_event_time = np.zeros((self.height, self.width), dtype=np.float64)
        self.last_timestamp = 0.0

    def _decay_flow(self, dt: float) -> None:
        if self.flow_x is None or self.flow_y is None:
            return
        decay = np.exp(-dt / self.decay_us)
        self.flow_x *= decay
        self.flow_y *= decay

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if self.flow_x is None or self.flow_y is None or self.last_event_time is None:
            return {}

        if events.size > 0:
            current_time = float(events["t"][-1])
            if self.last_timestamp > 0.0:
                dt = max(0.0, current_time - self.last_timestamp)
            else:
                dt = 0.0
            self.last_timestamp = current_time
            if dt > 0.0:
                self._decay_flow(dt)

            xs = events["x"].astype(np.int32)
            ys = events["y"].astype(np.int32)
            ts = events["t"].astype(np.float64)

            r = self.radius
            for x, y, t in zip(xs, ys, ts):
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                x0 = max(0, x - r)
                x1 = min(self.width - 1, x + r)
                y0 = max(0, y - r)
                y1 = min(self.height - 1, y + r)
                patch = self.last_event_time[y0 : y1 + 1, x0 : x1 + 1]
                if patch.size == 0:
                    prev_t = 0.0
                    prev_x = x
                    prev_y = y
                else:
                    prev_idx = np.argmax(patch)
                    prev_t = patch.flat[prev_idx]
                    if prev_t <= 0.0 or t - prev_t > self.history_us:
                        prev_t = 0.0
                        prev_x = x
                        prev_y = y
                    else:
                        dy, dx = divmod(prev_idx, patch.shape[1])
                        prev_x = x0 + dx
                        prev_y = y0 + dy

                if prev_t > 0.0:
                    dt_ev = max(1.0, t - prev_t)
                    vx = (x - prev_x) * 1000.0 / dt_ev  # px per ms
                    vy = (y - prev_y) * 1000.0 / dt_ev
                    self.flow_x[y, x] = 0.7 * self.flow_x[y, x] + 0.3 * vx
                    self.flow_y[y, x] = 0.7 * self.flow_y[y, x] + 0.3 * vy

                self.last_event_time[y, x] = t
        elif self.last_timestamp > 0.0:
            dt_hint = float(state.get("dt_hint_us", 0.0))
            if dt_hint > 0.0:
                self._decay_flow(dt_hint)

        flow = np.stack((self.flow_x, self.flow_y), axis=-1)
        return {
            "overlay_flow": flow.copy(),
            "flow_max": self.max_flow,
            "flow_alpha": self.alpha,
        }

