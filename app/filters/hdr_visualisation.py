"""High dynamic range visualisation filter."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from app.filters.base import BaseFilter


class HDRVisualizationFilter(BaseFilter):
    """Colour-code events by polarity and recency for an HDR look."""

    name = "HDR-Visualisierung"

    def __init__(
        self,
        tau_ms: float = 50.0,
        gamma: float = 0.8,
        pos_colour: Tuple[int, int, int] | None = None,
        neg_colour: Tuple[int, int, int] | None = None,
    ) -> None:
        self.tau_us = max(1.0, float(tau_ms) * 1000.0)
        self.gamma = float(gamma)
        self.pos_colour = tuple(pos_colour) if pos_colour else (255, 255, 200)
        self.neg_colour = tuple(neg_colour) if neg_colour else (80, 120, 255)
        self.surface_pos: np.ndarray | None = None
        self.surface_neg: np.ndarray | None = None
        self.last_timestamp: float = 0.0
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, object]:
        return {
            "tau_ms": self.tau_us / 1000.0,
            "gamma": self.gamma,
            "pos_colour": self.pos_colour,
            "neg_colour": self.neg_colour,
        }

    def set_params(self, **kwargs: object) -> None:
        if "tau_ms" in kwargs:
            self.tau_us = max(1.0, float(kwargs["tau_ms"]) * 1000.0)
        if "gamma" in kwargs:
            self.gamma = float(kwargs["gamma"])
        if "pos_colour" in kwargs:
            self.pos_colour = tuple(int(v) for v in kwargs["pos_colour"])
        if "neg_colour" in kwargs:
            self.neg_colour = tuple(int(v) for v in kwargs["neg_colour"])

    def reset(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.surface_pos = np.zeros((self.height, self.width), dtype=np.float32)
        self.surface_neg = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_timestamp = 0.0

    def _decay(self, dt: float) -> None:
        if self.surface_pos is None or self.surface_neg is None:
            return
        decay = np.exp(-dt / self.tau_us)
        self.surface_pos *= decay
        self.surface_neg *= decay

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if self.surface_pos is None or self.surface_neg is None:
            return {}

        if events.size > 0:
            current_time = float(events["t"][-1])
            if self.last_timestamp > 0.0:
                dt = max(0.0, current_time - self.last_timestamp)
            else:
                dt = 0.0
            self.last_timestamp = current_time
            if dt > 0.0:
                self._decay(dt)

            xs = events["x"].astype(np.int32)
            ys = events["y"].astype(np.int32)
            ps = events["p"].astype(np.int32)
            for x, y, p in zip(xs, ys, ps):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if p == 1:
                        self.surface_pos[y, x] = 1.0
                    else:
                        self.surface_neg[y, x] = 1.0
        elif self.last_timestamp > 0.0:
            dt_hint = float(state.get("dt_hint_us", 0.0))
            if dt_hint > 0.0:
                self._decay(dt_hint)

        pos = np.asarray(self.pos_colour, dtype=np.float32) / 255.0
        neg = np.asarray(self.neg_colour, dtype=np.float32) / 255.0

        pos_img = self.surface_pos[..., None] * pos
        neg_img = self.surface_neg[..., None] * neg
        combined = np.clip(pos_img + neg_img, 0.0, 1.0)
        if self.gamma != 1.0:
            combined = np.power(combined, self.gamma)
        frame = (combined * 255.0).astype(np.uint8)
        return {"frame": frame}

