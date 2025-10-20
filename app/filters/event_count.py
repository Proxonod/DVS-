"""Event count image filter."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseFilter


class EventCountFilter(BaseFilter):
    """Accumulate per-pixel event counts over a sliding time window."""

    name = "Event Count Image"

    def __init__(
        self,
        integration_ms: float = 50.0,
        normalisation: float = 10.0,
        alpha: float = 0.7,
    ) -> None:
        self.integration_us = max(1.0, float(integration_ms) * 1000.0)
        self.normalisation = max(1.0, float(normalisation))
        self.alpha = float(alpha)
        self.accum: np.ndarray | None = None
        self.last_timestamp: float = 0.0
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, float]:
        return {
            "integration_ms": self.integration_us / 1000.0,
            "normalisation": self.normalisation,
            "alpha": self.alpha,
        }

    def set_params(self, **kwargs: object) -> None:
        if "integration_ms" in kwargs:
            val = float(kwargs["integration_ms"])
            self.integration_us = max(1.0, val * 1000.0)
        if "normalisation" in kwargs:
            self.normalisation = max(1.0, float(kwargs["normalisation"]))
        if "alpha" in kwargs:
            self.alpha = float(kwargs["alpha"])

    def reset(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.accum = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_timestamp = 0.0

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if self.accum is None:
            return {}

        if events.size > 0:
            current_time = float(events["t"][-1])
            if self.last_timestamp > 0.0:
                dt = max(0.0, current_time - self.last_timestamp)
            else:
                dt = 0.0
            self.last_timestamp = current_time
            decay = np.exp(-dt / self.integration_us)
            self.accum *= decay

            xs = events["x"].astype(np.int32)
            ys = events["y"].astype(np.int32)
            for x, y in zip(xs, ys):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.accum[y, x] += 1.0
        elif self.last_timestamp > 0.0:
            dt_hint = float(state.get("dt_hint_us", 0.0))
            if dt_hint > 0.0:
                decay = np.exp(-dt_hint / self.integration_us)
                self.accum *= decay

        normalised = np.clip(self.accum / self.normalisation, 0.0, 1.0)
        return {
            "overlay_event_count": normalised.copy(),
            "event_count_alpha": self.alpha,
        }

