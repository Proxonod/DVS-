"""Event-based corner detection overlay."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseFilter


def _box_filter(image: np.ndarray) -> np.ndarray:
    """Apply a 3×3 box filter using simple padding."""

    padded = np.pad(image, 1, mode="edge")
    acc = (
        padded[0:-2, 0:-2]
        + padded[0:-2, 1:-1]
        + padded[0:-2, 2:]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, 0:-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return acc / 9.0


class EventCornerFilter(BaseFilter):
    """Approximate Harris-like corners on an event time surface."""

    name = "Event-basierte Corner"

    def __init__(self, tau_ms: float = 30.0, sensitivity: float = 0.04, alpha: float = 0.9) -> None:
        self.tau_us = max(1.0, float(tau_ms) * 1000.0)
        self.k = float(sensitivity)
        self.alpha = float(alpha)
        self.surface: np.ndarray | None = None
        self.last_timestamp: float = 0.0
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, float]:
        return {"tau_ms": self.tau_us / 1000.0, "sensitivity": self.k, "alpha": self.alpha}

    def set_params(self, **kwargs: object) -> None:
        if "tau_ms" in kwargs:
            self.tau_us = max(1.0, float(kwargs["tau_ms"]) * 1000.0)
        if "sensitivity" in kwargs:
            self.k = float(kwargs["sensitivity"])
        if "alpha" in kwargs:
            self.alpha = float(kwargs["alpha"])

    def reset(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.surface = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_timestamp = 0.0

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if self.surface is None:
            return {}

        if events.size > 0:
            current_time = float(events["t"][-1])
            if self.last_timestamp > 0.0:
                dt = max(0.0, current_time - self.last_timestamp)
            else:
                dt = 0.0
            self.last_timestamp = current_time
            decay = np.exp(-dt / self.tau_us)
            self.surface *= decay

            xs = events["x"].astype(np.int32)
            ys = events["y"].astype(np.int32)
            for x, y in zip(xs, ys):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.surface[y, x] = 1.0
        elif self.last_timestamp > 0.0:
            dt_hint = float(state.get("dt_hint_us", 0.0))
            if dt_hint > 0.0:
                decay = np.exp(-dt_hint / self.tau_us)
                self.surface *= decay

        gx, gy = np.gradient(self.surface)
        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy

        Ixx = _box_filter(gxx)
        Iyy = _box_filter(gyy)
        Ixy = _box_filter(gxy)

        det = Ixx * Iyy - Ixy * Ixy
        trace = Ixx + Iyy
        response = det - self.k * (trace * trace)

        if response.size == 0:
            return {}

        min_val = response.min()
        max_val = response.max()
        if max_val > min_val:
            norm = (response - min_val) / (max_val - min_val)
        else:
            norm = np.zeros_like(response)
        norm = np.clip(norm, 0.0, 1.0)
        return {
            "overlay_corners": norm.astype(np.float32),
            "corner_alpha": self.alpha,
        }

