"""Event-assisted deblurring filter."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from app.filters.base import BaseFilter


class EventDeblurFilter(BaseFilter):
    """Sharpen the accumulated event canvas using a simple unsharp mask."""

    name = "Event-gestützte Deblurring"

    def __init__(
        self,
        decay: float = 0.9,
        sharpen_amount: float = 0.6,
        pos_colour: Tuple[int, int, int] | None = None,
        neg_colour: Tuple[int, int, int] | None = None,
    ) -> None:
        self.decay = float(decay)
        self.sharpen_amount = float(sharpen_amount)
        self.pos_colour = tuple(pos_colour) if pos_colour else (0, 255, 170)
        self.neg_colour = tuple(neg_colour) if neg_colour else (255, 51, 102)
        self.canvas: np.ndarray | None = None
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "sharpen_amount": self.sharpen_amount,
            "pos_colour": self.pos_colour,
            "neg_colour": self.neg_colour,
        }

    def set_params(self, **kwargs: object) -> None:
        if "decay" in kwargs:
            self.decay = float(kwargs["decay"])
        if "sharpen_amount" in kwargs:
            self.sharpen_amount = float(kwargs["sharpen_amount"])
        if "pos_colour" in kwargs:
            self.pos_colour = tuple(int(v) for v in kwargs["pos_colour"])
        if "neg_colour" in kwargs:
            self.neg_colour = tuple(int(v) for v in kwargs["neg_colour"])

    def reset(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if self.canvas is None:
            return {}

        self.canvas *= self.decay

        if events.size > 0:
            xs = events["x"].astype(np.int32)
            ys = events["y"].astype(np.int32)
            ps = events["p"].astype(np.int32)

            pos = np.asarray(self.pos_colour, dtype=np.float32) / 255.0
            neg = np.asarray(self.neg_colour, dtype=np.float32) / 255.0

            for x, y, p in zip(xs, ys, ps):
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                if p == 1:
                    self.canvas[y, x] = np.minimum(1.0, self.canvas[y, x] + pos)
                else:
                    self.canvas[y, x] = np.minimum(1.0, self.canvas[y, x] + neg)

        # Simple unsharp mask using 4-neighbour average
        shifted = (
            np.roll(self.canvas, 1, axis=0)
            + np.roll(self.canvas, -1, axis=0)
            + np.roll(self.canvas, 1, axis=1)
            + np.roll(self.canvas, -1, axis=1)
        ) / 4.0
        sharpened = np.clip(self.canvas + self.sharpen_amount * (self.canvas - shifted), 0.0, 1.0)

        frame = (sharpened * 255.0).astype(np.uint8)
        return {"frame": frame}

