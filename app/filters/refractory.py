"""Per‑pixel refractory period filter.

This simple filter suppresses events that occur within a short time
window after a previous event at the same pixel.  It maintains a
per‑pixel array of the timestamp of the last accepted event.  When a
new event arrives, it is only accepted if the difference between its
timestamp and the stored timestamp is greater than ``refractory_us``.

This filter is useful for removing spurious bursts or chatter on
individual pixels and is a lightweight alternative to more expensive
background activity suppression.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseFilter


class RefractoryFilter(BaseFilter):
    """Suppress events occurring too soon after a previous one."""

    name = "RefractoryFilter"

    def __init__(self, refractory_us: int = 500) -> None:
        self.refractory_us = int(refractory_us)
        self.last_times: np.ndarray
        self.width = 0
        self.height = 0

    def params(self) -> Dict[str, object]:
        return {"refractory_us": self.refractory_us}

    def set_params(self, **kwargs: object) -> None:
        if "refractory_us" in kwargs:
            self.refractory_us = int(kwargs["refractory_us"])

    def reset(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # Initialise with very negative times so that early events are not suppressed
        self.last_times = np.full((height, width), -np.iinfo(np.int64).max, dtype=np.int64)

    def process(self, events: np.ndarray, state: Dict[str, object]) -> Dict[str, object]:
        if events.size == 0:
            return {"events": events}
        xs = events["x"].astype(np.int64)
        ys = events["y"].astype(np.int64)
        ts = events["t"].astype(np.int64)
        mask = np.zeros(len(events), dtype=bool)
        last = self.last_times
        refractory = self.refractory_us
        for i in range(len(events)):
            x = xs[i]
            y = ys[i]
            t = ts[i]
            # Accept if time difference exceeds refractory period
            if t - last[y, x] >= refractory:
                mask[i] = True
                last[y, x] = t
        return {"events": events[mask]}
