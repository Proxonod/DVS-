"""Tests for visualisation filters and compositor."""

from __future__ import annotations

import numpy as np

from app.filters.event_count import EventCountFilter
from app.filters.hdr_visualisation import HDRVisualizationFilter
from app.core.visualization import compose_frame


def make_events(coords, times, polarities):
    dtype = [("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)]
    arr = np.zeros(len(coords), dtype=dtype)
    arr["x"] = [c[0] for c in coords]
    arr["y"] = [c[1] for c in coords]
    arr["t"] = times
    arr["p"] = polarities
    return arr


def test_event_count_overlay_accumulates():
    filt = EventCountFilter(integration_ms=10.0, normalisation=2.0)
    filt.reset(4, 4)
    events = make_events([(1, 1), (1, 1)], [0, 1000], [1, 1])
    state = filt.process(events, {})
    overlay = state["overlay_event_count"]
    assert overlay[1, 1] > 0.4


def test_compose_frame_event_count_gradient():
    canvas = np.zeros((2, 2, 3), dtype=np.float32)
    overlay = np.zeros((2, 2), dtype=np.float32)
    overlay[0, 0] = 1.0
    frame = compose_frame(canvas, (0, 255, 170), (255, 51, 102), {"overlay_event_count": overlay})
    pos = np.array((0, 255, 170), dtype=np.float32)
    neg = np.array((255, 51, 102), dtype=np.float32)
    diff_pos = np.linalg.norm(frame[0, 0].astype(np.float32) - pos)
    diff_neg = np.linalg.norm(frame[0, 0].astype(np.float32) - neg)
    assert diff_pos < diff_neg


def test_compose_frame_resizes_overlays():
    canvas = np.zeros((3, 3, 3), dtype=np.float32)
    overlay = np.ones((6, 9), dtype=np.float32)
    frame = compose_frame(canvas, (0, 255, 0), (255, 0, 0), {"overlay_event_count": overlay})
    assert frame.shape == (3, 3, 3)
    assert frame.dtype == np.uint8
    # The resized overlay should have non-zero contribution everywhere.
    assert np.all(frame[..., 1] > 0)


def test_hdr_filter_produces_frame():
    filt = HDRVisualizationFilter(tau_ms=5.0)
    filt.reset(3, 3)
    events = make_events([(0, 0), (1, 1)], [0, 1000], [1, 0])
    out = filt.process(events, {})
    frame = out["frame"]
    assert frame.shape == (3, 3, 3)
    assert frame.dtype == np.uint8
    assert frame[0, 0].sum() > 0
