"""Unit tests for core filters and pipeline logic.

These tests verify that filters behave correctly on simple synthetic
inputs.  They are not exhaustive but cover typical scenarios such as
background activity suppression, refractory period enforcement, time
surface decay and basic pipeline composition.
"""

import numpy as np

from app.filters.baf import BackgroundActivityFilter
from app.filters.neighborhood import NeighborhoodActivityFilter
from app.filters.refractory import RefractoryFilter
from app.filters.time_surface import TimeSurfaceFilter
from app.core.pipeline import Pipeline


def make_events(coords, times, polarities):
    """Helper to construct a structured array of events.

    Parameters
    ----------
    coords:
        Sequence of (x, y) tuples.
    times:
        Sequence of timestamps in microseconds.
    polarities:
        Sequence of polarity values (0 or 1).

    Returns
    -------
    numpy.ndarray
        Structured array with fields x, y, t, p.
    """
    dtype = [("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)]
    arr = np.zeros(len(coords), dtype=dtype)
    arr["x"] = [c[0] for c in coords]
    arr["y"] = [c[1] for c in coords]
    arr["t"] = times
    arr["p"] = polarities
    return arr


def test_refractory_filter():
    """Events within the refractory period at the same pixel should be suppressed."""
    filt = RefractoryFilter(refractory_us=500)
    filt.reset(4, 4)
    events = make_events(
        coords=[(1, 1), (1, 1), (1, 1)],
        times=[0, 100, 700],
        polarities=[1, 1, 1],
    )
    out = filt.process(events, {})["events"]
    # Expect only first (t=0) and last (t=700) to pass
    assert len(out) == 2
    assert list(out["t"]) == [0, 700]


def test_baf_filter_requires_multiple_neighbours():
    """BAF should keep events only when enough neighbours exist in the window."""
    filt = BackgroundActivityFilter(window_ms=20.0, count_threshold=2, refractory_us=0, spatial_radius=2)
    filt.reset(6, 6)
    events = make_events(
        coords=[(2, 2), (1, 2), (3, 2)],
        times=[0, 2000, 5000],
        polarities=[1, 1, 1],
    )
    out = filt.process(events, {})["events"]
    assert len(out) == 1
    assert out["x"][0] == 3 and out["y"][0] == 2


def test_neighbourhood_filter_discards_isolated_events():
    """Neighbourhood filter should reject events without nearby support."""
    filt = NeighborhoodActivityFilter(radius=1, time_step_us=2000, time_steps=3, min_neighbours=1)
    filt.reset(6, 6)
    events = make_events(
        coords=[(2, 2), (4, 4), (2, 3), (2, 4)],
        times=[0, 0, 2500, 15000],
        polarities=[1, 1, 1, 1],
    )
    out = filt.process(events, {})["events"]
    # Only (2,3) should remain: it has a neighbour (2,2) within both radius and time steps.
    assert len(out) == 1
    assert (int(out["x"][0]), int(out["y"][0])) == (2, 3)


def test_time_surface_decay_and_boost():
    """TimeSurface should decay intensities and boost on new events."""
    filt = TimeSurfaceFilter(tau_ms=10.0, polarity_separate=True)
    filt.reset(3, 3)
    # First slice: one positive event
    events1 = make_events(coords=[(1, 1)], times=[0], polarities=[1])
    filt.process(events1, {})
    ts_pos1 = filt.surface_pos.copy()
    # After 5ms, intensity should decay approximately exp(-5ms/10ms) ~ 0.6065
    events2 = make_events(coords=[(0, 0)], times=[5000], polarities=[1])
    filt.process(events2, {})
    # Intensity at (1,1) should have decayed
    assert np.isclose(filt.surface_pos[1, 1], ts_pos1[1, 1] * np.exp(-5000 / (10 * 1000)), atol=1e-4)
    # Intensity at (0,0) should be reset to 1 due to new event
    assert np.isclose(filt.surface_pos[0, 0], 1.0, atol=1e-6)


def test_time_surface_cuda_env(monkeypatch):
    """Setting DVS_USE_CUDA enables GPU buffers when CuPy is available."""

    monkeypatch.setenv("DVS_USE_CUDA", "1")
    filt = TimeSurfaceFilter(tau_ms=5.0)
    filt.reset(2, 2)
    events = make_events(coords=[(0, 0)], times=[0], polarities=[1])
    state: dict[str, object] = {}
    filt.process(events, state)
    # GPU buffers stay internal but exported overlays must always be NumPy arrays
    assert isinstance(state["time_surface_pos"], np.ndarray)
    assert isinstance(state["time_surface_neg"], np.ndarray)
    if filt.device == "cuda":
        # When running on CUDA, the internal arrays should be CuPy-backed
        assert filt.surface_pos.__class__.__module__.startswith("cupy")
    else:
        assert filt.device == "cpu"


def test_pipeline_render_colours():
    """Pipeline should colour events according to polarity."""
    pl = Pipeline()
    pl.reset(4, 4)
    # Events of both polarities
    events = make_events(
        coords=[(0, 0), (1, 1), (2, 2)],
        times=[0, 0, 0],
        polarities=[1, 0, 1],
    )
    state = pl.process_events(events)
    frame = pl.get_frame(state)
    # Positive colour default is (0,255,170), negative is (255,51,102)
    # Check some pixels
    assert tuple(frame[0, 0]) == pl.pos_colour  # positive event
    assert tuple(frame[1, 1]) == pl.neg_colour  # negative event
    assert tuple(frame[2, 2]) == pl.pos_colour  # positive event