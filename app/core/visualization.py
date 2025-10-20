"""Utilities for composing visual overlays on top of the event canvas.

This module centralises the logic used by both the UI and the export
pipeline to blend auxiliary visualisations – such as event count heat
maps, corner responses or optical flow estimates – onto the base event
canvas.  Filters implemented in :mod:`app.filters` expose their results
through well defined keys in the state dictionary.  The
``compose_frame`` helper translates these overlays into an RGB image
ready for display or export.

The function accepts the current accumulation canvas (float image in
``[0, 1]``) together with the configured positive/negative event colours
and an optional overlay state.  If the overlay state provides a
pre-composited frame via the ``"frame"`` key it is returned directly;
otherwise the helper blends the registered overlays sequentially on top
of the base canvas.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

__all__ = ["compose_frame"]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Return a copy of ``image`` clipped to ``uint8``."""

    if image.dtype == np.uint8:
        return image.copy()
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    return clipped


def _ensure_float_frame(canvas: np.ndarray) -> np.ndarray:
    """Convert the base canvas to an ``float32`` RGB image in ``[0, 1]``."""

    if canvas.dtype != np.float32:
        frame = canvas.astype(np.float32, copy=True)
    else:
        frame = canvas.copy()
    return np.clip(frame, 0.0, 1.0)


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """Blend ``overlay`` onto ``base`` with opacity ``alpha``."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return base
    return np.clip(base * (1.0 - alpha) + overlay * alpha, 0.0, 1.0)


def _gradient_colour(
    intensity: np.ndarray,
    colour_low: Tuple[int, int, int],
    colour_high: Tuple[int, int, int],
) -> np.ndarray:
    """Interpolate between two RGB colours given ``intensity`` in ``[0, 1]``."""

    low = np.asarray(colour_low, dtype=np.float32) / 255.0
    high = np.asarray(colour_high, dtype=np.float32) / 255.0
    interp = low + intensity[..., None] * (high - low)
    return np.clip(interp, 0.0, 1.0)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV image (values in ``[0, 1]``) to RGB (also ``[0, 1]``)."""

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    i_mod = i % 6

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    masks = [i_mod == k for k in range(6)]

    r[masks[0]] = v[masks[0]]
    g[masks[0]] = t[masks[0]]
    b[masks[0]] = p[masks[0]]

    r[masks[1]] = q[masks[1]]
    g[masks[1]] = v[masks[1]]
    b[masks[1]] = p[masks[1]]

    r[masks[2]] = p[masks[2]]
    g[masks[2]] = v[masks[2]]
    b[masks[2]] = t[masks[2]]

    r[masks[3]] = p[masks[3]]
    g[masks[3]] = q[masks[3]]
    b[masks[3]] = v[masks[3]]

    r[masks[4]] = t[masks[4]]
    g[masks[4]] = p[masks[4]]
    b[masks[4]] = v[masks[4]]

    r[masks[5]] = v[masks[5]]
    g[masks[5]] = p[masks[5]]
    b[masks[5]] = q[masks[5]]

    rgb = np.stack((r, g, b), axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _flow_to_rgb(flow: np.ndarray, max_flow: float) -> np.ndarray:
    """Map a flow field ``(vx, vy)`` to an RGB representation."""

    vx = flow[..., 0]
    vy = flow[..., 1]
    magnitude = np.sqrt(vx * vx + vy * vy)
    max_flow = max(1e-6, float(max_flow))
    mag_norm = np.clip(magnitude / max_flow, 0.0, 1.0)
    angle = np.arctan2(vy, vx)
    hue = (angle + np.pi) / (2.0 * np.pi)
    saturation = mag_norm
    value = np.clip(0.2 + 0.8 * mag_norm, 0.0, 1.0)
    hsv = np.stack((hue, saturation, value), axis=-1)
    return _hsv_to_rgb(hsv)


def compose_frame(
    canvas: np.ndarray,
    pos_colour: Tuple[int, int, int],
    neg_colour: Tuple[int, int, int],
    overlay_state: Dict[str, object] | None,
) -> np.ndarray:
    """Blend overlays contained in ``overlay_state`` on top of ``canvas``.

    Parameters
    ----------
    canvas:
        Floating point RGB image (``H × W × 3``) with values in ``[0, 1]``
        representing the accumulated event canvas.
    pos_colour, neg_colour:
        Current colours configured for positive and negative events.
    overlay_state:
        Dictionary returned by visual filters.  Recognised keys are:

        ``"frame"``: ``uint8`` RGB frame that will be returned directly.
        ``"overlay_event_count"``: float image in ``[0, 1]`` blended using a
            gradient between ``neg_colour`` and ``pos_colour``.
        ``"overlay_corners"``: corner intensity map in ``[0, 1]`` drawn in
            amber tones.
        ``"overlay_flow"``: flow field (``H × W × 2``) converted to an HSV
            colour wheel.

        Additional optional alpha parameters (``event_count_alpha``,
        ``corner_alpha``, ``flow_alpha``) control the opacity of individual
        overlays.
    """

    if overlay_state and "frame" in overlay_state:
        frame = overlay_state["frame"]
        if isinstance(frame, np.ndarray):
            return _to_uint8(frame)

    frame = _ensure_float_frame(canvas)

    if not overlay_state:
        return _to_uint8(frame * 255.0)

    if "overlay_event_count" in overlay_state:
        intensity = np.asarray(overlay_state["overlay_event_count"], dtype=np.float32)
        overlay_rgb = _gradient_colour(intensity, neg_colour, pos_colour)
        alpha = float(overlay_state.get("event_count_alpha", 0.7))
        frame = _blend(frame, overlay_rgb, alpha)

    if "overlay_corners" in overlay_state:
        corners = np.asarray(overlay_state["overlay_corners"], dtype=np.float32)
        amber = np.zeros((*corners.shape, 3), dtype=np.float32)
        # Amber tone (approx. RGB (255, 191, 0))
        amber[..., 0] = 1.0
        amber[..., 1] = 0.75
        overlay_rgb = amber * corners[..., None]
        alpha = float(overlay_state.get("corner_alpha", 0.9))
        frame = _blend(frame, overlay_rgb, alpha)

    if "overlay_flow" in overlay_state:
        flow = np.asarray(overlay_state["overlay_flow"], dtype=np.float32)
        max_flow = float(overlay_state.get("flow_max", 1.0))
        overlay_rgb = _flow_to_rgb(flow, max_flow)
        alpha = float(overlay_state.get("flow_alpha", 0.8))
        frame = _blend(frame, overlay_rgb, alpha)

    return _to_uint8(frame * 255.0)

