"""Registry for visualisation filters."""

from __future__ import annotations

from typing import Dict, Type

from app.filters.base import BaseFilter
from app.filters.event_count import EventCountFilter
from app.filters.event_corners import EventCornerFilter
from app.filters.event_optical_flow import EventOpticalFlowFilter
from app.filters.event_deblurring import EventDeblurFilter
from app.filters.hdr_visualisation import HDRVisualizationFilter


VISUAL_FILTERS: Dict[str, Type[BaseFilter]] = {
    EventCountFilter.name: EventCountFilter,
    EventCornerFilter.name: EventCornerFilter,
    EventOpticalFlowFilter.name: EventOpticalFlowFilter,
    EventDeblurFilter.name: EventDeblurFilter,
    HDRVisualizationFilter.name: HDRVisualizationFilter,
}


def list_visual_filters() -> list[str]:
    """Return the registered visualisation filter names."""

    return list(VISUAL_FILTERS.keys())


def create_visual_filter(name: str, **params: object) -> BaseFilter:
    """Instantiate a visual filter by name."""

    cls = VISUAL_FILTERS[name]
    return cls(**params)

