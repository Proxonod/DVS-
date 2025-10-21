"""Filter registry.

This module maintains a mapping of filter names to their corresponding
classes.  The UI can query this registry to obtain a list of
available filters and instantiate them dynamically when a user adds
one to the pipeline.  New filters should be imported and registered
here.
"""

from __future__ import annotations

from typing import Dict, Type

from app.filters.base import BaseFilter
from app.filters.baf import BackgroundActivityFilter
from app.filters.refractory import RefractoryFilter
from app.filters.time_surface import TimeSurfaceFilter


# Map human‑readable names to filter classes.  When adding new filters
# import them above and register here.  The key should correspond to
# the ``name`` attribute of the class.
FILTERS: Dict[str, Type[BaseFilter]] = {
    BackgroundActivityFilter.name: BackgroundActivityFilter,
    RefractoryFilter.name: RefractoryFilter,
    TimeSurfaceFilter.name: TimeSurfaceFilter,
}


def list_filter_names() -> list[str]:
    """Return a list of registered filter names."""
    return list(FILTERS.keys())


def create_filter(name: str, **params: object) -> BaseFilter:
    """Instantiate a filter by name.

    Parameters
    ----------
    name:
        Name of the filter as returned by :func:`list_filter_names`.
    params:
        Keyword arguments to pass to the filter constructor.

    Returns
    -------
    BaseFilter
        An instance of the requested filter.

    Raises
    ------
    KeyError
        If ``name`` is not a registered filter.
    """
    cls = FILTERS[name]
    return cls(**params)
