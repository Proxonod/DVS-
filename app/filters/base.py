"""Base classes and utilities for event filters.

This module defines an abstract base class, :class:`BaseFilter`,
establishing the interface that all filters must implement.  Filters
receive slices of events and an arbitrary state dictionary and may
return a dictionary containing an updated event array and auxiliary
data.  Parameters can be queried and set via :meth:`params` and
``set_params``; filters that allocate arrays based on sensor
dimensions should override :meth:`reset`.

Concrete filters reside in submodules under :mod:`app.filters` and
should inherit from :class:`BaseFilter`.  They must provide a
``name`` attribute identifying the filter in the UI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseFilter(ABC):
    """Abstract base class for event filters.

    Filters operate on NumPy structured arrays containing DVS events
    (fields ``x``, ``y``, ``t``, ``p``).  They may inspect and modify the
    events and maintain internal state across calls.  Implementations
    should be thread‑safe if the pipeline is used concurrently from
    multiple threads.
    """

    #: Human‑readable name for the filter
    name: str = "UnnamedFilter"

    def params(self) -> Dict[str, Any]:
        """Return the filter’s parameters as a dictionary.

        Subclasses should override this to expose their tunable
        parameters.  The returned dictionary should contain serialisable
        values suitable for presets.  By default, returns an empty
        dictionary.
        """
        return {}

    def set_params(self, **kwargs: Any) -> None:
        """Update filter parameters from keyword arguments.

        Subclasses should override this method to handle parameter
        updates.  Unknown parameters should be ignored.  Values should
        be validated where appropriate.
        """
        pass

    def reset(self, width: int, height: int) -> None:
        """Reset internal state for a new sensor geometry.

        Filters that allocate per‑pixel arrays (e.g. BAF, refractory
        filters, time surfaces) must override this method to
        (re)initialise those arrays.  The default implementation does
        nothing.

        Parameters
        ----------
        width:
            Sensor width in pixels.
        height:
            Sensor height in pixels.
        """
        pass

    @abstractmethod
    def process(self, events: np.ndarray, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a slice of events and return a result dictionary.

        Subclasses must implement this method.  The input ``events`` is
        a structured NumPy array with fields ``x``, ``y``, ``t`` and
        ``p``.  The ``state`` dictionary may contain auxiliary data from
        previous filters in the pipeline.  The return value should be a
        dictionary; to modify the event stream, include a key
        ``"events"`` mapping to a structured array.  Additional keys
        represent overlays or intermediate results (e.g. time surfaces).

        Parameters
        ----------
        events:
            Structured array of input events.
        state:
            State dictionary shared between filters.

        Returns
        -------
        dict
            A dictionary containing the (possibly filtered) events and
            auxiliary data.  Filters may mutate their own internal
            state but should not mutate the ``state`` dictionary
            directly.
        """
        ...
