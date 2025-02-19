"""Protocol for PSP interpolation."""

__all__ = ["PhaseSpaceObjectInterpolant"]

from typing import Protocol, runtime_checkable

import unxt as u

import galax._custom_types as gt
from .base import AbstractPhaseSpaceObject


@runtime_checkable
class PhaseSpaceObjectInterpolant(Protocol):
    """Protocol for interpolating phase-space positions."""

    @property
    def units(self) -> u.AbstractUnitSystem:
        """The unit system for the interpolation."""

    def __call__(self, t: gt.QuSzTime) -> AbstractPhaseSpaceObject:
        """Evaluate the interpolation.

        Parameters
        ----------
        t : Quantity[float, (time,), 'time']
            The times at which to evaluate the interpolation.

        """
        ...
