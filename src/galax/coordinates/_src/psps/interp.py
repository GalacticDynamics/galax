"""Protocol for PSP interpolation."""

__all__ = ["PhaseSpacePositionInterpolant"]

from typing import Protocol, runtime_checkable

import unxt as u

import galax.typing as gt
from .base import AbstractPhaseSpacePosition


@runtime_checkable
class PhaseSpacePositionInterpolant(Protocol):
    """Protocol for interpolating phase-space positions."""

    @property
    def units(self) -> u.AbstractUnitSystem:
        """The unit system for the interpolation."""

    def __call__(self, t: gt.QuSzTime) -> AbstractPhaseSpacePosition:
        """Evaluate the interpolation.

        Parameters
        ----------
        t : Quantity[float, (time,), 'time']
            The times at which to evaluate the interpolation.

        Returns
        -------
        :class:`galax.coordinates.PhaseSpacePosition`
            The interpolated phase-space positions.
        """
        ...
