"""Wrapper to add frame operations to a potential."""

__all__ = ["AbstractTransformedPotential"]


from typing import cast

import unxt as u
from xmmutablemap import ImmutableMap

from .base import AbstractPotential


class AbstractTransformedPotential(AbstractPotential):
    """ABC for transformations of a potential."""

    original_potential: AbstractPotential

    @property
    def units(self) -> u.AbstractUnitSystem:
        """The unit system of the potential."""
        return cast(u.AbstractUnitSystem, self.original_potential.units)

    @property
    def constants(self) -> ImmutableMap[str, u.Quantity]:
        """The constants of the potential."""
        return cast(ImmutableMap[str, u.Quantity], self.original_potential.constants)
