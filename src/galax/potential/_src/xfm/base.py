"""Wrapper to add frame operations to a potential."""

__all__ = ["AbstractTransformedPotential"]


from typing import cast

import unxt as u
from xmmutablemap import ImmutableMap

from galax.potential._src.base import AbstractPotential


class AbstractTransformedPotential(AbstractPotential):
    """ABC for transformations of a potential."""

    base_potential: AbstractPotential
    """The base potential."""

    @property
    def units(self) -> u.AbstractUnitSystem:
        """The unit system of the potential."""
        return cast(u.AbstractUnitSystem, self.base_potential.units)

    @property
    def constants(self) -> ImmutableMap[str, u.AbstractQuantity]:
        """The constants of the potential."""
        return cast(
            ImmutableMap[str, u.AbstractQuantity], self.base_potential.constants
        )
