__all__ = ["AbstractCompositePotential", "CompositePotential"]


import uuid
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, TypeVar, final

import array_api_jax_compat as xp
import equinox as eqx
import jax

from .base import AbstractPotentialBase
from galax.typing import BatchableRealScalarLike, BatchFloatScalar, BatchVec3
from galax.units import UnitSystem, unitsystem
from galax.utils import ImmutableDict
from galax.utils._misc import first

K = TypeVar("K")
V = TypeVar("V")


# Note: cannot have `strict=True` because of inheriting from ImmutableDict.
class AbstractCompositePotential(
    ImmutableDict[AbstractPotentialBase], AbstractPotentialBase, strict=False
):
    # === Potential ===

    @partial(jax.jit)
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableRealScalarLike
    ) -> BatchFloatScalar:
        return xp.sum(
            xp.asarray(
                [p._potential_energy(q, t) for p in self.values()]  # noqa: SLF001
            ),
            axis=0,
        )

    ###########################################################################
    # Composite potentials

    def __or__(self, other: Any) -> "CompositePotential":
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        return CompositePotential(  # combine the two dictionaries
            self._data
            | (  # make `other` into a compatible dictionary.
                other._data
                if isinstance(other, CompositePotential)
                else {str(uuid.uuid4()): other}
            )
        )

    def __ror__(self, other: Any) -> "CompositePotential":
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        return CompositePotential(  # combine the two dictionaries
            (  # make `other` into a compatible dictionary.
                other._data
                if isinstance(other, CompositePotential)
                else {str(uuid.uuid4()): other}
            )
            | self._data
        )

    def __add__(self, other: AbstractPotentialBase) -> "CompositePotential":
        return self | other


###########################################################################


@final
class CompositePotential(AbstractCompositePotential):
    """Composite Potential."""

    _data: dict[str, AbstractPotentialBase]
    _: KW_ONLY
    units: UnitSystem = eqx.field(init=False, static=True, converter=unitsystem)
    _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

    def __init__(
        self,
        potentials: (
            dict[str, AbstractPotentialBase]
            | tuple[tuple[str, AbstractPotentialBase], ...]
        ) = (),
        /,
        *,
        units: Any = None,
        **kwargs: AbstractPotentialBase,
    ) -> None:
        super().__init__(potentials, **kwargs)

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        units_ = units if units is not None else first(self.values()).units
        usys = unitsystem(units_)
        if not all(p.units == usys for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", usys)

        # Apply the unit system to any parameters.
        self._init_units()
