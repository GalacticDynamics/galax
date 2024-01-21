__all__ = ["CompositePotential"]


import uuid
from dataclasses import KW_ONLY
from typing import Any, TypeVar, final

import equinox as eqx
import jax.experimental.array_api as xp
from typing_extensions import override

from galax.typing import (
    BatchableFloatOrIntScalarLike,
    BatchFloatScalar,
    BatchVec3,
)
from galax.units import UnitSystem
from galax.utils import ImmutableDict, partial_jit
from galax.utils._misc import first

from .base import AbstractPotentialBase
from .utils import converter_to_usys

K = TypeVar("K")
V = TypeVar("V")


# Note: cannot have `strict=True` because of inheriting from ImmutableDict.
class AbstractCompositePotential(
    ImmutableDict[AbstractPotentialBase], AbstractPotentialBase, strict=False
):
    _data: dict[str, AbstractPotentialBase]
    _: KW_ONLY
    units: UnitSystem = eqx.field(init=False, static=True, converter=converter_to_usys)
    _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

    # === Potential ===

    @partial_jit()
    def _potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        return xp.sum(xp.asarray([p.potential_energy(q, t) for p in self.values()]))

    ###########################################################################
    # Composite potentials

    @override
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
    units: UnitSystem = eqx.field(init=False, static=True, converter=converter_to_usys)
    _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

    def __init__(
        self,
        potentials: dict[str, AbstractPotentialBase]
        | tuple[tuple[str, AbstractPotentialBase], ...] = (),
        /,
        *,
        units: UnitSystem | None = None,
        **kwargs: AbstractPotentialBase,
    ) -> None:
        super().__init__(potentials, **kwargs)  # type: ignore[arg-type]

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        units_ = units if units is not None else first(self.values()).units
        if not all(p.units == units_ for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", units_)

        # Apply the unit system to any parameters.
        self._init_units()
