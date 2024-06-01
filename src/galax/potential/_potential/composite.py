__all__ = ["AbstractCompositePotential", "CompositePotential"]


import uuid
from dataclasses import KW_ONLY
from functools import partial
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar, final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity, unitsystem

import galax.typing as gt
from .base import AbstractPotentialBase, default_constants
from .param.attr import CompositeParametersAttribute
from galax.utils import ImmutableDict
from galax.utils._misc import zeroth

K = TypeVar("K")
V = TypeVar("V")


# Note: cannot have `strict=True` because of inheriting from ImmutableDict.
class AbstractCompositePotential(
    ImmutableDict[AbstractPotentialBase], AbstractPotentialBase, strict=False
):
    def __init__(
        self,
        potentials: (
            dict[str, AbstractPotentialBase]
            | tuple[tuple[str, AbstractPotentialBase], ...]
        ) = (),
        /,
        *,
        units: Any = None,
        constants: Any = default_constants,
        **kwargs: AbstractPotentialBase,
    ) -> None:
        super().__init__(potentials, **kwargs)  # <- ImmutableDict.__init__

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        units_ = units if units is not None else zeroth(self.values()).units
        usys = unitsystem(units_)
        if not all(p.units == usys for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", usys)  # TODO: not call `object.__setattr__`

        # TODO: some similar check that the same constants are the same, e.g.
        #       `G` is the same for all potentials. Or use `constants` to update
        #       the `constants` of every potential (before `super().__init__`)
        object.__setattr__(self, "constants", constants)

        # Apply the unit system to any parameters.
        self._init_units()

    # === Potential ===

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        return xp.sum(
            xp.asarray(
                [p._potential(q, t) for p in self.values()]  # noqa: SLF001
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

    parameters: ClassVar = CompositeParametersAttribute(MappingProxyType({}))

    _data: dict[str, AbstractPotentialBase]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(init=False, static=True, converter=unitsystem)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )
