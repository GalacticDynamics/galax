from __future__ import annotations

__all__ = ["CompositePotential"]


from dataclasses import KW_ONLY
from typing import TypeVar, final

import equinox as eqx
import jax.numpy as xp
import jax.typing as jt

from galdynamix.units import UnitSystem, dimensionless
from galdynamix.utils import ImmutableDict, partial_jit

from .base import AbstractPotentialBase

K = TypeVar("K")
V = TypeVar("V")


@final
class CompositePotential(ImmutableDict[AbstractPotentialBase], AbstractPotentialBase):
    """Composite Potential."""

    _data: dict[str, AbstractPotentialBase]
    _: KW_ONLY
    units: UnitSystem = eqx.field(
        static=True, converter=lambda x: dimensionless if x is None else UnitSystem(x)
    )
    _G: float = eqx.field(init=False, static=True)

    def __init__(
        self,
        potentials: dict[str, AbstractPotentialBase]
        | tuple[tuple[str, AbstractPotentialBase], ...] = (),
        /,
        units: UnitSystem | None = None,
        **kwargs: AbstractPotentialBase,
    ) -> None:
        super().__init__(potentials, **kwargs)  # type: ignore[arg-type]
        self.units = self.__dataclass_fields__["units"].metadata["converter"](units)
        # TODO: check unit systems of contained potentials to make sure they match.

        self._init_units()

    # === Potential ===

    @partial_jit()
    def potential_energy(
        self,
        q: jt.Array,
        t: jt.Array,
    ) -> jt.Array:
        return xp.sum(xp.array([p.potential_energy(q, t) for p in self.values()]))
