__all__ = ["AbstractPotential"]

import uuid
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx

from galdynamix.units import UnitSystem, dimensionless

from .base import AbstractPotentialBase
from .composite import CompositePotential


class AbstractPotential(AbstractPotentialBase):
    _: KW_ONLY
    units: UnitSystem = eqx.field(
        default=None,
        converter=lambda x: dimensionless if x is None else UnitSystem(x),
        static=True,
    )
    _G: float = eqx.field(init=False, static=True, repr=False)

    def __post_init__(self) -> None:
        self._init_units()

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        from galdynamix.potential._potential.composite import CompositePotential

        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})
