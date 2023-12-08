__all__ = ["AbstractPotential"]

import uuid
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx

from galdynamix.units import UnitSystem

from .base import AbstractPotentialBase
from .composite import CompositePotential
from .utils import converter_to_usys


class AbstractPotential(AbstractPotentialBase):
    _: KW_ONLY
    units: UnitSystem = eqx.field(
        default=None, converter=converter_to_usys, static=True
    )
    _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

    def __post_init__(self) -> None:
        self._init_units()

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        from galdynamix.potential._potential.composite import CompositePotential

        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})
