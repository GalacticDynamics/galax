__all__ = ["AbstractPotential"]

import abc
import uuid
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx

from .base import AbstractPotentialBase
from .composite import CompositePotential
from galax.typing import FloatOrIntScalar, FloatScalar, Vec3
from galax.units import UnitSystem, unitsystem


class AbstractPotential(AbstractPotentialBase, strict=True):
    """Abstract base class for all potential objects."""

    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)
    _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

    def __post_init__(self) -> None:
        self._init_units()

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    def _potential_energy(self, q: Vec3, /, t: FloatOrIntScalar) -> FloatScalar:
        raise NotImplementedError

    ###########################################################################

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})
