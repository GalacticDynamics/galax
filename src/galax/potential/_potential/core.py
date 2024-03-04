__all__ = ["AbstractPotential"]

import abc
import uuid
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx

from unxt import Quantity

from .base import AbstractPotentialBase, default_constants
from .composite import CompositePotential
from galax.typing import FloatQScalar, QVec3, RealScalar
from galax.units import UnitSystem, unitsystem
from galax.utils import ImmutableDict


class AbstractPotential(AbstractPotentialBase, strict=True):
    """Abstract base class for all potential objects."""

    _: KW_ONLY
    units: UnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    def __post_init__(self) -> None:
        self._init_units()

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    # TODO: inputs w/ units
    @abc.abstractmethod
    def _potential_energy(self, q: QVec3, t: RealScalar, /) -> FloatQScalar:
        raise NotImplementedError

    ###########################################################################

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})
