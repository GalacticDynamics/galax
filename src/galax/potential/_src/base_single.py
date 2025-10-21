__all__ = ["AbstractSinglePotential"]

import uuid
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx

import unxt as u
from xmmutablemap import ImmutableMap

from .base import AbstractPotential, default_constants
from .composite import CompositePotential


class AbstractSinglePotential(AbstractPotential):
    """Abstract base class for all potential objects."""

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def __post_init__(self) -> None:
        self._apply_unitsystem()

    ###########################################################################

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotential):
            return NotImplemented

        # CompositePotential has better methods for combining potentials
        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})
