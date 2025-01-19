__all__ = ["AbstractSinglePotential"]

import abc
import uuid
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx

import unxt as u
from xmmutablemap import ImmutableMap

import galax.typing as gt
from .base import AbstractPotential, default_constants
from .composite import CompositePotential


class AbstractSinglePotential(AbstractPotential, strict=True):
    """Abstract base class for all potential objects."""

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def __post_init__(self) -> None:
        self._apply_unitsystem()

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    # TODO: inputs w/ units
    @abc.abstractmethod
    def _potential(self, q: gt.BtQuSz3, t: gt.BBtRealSz0, /) -> gt.SpecificEnergyBtSz0:
        raise NotImplementedError

    ###########################################################################

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotential):
            return NotImplemented

        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})
