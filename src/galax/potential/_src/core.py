__all__ = ["AbstractPotential"]

import abc
import uuid
from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx

from unxt import AbstractUnitSystem, Quantity, unitsystem
from xmmutablemap import ImmutableMap

import galax.typing as gt
from .base import AbstractPotentialBase, default_constants
from .composite import CompositePotential


class AbstractPotential(AbstractPotentialBase, strict=True):
    """Abstract base class for all potential objects."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def __post_init__(self) -> None:
        self._apply_unitsystem()

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    # TODO: inputs w/ units
    @abc.abstractmethod
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        raise NotImplementedError

    ###########################################################################

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})
