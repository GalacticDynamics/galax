"""Composite Potential."""

__all__ = ["CompositePotential"]


from dataclasses import KW_ONLY
from types import MappingProxyType
from typing import ClassVar, final

import equinox as eqx

from unxt import AbstractUnitSystem, Quantity, unitsystem
from xmmutablemap import ImmutableMap

from .base import default_constants
from .base_multi import AbstractCompositePotential
from .params.attr import CompositeParametersAttribute


@final
class CompositePotential(AbstractCompositePotential):
    """Composite Potential."""

    parameters: ClassVar = CompositeParametersAttribute(MappingProxyType({}))

    _data: dict[str, AbstractBasePotential]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(init=False, static=True, converter=unitsystem)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )
