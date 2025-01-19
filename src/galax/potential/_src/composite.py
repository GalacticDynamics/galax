"""Composite Potential."""

__all__ = ["CompositePotential"]


from dataclasses import KW_ONLY
from types import MappingProxyType
from typing import ClassVar, final

import equinox as eqx

import unxt as u
from xmmutablemap import ImmutableMap

from .base import AbstractPotential, default_constants
from .base_multi import AbstractCompositePotential
from .params.attr import CompositeParametersAttribute


@final
class CompositePotential(AbstractCompositePotential):
    """Composite Potential."""

    parameters: ClassVar = CompositeParametersAttribute(MappingProxyType({}))

    _data: dict[str, AbstractPotential]
    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(
        init=False, static=True, converter=u.unitsystem
    )
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )
