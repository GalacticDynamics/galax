"""galdynamix: Galactic Dynamix in Jax."""


from functools import singledispatch
from typing import Any

from galdynamix.units import UnitSystem, dimensionless, galactic, solarsystem


@singledispatch
def converter_to_usys(value: Any, /) -> UnitSystem:
    """Argument to ``eqx.field(converter=...)``."""
    msg = f"cannot convert {value} to a UnitSystem"
    raise NotImplementedError(msg)


@converter_to_usys.register
def _from_usys(value: UnitSystem, /) -> UnitSystem:
    return value


@converter_to_usys.register
def _from_none(value: None, /) -> UnitSystem:
    return dimensionless


@converter_to_usys.register(tuple)
def _from_args(value: tuple[Any, ...], /) -> UnitSystem:
    return UnitSystem(*value)


@converter_to_usys.register
def _from_named(value: str, /) -> UnitSystem:
    if value == "dimensionless":
        return dimensionless
    if value == "solarsystem":
        return solarsystem
    if value == "galactic":
        return galactic

    msg = f"cannot convert {value} to a UnitSystem"
    raise NotImplementedError(msg)
