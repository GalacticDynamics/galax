"""galdynamix: Galactic Dynamix in Jax."""


from functools import singledispatch
from typing import Any

from galdynamix.units import UnitSystem


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
    from galdynamix.units import dimensionless

    return dimensionless


@converter_to_usys.register(tuple)
def _from_args(value: tuple[Any, ...], /) -> UnitSystem:
    return UnitSystem(*value)


@converter_to_usys.register
def _from_named(value: str, /) -> UnitSystem:
    if value == "dimensionless":
        from galdynamix.units import dimensionless

        return dimensionless
    if value == "solarsystem":
        from galdynamix.units import solarsystem

        return solarsystem
    if value == "galactic":
        from galdynamix.units import galactic

        return galactic

    msg = f"cannot convert {value} to a UnitSystem"
    raise NotImplementedError(msg)
