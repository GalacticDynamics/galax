"""galax: Galactic Dynamix in Jax."""


from functools import singledispatch
from typing import Any, TypeVar

import jax.numpy as xp
from astropy.coordinates import BaseRepresentation, BaseRepresentationOrDifferential
from astropy.units import Quantity
from jax import Array

from galax.units import UnitSystem, dimensionless, galactic, solarsystem


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
@converter_to_usys.register(list)
def _from_args(value: tuple[Any, ...] | list[Any], /) -> UnitSystem:
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


# =============================================================================


def parse_inputs(*args: Any, units: UnitSystem, **kwargs: Any) -> tuple[Array, ...]:
    """Parse input arguments."""
    return tuple(parse_input(arg, units=units, **kwargs) for arg in args)


# --------------------------------------------------------------

Value = TypeVar("Value", int, float, Array)


@singledispatch
def parse_input(value: Any, /, *, units: UnitSystem, **kwargs: Any) -> Any:
    """Parse input arguments.

    This function uses :func:`~functools.singledispatch` to dispatch on the type
    of the input argument.

    Parameters
    ----------
    value : Any, positional-only
        Input value.
    units : UnitSystem, keyword-only
        Unit system.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        Parsed input value.
    """
    msg = f"cannot parse {value} with units {units}"
    raise NotImplementedError(msg)


@parse_input.register(int)
@parse_input.register(float)
@parse_input.register(Array)
def _parse_from_jax_array(
    value: Value, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    return xp.asarray(value)


@parse_input.register(Quantity)
def _parse_from_quantity(
    value: Quantity, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    return xp.asarray(value.decompose(units).value)


@parse_input.register(BaseRepresentationOrDifferential)
def _parse_from_baserep(
    value: BaseRepresentationOrDifferential, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    return xp.stack(
        [getattr(value, attr).decompose(units).value for attr in value.components]
    )


@parse_input.register(BaseRepresentation)
def _parse_from_representation(
    value: BaseRepresentation, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    if "s" in value.differentials and not kwargs.get("no_differentials", False):
        return xp.stack(
            (
                _parse_from_baserep(value, units=units),
                _parse_from_baserep(value.differentials["s"], units=units),
            )
        )
    return _parse_from_baserep(value, units=units)
