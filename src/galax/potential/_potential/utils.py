"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import singledispatch
from typing import Any, TypeVar

import jax.numpy as xp
from astropy.coordinates import BaseRepresentation, BaseRepresentationOrDifferential
from astropy.units import Quantity
from jax import Array

from galax.units import (
    DimensionlessUnitSystem,
    UnitSystem,
    dimensionless,
    galactic,
    solarsystem,
)


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


def convert_inputs_to_arrays(
    *args: Any, units: UnitSystem, **kwargs: Any
) -> tuple[Array, ...]:
    """Parse input arguments.

    Parameters
    ----------
    *args : Any, positional-only
        Input arguments to parse to arrays.
    units : UnitSystem, keyword-only
        Unit system.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    tuple[Array, ...]
        Parsed input arguments.
    """
    return tuple(convert_input_to_array(arg, units=units, **kwargs) for arg in args)


# --------------------------------------------------------------

Value = TypeVar("Value", int, float, Array)


@singledispatch
def convert_input_to_array(value: Any, /, *, units: UnitSystem, **kwargs: Any) -> Any:
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
    msg = f"cannot convert {value} using units {units}"
    raise NotImplementedError(msg)


@convert_input_to_array.register(int)
@convert_input_to_array.register(float)
@convert_input_to_array.register(Array)
def _convert_from_arraylike(
    value: Value, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    return xp.asarray(value)


@convert_input_to_array.register(Quantity)
def _convert_from_quantity(
    value: Quantity, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    return xp.asarray(value.decompose(units).value)


@convert_input_to_array.register(BaseRepresentationOrDifferential)
def _convert_from_baserep(
    value: BaseRepresentationOrDifferential, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    return xp.stack(
        [getattr(value, attr).decompose(units).value for attr in value.components]
    )


@convert_input_to_array.register(BaseRepresentation)
def _convert_from_representation(
    value: BaseRepresentation, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    if "s" in value.differentials and not kwargs.get("no_differentials", False):
        return xp.stack(
            (
                _convert_from_baserep(value, units=units),
                _convert_from_baserep(value.differentials["s"], units=units),
            )
        )
    return _convert_from_baserep(value, units=units)


##############################################################################
# Gala compatibility
# TODO: move this to an interoperability module

# isort: split
from galax.utils._optional_deps import HAS_GALA  # noqa: E402

if HAS_GALA:
    from gala.units import (
        DimensionlessUnitSystem as GalaDimensionlessUnitSystem,
        UnitSystem as GalaUnitSystem,
    )

    @converter_to_usys.register
    def _from_gala_dimensionless(
        value: GalaDimensionlessUnitSystem, /
    ) -> DimensionlessUnitSystem:
        return dimensionless

    @converter_to_usys.register
    def _from_gala(value: GalaUnitSystem, /) -> UnitSystem:
        usys = UnitSystem(*value._core_units)  # noqa: SLF001
        usys._registry = value._registry  # noqa: SLF001
        return usys
