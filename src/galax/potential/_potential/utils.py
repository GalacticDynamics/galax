"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import singledispatch
from typing import Any, TypeVar

import jax.numpy as jnp
import jax.numpy as xp
import numpy as np
from astropy.coordinates import BaseRepresentation, BaseRepresentationOrDifferential
from astropy.units import Quantity as APYQuantity
from jax.dtypes import canonicalize_dtype
from jaxtyping import Array, Shaped
from plum import convert, dispatch

from coordinax import Abstract3DVector, Cartesian3DVector
from unxt import Quantity

from galax.units import DimensionlessUnitSystem, UnitSystem, dimensionless


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
def convert_input_to_array(value: Any, /, *, units: UnitSystem, **_: Any) -> Any:
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
@convert_input_to_array.register(np.ndarray)
def _convert_from_arraylike(
    x: Any,
    /,
    *,
    units: UnitSystem,  # noqa: ARG001
    **_: Any,
) -> Array:
    arr = xp.asarray(x, dtype=None)
    dtype = jnp.promote_types(arr.dtype, canonicalize_dtype(float))
    return xp.asarray(arr, dtype=dtype)


@convert_input_to_array.register(Abstract3DVector)
def _convert_from_3dvec(
    x: Abstract3DVector, /, *, units: UnitSystem, **_: Any
) -> Shaped[Array, "*batch 3"]:
    cart = x.represent_as(Cartesian3DVector)
    qarr: Quantity = convert(cart, Quantity).decompose(units)  # in right units
    return qarr.value


@convert_input_to_array.register(Quantity)
def _convert_from_quantity(
    value: Quantity, /, *, units: UnitSystem, **_: Any
) -> Shaped[Array, "*#batch 3"]:
    return xp.asarray(value.decompose(units).value)


# ---------------------------
# Astropy compat


@convert_input_to_array.register(APYQuantity)
def _convert_from_astropy_quantity(
    value: APYQuantity, /, *, units: UnitSystem, **_: Any
) -> Array:
    return xp.asarray(value.decompose(units).value)


@convert_input_to_array.register(BaseRepresentationOrDifferential)
def _convert_from_astropy_baserep(
    value: BaseRepresentationOrDifferential, /, *, units: UnitSystem, **_: Any
) -> Array:
    return xp.stack(
        [getattr(value, attr).decompose(units).value for attr in value.components],
        axis=-1,
    )


@convert_input_to_array.register(BaseRepresentation)
def _convert_from_astropy_representation(
    value: BaseRepresentation, /, *, units: UnitSystem, **kwargs: Any
) -> Array:
    value = value.to_cartesian()
    if "s" in value.differentials and not kwargs.get("no_differentials", False):
        return xp.concat(
            (
                _convert_from_astropy_baserep(value, units=units),
                _convert_from_astropy_baserep(value.differentials["s"], units=units),
            ),
            axis=-1,
        )
    return _convert_from_astropy_baserep(value, units=units)


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

    @dispatch
    def unitsystem(value: GalaUnitSystem, /) -> UnitSystem:
        usys = UnitSystem(*value._core_units)  # noqa: SLF001
        usys._registry = value._registry  # noqa: SLF001
        return usys

    @dispatch  # type: ignore[no-redef]
    def unitsystem(_: GalaDimensionlessUnitSystem, /) -> DimensionlessUnitSystem:
        return dimensionless
