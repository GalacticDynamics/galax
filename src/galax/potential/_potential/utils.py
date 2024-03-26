"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import singledispatch
from typing import Any, TypeVar

import jax.numpy as jnp
import jax.numpy as xp
import numpy as np
from astropy.coordinates import BaseRepresentation
from astropy.units import Quantity as APYQuantity
from jax.dtypes import canonicalize_dtype
from jaxtyping import Array, Shaped
from plum import convert, dispatch

import coordinax as cx
from unxt import Quantity
from unxt.unitsystems import DimensionlessUnitSystem, UnitSystem, dimensionless

from galax.coordinates import AbstractPhaseSpacePosition
from galax.typing import Unit

# --------------------------------------------------------------

Value = TypeVar("Value", int, float, Array)


@singledispatch
def parse_to_quantity(value: Any, /, *, units: UnitSystem, **_: Any) -> Any:
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


@parse_to_quantity.register(int)
@parse_to_quantity.register(float)
@parse_to_quantity.register(Array)
@parse_to_quantity.register(np.ndarray)
def _convert_from_arraylike(x: Any, /, *, unit: Unit, **_: Any) -> Quantity:
    arr = xp.asarray(x, dtype=None)
    dtype = jnp.promote_types(arr.dtype, canonicalize_dtype(float))
    return Quantity(xp.asarray(arr, dtype=dtype), unit=unit)


@parse_to_quantity.register(AbstractPhaseSpacePosition)
def _convert_from_psp(
    x: AbstractPhaseSpacePosition, /, **_: Any
) -> Shaped[Array, "*batch 3"]:
    return _convert_from_3dvec(x.q)


@parse_to_quantity.register(cx.Abstract3DVector)
def _convert_from_3dvec(x: cx.Abstract3DVector, /, **_: Any) -> gt.LengthBatchVec3:
    cart = x.represent_as(cx.Cartesian3DVector)
    qarr: Quantity = convert(cart, Quantity)
    return qarr


@parse_to_quantity.register(Quantity)
def _convert_from_quantity(
    value: Quantity, /, **_: Any
) -> Shaped[Quantity, "*#batch 3"]:
    return value


# ---------------------------
# Astropy compat


@parse_to_quantity.register(APYQuantity)
def _convert_from_astropy_quantity(value: APYQuantity, /, **_: Any) -> Array:
    return convert(value, Quantity)


@parse_to_quantity.register(BaseRepresentation)
def _convert_from_astropy_baserep(value: BaseRepresentation, /, **_: Any) -> Array:
    return _convert_from_3dvec(convert(value, cx.Cartesian3DVector))


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
