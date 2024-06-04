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
from plum import convert

import coordinax as cx
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem

import galax.typing as gt
from galax.coordinates import AbstractPhaseSpacePosition

# --------------------------------------------------------------

Value = TypeVar("Value", int, float, Array)


@singledispatch
def parse_to_quantity(value: Any, /, *, units: AbstractUnitSystem, **_: Any) -> Any:
    """Parse input arguments.

    This function uses :func:`~functools.singledispatch` to dispatch on the type
    of the input argument.

    Parameters
    ----------
    value : Any, positional-only
        Input value.
    units : AbstractUnitSystem, keyword-only
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
def _convert_from_arraylike(x: Any, /, *, unit: gt.Unit, **_: Any) -> Quantity:
    arr = xp.asarray(x, dtype=None)
    dtype = jnp.promote_types(arr.dtype, canonicalize_dtype(float))
    return Quantity(xp.asarray(arr, dtype=dtype), unit=unit)


@parse_to_quantity.register(AbstractPhaseSpacePosition)
def _convert_from_psp(
    x: AbstractPhaseSpacePosition, /, **_: Any
) -> Shaped[Array, "*batch 3"]:
    return _convert_from_3dvec(x.q)


@parse_to_quantity.register(cx.AbstractPosition3D)
def _convert_from_3dvec(x: cx.AbstractPosition3D, /, **_: Any) -> gt.LengthBatchVec3:
    cart = x.represent_as(cx.CartesianPosition3D)
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
    return _convert_from_3dvec(convert(value, cx.CartesianPosition3D))
