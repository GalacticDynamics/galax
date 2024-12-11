"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import Any, TypeVar

import jax.numpy as jnp
import numpy as np
from jax.dtypes import canonicalize_dtype
from jaxtyping import Array, Shaped
from plum import convert, dispatch

import coordinax as cx
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem

import galax.coordinates as gc
import galax.typing as gt

# --------------------------------------------------------------

Value = TypeVar("Value", int, float, Array)


@dispatch.abstract
def parse_to_quantity(value: Any, /, *, units: AbstractUnitSystem, **_: Any) -> Any:
    """Parse input arguments.

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


@dispatch
def parse_to_quantity(
    x: int | float | Array | np.ndarray, /, *, unit: gt.Unit, **_: Any
) -> Quantity:
    arr = jnp.asarray(x, dtype=None)
    dtype = jnp.promote_types(arr.dtype, canonicalize_dtype(float))
    return Quantity(jnp.asarray(arr, dtype=dtype), unit=unit)


@dispatch
def parse_to_quantity(
    x: gc.AbstractPhaseSpacePosition, /, **_: Any
) -> Shaped[Array, "*batch 3"]:
    return parse_to_quantity(x.q)


@dispatch
def parse_to_quantity(x: cx.vecs.AbstractPos3D, /, **_: Any) -> gt.LengthBatchVec3:
    cart = x.vconvert(cx.CartesianPos3D)
    qarr: Quantity = convert(cart, Quantity)
    return qarr


@dispatch
def parse_to_quantity(value: Quantity, /, **_: Any) -> Shaped[Quantity, "*#batch 3"]:
    return value
