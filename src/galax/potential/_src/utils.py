"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from typing import Any

import numpy as np
from jax.dtypes import canonicalize_dtype
from jaxtyping import ArrayLike
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AbstractQuantity
from unxt.unitsystems import AbstractUnitSystem

import galax.coordinates as gc
import galax.typing as gt


def parse_dtypes(dtype2: np.dtype, dtype1: Any, /) -> np.dtype | None:
    return (
        dtype2
        if dtype1 is None
        else jnp.promote_types(dtype2, canonicalize_dtype(dtype1))
    )


# ==============================================================================


@dispatch.abstract
def parse_to_quantity_or_array(
    value: Any, /, *, units: AbstractUnitSystem, **_: Any
) -> Any:
    """Parse input arguments."""
    msg = f"cannot convert {value} using units {units}"  # pragma: no cover
    raise NotImplementedError(msg)  # pragma: no cover


@dispatch
def parse_to_quantity_or_array(
    x: ArrayLike, /, *, dtype: Any = None, **_: Any
) -> gt.BtRealSz3:
    arr = jnp.asarray(x, dtype=None)
    dtype = parse_dtypes(arr.dtype, dtype)
    return jnp.asarray(arr, dtype=dtype)


@dispatch
def parse_to_quantity_or_array(
    q: AbstractQuantity, /, *, dtype: Any = None, **_: Any
) -> gt.BtRealQuSz3:
    return jnp.asarray(q, dtype=parse_dtypes(q.dtype, dtype))


@dispatch
def parse_to_quantity_or_array(
    x: cx.vecs.AbstractPos3D, /, **kw: Any
) -> gt.BtRealQuSz3:
    cart = x.vconvert(cx.CartesianPos3D)
    q = convert(cart, u.Quantity)
    return parse_to_quantity_or_array(q, **kw)


@dispatch
def parse_to_quantity_or_array(
    coord: cx.vecs.FourVector, /, **kw: Any
) -> gt.BtRealQuSz3:
    return parse_to_quantity_or_array(coord.q, **kw)


@dispatch
def parse_to_quantity_or_array(space: cx.vecs.Space, /, **kw: Any) -> gt.BtRealQuSz3:
    return parse_to_quantity_or_array(space["length"], **kw)


@dispatch
def parse_to_quantity_or_array(
    coord: cx.frames.AbstractCoordinate, /, **kw: Any
) -> gt.BtRealQuSz3:
    coord = coord.to_frame(gc.frames.SimulationFrame())  # TODO: frame
    return parse_to_quantity_or_array(coord.data, **kw)


@dispatch
def parse_to_quantity_or_array(
    coord: gc.AbstractPhaseSpaceObject, /, **kw: Any
) -> gt.BtRealQuSz3:
    coord = coord.to_frame(gc.frames.SimulationFrame())  # TODO: frame
    return parse_to_quantity_or_array(coord.q, **kw)


# ==============================================================================


# TODO: defer most to parse_to_quantity_or_array
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
    msg = f"cannot convert {value} using units {units}"  # pragma: no cover
    raise NotImplementedError(msg)  # pragma: no cover


@dispatch
def parse_to_quantity(
    q: AbstractQuantity, /, *, dtype: Any = None, **_: Any
) -> gt.BtRealQuSz3:
    return parse_to_quantity_or_array(q, dtype=dtype)


@dispatch
def parse_to_quantity(
    x: ArrayLike, /, *, dtype: Any = None, unit: gt.Unit, **_: Any
) -> gt.BtRealQuSz3:
    arr = jnp.asarray(x, dtype=None)
    dtype = parse_dtypes(arr.dtype, dtype)
    return u.Quantity(jnp.asarray(arr, dtype=dtype), unit=unit)


@dispatch
def parse_to_quantity(
    x: gc.AbstractPhaseSpaceObject | cx.vecs.FourVector, /, **kw: Any
) -> gt.BtRealQuSz3:
    return parse_to_quantity(x.q, **kw)


@dispatch
def parse_to_quantity(x: cx.vecs.AbstractPos3D, /, **kw: Any) -> gt.BtRealQuSz3:
    cart = x.vconvert(cx.CartesianPos3D)
    q = convert(cart, u.Quantity)
    return parse_to_quantity(q, **kw)
