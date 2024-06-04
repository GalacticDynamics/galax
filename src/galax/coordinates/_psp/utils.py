"""Utilities for phase-space positions."""

__all__: list[str] = []

from collections.abc import Sequence
from functools import partial, singledispatch
from typing import Any, Protocol, cast, runtime_checkable

import astropy.coordinates as apyc
import jax
from jaxtyping import Array, Shaped

import coordinax as cx
import quaxed.array_api as xp
from unxt import Quantity

import galax.typing as gt


@partial(jax.jit, static_argnames="axis")
def interleave_concat(
    arrays: Sequence[Shaped[Array, "shape"]] | Sequence[Shaped[Quantity, "shape"]],
    /,
    axis: int,
) -> Shaped[Array, "..."] | Shaped[Quantity, "..."]:  # TODO: shape hint
    # Check if input is a non-empty list
    if not arrays or not isinstance(arrays, Sequence):
        msg = "Input should be a non-empty sequence of arrays."
        raise ValueError(msg)

    # Ensure all arrays have the same shape
    shape0 = arrays[0].shape
    if not all(arr.shape == shape0 for arr in arrays):
        msg = "All arrays must have the same shape."
        raise ValueError(msg)

    # Stack the arrays along a new axis to prepare for interleaving
    axis = axis % len(shape0)  # allows for negative axis
    stacked = xp.stack(arrays, axis=axis + 1)

    # Flatten the new axis by interleaving values
    return xp.reshape(
        stacked, (*shape0[:axis], len(arrays) * shape0[axis], *shape0[axis + 1 :])
    )


@runtime_checkable
class HasShape(Protocol):
    """Protocol for an object with a shape attribute."""

    shape: gt.Shape


def _getitem_broadscalartime_index_tuple(
    index: tuple[Any, ...],
    t: gt.FloatQAnyShape,  # noqa: ARG001
) -> Any:
    """Get the time index from a slice."""
    if len(index) == 0:  # slice is an empty tuple
        return slice(None)
    return index


def getitem_broadscalartime_index(index: Any, t: gt.FloatQAnyShape) -> Any:
    """Get the time index from an index."""
    if isinstance(index, tuple):
        return _getitem_broadscalartime_index_tuple(index, t)
    return index


# -----------------------------------------------------------------------------


def _getitem_vec1time_index_tuple(index: tuple[Any, ...], t: gt.FloatQAnyShape) -> Any:
    """Get the time index from a slice."""
    if len(index) == 0:  # slice is an empty tuple
        return slice(None)
    if t.ndim == 1:  # slicing a Vec1
        return slice(None)
    if len(index) >= t.ndim:
        msg = f"Index {index} has too many dimensions for time array of shape {t.shape}"
        raise IndexError(msg)
    return index


def _getitem_vec1time_index_shaped(index: HasShape, t: gt.FloatQAnyShape) -> HasShape:
    """Get the time index from a shaped index array."""
    if t.ndim == 1:  # Vec1
        return cast(HasShape, xp.asarray([True]))
    if len(index.shape) >= t.ndim:
        msg = f"Index {index} has too many dimensions for time array of shape {t.shape}"
        raise IndexError(msg)
    return index


def getitem_vec1time_index(index: Any, t: gt.FloatQAnyShape) -> Any:
    """Get the time index from an index.

    Parameters
    ----------
    index : Any
        The index to get the time index from.
    t : FloatQAnyShape
        The time array.

    Returns
    -------
    Any
        The time index.

    Examples
    --------
    We set up a time array.
    >>> import jax.numpy as jnp
    >>> from unxt import Quantity
    >>> t = Quantity(jnp.ones((10, 3), dtype=float), "s")

    Some standard indexes.
    >>> getitem_vec1time_index(0, t)
    0

    >>> getitem_vec1time_index(slice(0, 10), t)
    slice(0, 10, None)

    Tuples:
    >>> getitem_vec1time_index((0,), t)
    (0,)

    >>> t = Quantity(jnp.ones((1, 2, 3), dtype=float), "s")
    >>> getitem_vec1time_index((0, 1), t)
    (0, 1)

    Shaped:
    >>> import jax.numpy as jnp
    >>> index = jnp.asarray([True, False, True])
    >>> getitem_vec1time_index(index, t)
    Array([ True, False,  True], dtype=bool)
    """
    if isinstance(index, tuple):
        return _getitem_vec1time_index_tuple(index, t)
    if isinstance(index, HasShape):
        return _getitem_vec1time_index_shaped(index, t)
    return index


# -----------------------------------------------------------------------------


@singledispatch
def _q_converter(x: Any) -> cx.AbstractPosition3D:
    """Convert input to a 3D vector."""
    return cx.CartesianPosition3D.constructor(x)


@_q_converter.register
def _q_converter_vec(x: cx.AbstractPosition3D) -> cx.AbstractPosition3D:
    return x


# TODO: move this into coordinax
_apyc_to_cx_vecs = {
    apyc.CartesianRepresentation: cx.CartesianPosition3D,
    apyc.CylindricalRepresentation: cx.CylindricalPosition,
    apyc.SphericalRepresentation: cx.LonLatSphericalPosition,
    apyc.PhysicsSphericalRepresentation: cx.SphericalPosition,
}


@_q_converter.register
def _q_converter_apy(x: apyc.BaseRepresentation) -> cx.AbstractPosition3D:
    return _apyc_to_cx_vecs[type(x)].constructor(x)


# -----------------------------------------------------------------------------


@singledispatch
def _p_converter(x: Any) -> cx.AbstractVelocity3D:
    """Convert input to a 3D vector differential."""
    return cx.CartesianVelocity3D.constructor(x)


@_p_converter.register
def _p_converter_vec(
    x: cx.AbstractVelocity3D,
) -> cx.AbstractVelocity3D:
    return x


# TODO: move this into coordinax
_apyc_to_cx_difs = {
    apyc.CartesianDifferential: cx.CartesianVelocity3D,
    apyc.CylindricalDifferential: cx.CylindricalVelocity,
    apyc.SphericalDifferential: cx.LonLatSphericalVelocity,
    apyc.SphericalCosLatDifferential: cx.LonCosLatSphericalVelocity,
    apyc.PhysicsSphericalDifferential: cx.SphericalVelocity,
}


@_p_converter.register
def _p_converter_apy(x: apyc.BaseDifferential) -> cx.AbstractVelocity3D:
    return _apyc_to_cx_difs[type(x)].constructor(x)
