"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import singledispatch
from typing import Any, Protocol, cast, runtime_checkable

import astropy.coordinates as apyc
from jaxtyping import Shaped

import coordinax as cx
import quaxed.array_api as xp

import galax.typing as gt


@runtime_checkable
class HasShape(Protocol):
    """Protocol for a shaped object."""

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


def _getitem_vec1time_index_shaped(index: Shaped, t: gt.FloatQAnyShape) -> Shaped:
    """Get the time index from a shaped index array."""
    if t.ndim == 1:  # Vec1
        return cast(Shaped, xp.asarray([True]))
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
def _q_converter(x: Any) -> cx.Abstract3DVector:
    """Convert input to a 3D vector."""
    return cx.Cartesian3DVector.constructor(x)


@_q_converter.register
def _q_converter_vec(x: cx.Abstract3DVector) -> cx.Abstract3DVector:
    return x


# TODO: move this into coordinax
_apyc_to_cx_vecs = {
    apyc.CartesianRepresentation: cx.Cartesian3DVector,
    apyc.CylindricalRepresentation: cx.CylindricalVector,
    apyc.SphericalRepresentation: cx.LonLatSphericalVector,
    apyc.PhysicsSphericalRepresentation: cx.SphericalVector,
}


@_q_converter.register
def _q_converter_apy(x: apyc.BaseRepresentation) -> cx.Abstract3DVector:
    return _apyc_to_cx_vecs[type(x)].constructor(x)


# -----------------------------------------------------------------------------


@singledispatch
def _p_converter(x: Any) -> cx.Abstract3DVectorDifferential:
    """Convert input to a 3D vector differential."""
    return cx.CartesianDifferential3D.constructor(x)


@_p_converter.register
def _p_converter_vec(
    x: cx.Abstract3DVectorDifferential,
) -> cx.Abstract3DVectorDifferential:
    return x


# TODO: move this into coordinax
_apyc_to_cx_difs = {
    apyc.CartesianDifferential: cx.CartesianDifferential3D,
    apyc.CylindricalDifferential: cx.CylindricalDifferential,
    apyc.SphericalDifferential: cx.LonLatSphericalDifferential,
    apyc.SphericalCosLatDifferential: cx.LonCosLatSphericalDifferential,
    apyc.PhysicsSphericalDifferential: cx.SphericalDifferential,
}


@_p_converter.register
def _p_converter_apy(x: apyc.BaseDifferential) -> cx.Abstract3DVectorDifferential:
    return _apyc_to_cx_difs[type(x)].constructor(x)
