"""Functional API for galax potentials."""

__all__ = [
    "potential",
    "gradient",
    "laplacian",
    "density",
    "hessian",
    "acceleration",
    "tidal_tensor",
    "circular_velocity",
]

from typing import Any

from plum import dispatch

import coordinax as cx
import unxt as u

import galax.typing as gt


@dispatch.abstract
def potential(*args: Any, **kwargs: Any) -> u.Quantity["specific energy"]:
    """Compute the potential energy at the given position(s)."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def gradient(*args: Any, **kwargs: Any) -> cx.vecs.CartesianAcc3D:
    """Compute the gradient of the potential at the given position(s)."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def laplacian(*args: Any, **kwargs: Any) -> u.Quantity["1/s^2"]:
    """Compute the laplacian of the potential at the given position(s)."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def density(*args: Any, **kwargs: Any) -> u.Quantity["mass density"]:
    """Compute the density at the given position(s)."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def hessian(*args: Any, **kwargs: Any) -> gt.BtQuSz33:
    """Compute the hessian of the potential at the given position(s)."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def acceleration(*args: Any, **kwargs: Any) -> cx.vecs.CartesianAcc3D:
    """Compute the acceleration due to the potential at the given position(s)."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def tidal_tensor(*args: Any, **kwargs: Any) -> gt.BtQuSz33:
    """Compute the tidal tensor."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def circular_velocity(*args: Any, **kwargs: Any) -> gt.BBtRealQuSz0:
    """Estimate the circular velocity at the given position."""
    raise NotImplementedError  # pragma: no cover
