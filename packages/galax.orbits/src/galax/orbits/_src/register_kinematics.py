"""Dispatches of `angular_speed` and `specific_angular_momentum`."""

__all__ = [
    "specific_angular_momentum",
    "angular_speed",
]


import functools as ft

import jax
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
from unxt.quantity import Quantity

import galax.coordinates as gc
import galax.orbits.custom_types as gt
from . import kinematics as api

# ===================================================================
# Specific angular momentum


@dispatch.multi(
    (gt.BBtSz3, gt.BBtSz3),
    (gt.BBtQuSz3, gt.BBtQuSz3),
)
@ft.partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: gt.BBtSz3 | gt.BBtQuSz3, v: gt.BBtSz3 | gt.BBtQuSz3, /
) -> gt.BBtSz3 | gt.BBtQuSz3:
    """Compute from `jax.Array` or `unxt.Quantity`s as Cartesian coordinates."""
    return jnp.linalg.cross(x, v)


@dispatch
@ft.partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: cx.vecs.AbstractPos3D, v: cx.vecs.AbstractVel3D, /
) -> cx.vecs.Cartesian3D:
    """Compute from `coordinax.vecs.AbstractVector`s."""
    v = convert(cx.vconvert(cx.CartesianVel3D, v, x), Quantity)
    x = convert(cx.vconvert(cx.CartesianPos3D, x), Quantity)
    h = api.specific_angular_momentum(x, v)
    return cx.vecs.Cartesian3D(x=h[..., 0], y=h[..., 1], z=h[..., 2])


@dispatch
@jax.jit
def specific_angular_momentum(w: cx.KinematicSpace, /) -> cx.vecs.Cartesian3D:
    """Compute from `coordinax.KinematicSpace`."""
    return api.specific_angular_momentum(w["length"], w["speed"])


@dispatch
@jax.jit
def specific_angular_momentum(
    w: cx.frames.AbstractCoordinate, /
) -> cx.vecs.Cartesian3D:
    """Compute from `coordinax.frames.AbstractCoordinate`."""
    return api.specific_angular_momentum(w.data)


@dispatch
@ft.partial(jax.jit, inline=True)
def specific_angular_momentum(w: gc.AbstractPhaseSpaceObject, /) -> cx.vecs.Cartesian3D:
    """Compute from `galax.coordinates.AbstractPhaseSpaceObject`."""
    return api.specific_angular_momentum(w.q, w.p)


# ===================================================================
# angular_speed


@dispatch.multi(
    (gt.BBtSz3, gt.BBtSz3),
    (gt.BBtQuSz3, gt.BBtQuSz3),
)
@jax.jit
def angular_speed(
    x: gt.BBtSz3 | gt.BBtQuSz3, v: gt.BBtSz3 | gt.BBtQuSz3, /
) -> gt.BBtSz0 | gt.BBtQuSz0:
    """Compute from `unxt.Quantity`s as Cartesian coordinates."""
    r = jnp.linalg.vector_norm(x, axis=-1, keepdims=True)
    om = jnp.linalg.cross(x, v) / r**2
    return jnp.linalg.vector_norm(om, axis=-1)


@dispatch
@jax.jit
def angular_speed(x: cx.vecs.AbstractPos3D, v: cx.vecs.AbstractVel3D, /) -> gt.BBtQuSz0:
    """Compute from `coordinax.vecs.AbstractVector`s."""
    # TODO: more directly using the vectors
    v = convert(cx.vconvert(cx.CartesianVel3D, v, x), Quantity)
    x = convert(cx.vconvert(cx.CartesianPos3D, x), Quantity)
    return api.angular_speed(x, v)


@dispatch
@jax.jit
def angular_speed(w: cx.KinematicSpace, /) -> gt.BBtQuSz0:
    """Compute from a `coordinax.KinematicSpace`."""
    return api.angular_speed(w["length"], w["speed"])


@dispatch
@jax.jit
def angular_speed(w: cx.frames.AbstractCoordinate, /) -> gt.BBtQuSz0:
    """Compute from a `coordinax.frames.AbstractCoordinate`."""
    return api.angular_speed(w.data)


@dispatch
@jax.jit
def angular_speed(w: gc.AbstractPhaseSpaceObject, /) -> gt.BBtQuSz0:
    """Compute from a `galax.coordinates.AbstractPhaseSpaceObject`."""
    return api.angular_speed(w.q, w.p)
