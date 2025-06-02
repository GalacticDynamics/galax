"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "specific_angular_momentum",
    "omega",
]


import functools as ft

import jax
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
from unxt.quantity import BareQuantity

import galax._custom_types as gt
import galax.coordinates as gc
from . import api

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
    v = convert(cx.vconvert(cx.CartesianVel3D, v, x), BareQuantity)
    x = convert(cx.vconvert(cx.CartesianPos3D, x), BareQuantity)
    h = api.specific_angular_momentum(x, v)
    return cx.vecs.Cartesian3D(x=h[..., 0], y=h[..., 1], z=h[..., 2])


@dispatch
@ft.partial(jax.jit)
def specific_angular_momentum(w: cx.Space, /) -> cx.vecs.Cartesian3D:
    """Compute from `coordinax.Space`."""
    return api.specific_angular_momentum(w["length"], w["speed"])


@dispatch
@ft.partial(jax.jit)
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
# omega


@dispatch.multi(
    (gt.BBtSz3, gt.BBtSz3),
    (gt.BBtQuSz3, gt.BBtQuSz3),
)
@ft.partial(jax.jit)
def omega(
    x: gt.BBtSz3 | gt.BBtQuSz3, v: gt.BBtSz3 | gt.BBtQuSz3, /
) -> gt.BBtSz0 | gt.BBtQuSz0:
    """Compute from `unxt.Quantity`s as Cartesian coordinates."""
    r = jnp.linalg.vector_norm(x, axis=-1, keepdims=True)
    om = jnp.linalg.cross(x, v) / r**2
    return jnp.linalg.vector_norm(om, axis=-1)


@dispatch
@ft.partial(jax.jit)
def omega(x: cx.vecs.AbstractPos3D, v: cx.vecs.AbstractVel3D, /) -> gt.BBtQuSz0:
    """Compute from `coordinax.vecs.AbstractVector`s."""
    # TODO: more directly using the vectors
    v = convert(cx.vconvert(cx.CartesianVel3D, v, x), BareQuantity)
    x = convert(cx.vconvert(cx.CartesianPos3D, x), BareQuantity)
    return api.omega(x, v)


@dispatch
@ft.partial(jax.jit)
def omega(w: cx.Space, /) -> gt.BBtQuSz0:
    """Compute from a `coordinax.Space`."""
    return api.omega(w["length"], w["speed"])


@dispatch
@ft.partial(jax.jit)
def omega(w: cx.frames.AbstractCoordinate, /) -> gt.BBtQuSz0:
    """Compute from a `coordinax.frames.AbstractCoordinate`."""
    return api.omega(w.data)


@dispatch
@ft.partial(jax.jit)
def omega(w: gc.AbstractPhaseSpaceObject, /) -> gt.BBtQuSz0:
    """Compute from a `galax.coordinates.AbstractPhaseSpaceObject`."""
    return api.omega(w.q, w.p)
