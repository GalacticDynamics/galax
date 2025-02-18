"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "specific_angular_momentum",
    "omega",
]


from functools import partial

import jax
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
from unxt.quantity import BareQuantity

import galax.coordinates as gc
import galax.typing as gt
from . import api

# ===================================================================
# Specific angular momentum


@dispatch.multi(
    (gt.BBtRealSz3, gt.BBtRealSz3),
    (gt.BBtRealQuSz3, gt.BBtRealQuSz3),
)
@partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: gt.BBtRealSz3 | gt.BBtRealQuSz3, v: gt.BBtRealSz3 | gt.BBtRealQuSz3, /
) -> gt.BBtRealSz3 | gt.BBtRealQuSz3:
    """Compute from `jax.Array` or `unxt.Quantity`s as Cartesian coordinates."""
    return jnp.linalg.cross(x, v)


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: cx.vecs.AbstractPos3D, v: cx.vecs.AbstractVel3D, /
) -> cx.vecs.CartesianGeneric3D:
    """Compute from `coordinax.vecs.AbstractVector`s."""
    v = convert(cx.vconvert(cx.CartesianVel3D, v, x), BareQuantity)
    x = convert(cx.vconvert(cx.CartesianPos3D, x), BareQuantity)
    h = api.specific_angular_momentum(x, v)
    return cx.vecs.CartesianGeneric3D(x=h[..., 0], y=h[..., 1], z=h[..., 2])


@dispatch
@partial(jax.jit)
def specific_angular_momentum(w: cx.Space, /) -> cx.vecs.CartesianGeneric3D:
    """Compute from `coordinax.Space`."""
    return api.specific_angular_momentum(w["length"], w["speed"])


@dispatch
@partial(jax.jit)
def specific_angular_momentum(
    w: cx.frames.AbstractCoordinate, /
) -> cx.vecs.CartesianGeneric3D:
    """Compute from `coordinax.frames.AbstractCoordinate`."""
    return api.specific_angular_momentum(w.data)


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(
    w: gc.AbstractPhaseSpaceObject, /
) -> cx.vecs.CartesianGeneric3D:
    """Compute from `galax.coordinates.AbstractPhaseSpaceObject`."""
    return api.specific_angular_momentum(w.q, w.p)


# ===================================================================
# omega


@dispatch.multi(
    (gt.BBtRealSz3, gt.BBtRealSz3),
    (gt.BBtRealQuSz3, gt.BBtRealQuSz3),
)
@partial(jax.jit)
def omega(
    x: gt.BBtRealSz3 | gt.BBtRealQuSz3, v: gt.BBtRealSz3 | gt.BBtRealQuSz3, /
) -> gt.BBtRealSz0 | gt.BBtRealQuSz0:
    """Compute from `unxt.Quantity`s as Cartesian coordinates."""
    r = jnp.linalg.vector_norm(x, axis=-1, keepdims=True)
    om = jnp.linalg.cross(x, v) / r**2
    return jnp.linalg.vector_norm(om, axis=-1)


@dispatch
@partial(jax.jit)
def omega(x: cx.vecs.AbstractPos3D, v: cx.vecs.AbstractVel3D, /) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractVector`s."""
    # TODO: more directly using the vectors
    v = convert(cx.vconvert(cx.CartesianVel3D, v, x), BareQuantity)
    x = convert(cx.vconvert(cx.CartesianPos3D, x), BareQuantity)
    return omega(x, v)


@dispatch
@partial(jax.jit)
def omega(w: cx.Space, /) -> gt.BBtRealQuSz0:
    """Compute from a `coordinax.Space`."""
    return omega(w["length"], w["speed"])


@dispatch
@partial(jax.jit)
def omega(w: cx.frames.AbstractCoordinate, /) -> gt.BBtRealQuSz0:
    """Compute from a `coordinax.frames.AbstractCoordinate`."""
    return omega(w.data)


@dispatch
@partial(jax.jit)
def omega(w: gc.AbstractPhaseSpaceObject, /) -> gt.BBtRealQuSz0:
    """Compute from a `galax.coordinates.AbstractPhaseSpaceObject`."""
    return omega(w.q, w.p)
