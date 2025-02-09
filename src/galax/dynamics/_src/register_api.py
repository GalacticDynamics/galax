"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "specific_angular_momentum",
    "omega",
]


from functools import partial

import jax
from jaxtyping import Shaped
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.typing as gt
from . import api

# ===================================================================
# Specific angular momentum


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: gt.LengthBtSz3, v: gt.SpeedBtSz3, /
) -> Shaped[u.Quantity["angular momentum"], "*batch 3"]:
    """Compute the specific angular momentum.

    Arguments:
    ---------
    x: Quantity[Any, (3,), "length"]
        3d Cartesian position (x, y, z).
    v: Quantity[Any, (3,), "speed"]
        3d Cartesian velocity (v_x, v_y, v_z).

    """
    return jnp.linalg.cross(x, v)


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: cx.vecs.AbstractPos3D, v: cx.vecs.AbstractVel3D, /
) -> gt.BtQuSz3:
    """Compute the specific angular momentum."""
    # TODO: keep as a vector.
    #       https://github.com/GalacticDynamics/vector/issues/27
    x = convert(x.vconvert(cx.CartesianPos3D), u.Quantity)
    v = convert(v.vconvert(cx.CartesianVel3D, x), u.Quantity)
    return api.specific_angular_momentum(x, v)


@dispatch
@partial(jax.jit)
def specific_angular_momentum(w: cx.Space, /) -> gt.BtQuSz3:
    """Compute the specific angular momentum."""
    # TODO: keep as a vector.
    #       https://github.com/GalacticDynamics/vector/issues/27
    return api.specific_angular_momentum(w["length"], w["speed"])


@dispatch
@partial(jax.jit)
def specific_angular_momentum(w: cx.frames.AbstractCoordinate, /) -> gt.BtQuSz3:
    """Compute the specific angular momentum."""
    # TODO: keep as a vector.
    #       https://github.com/GalacticDynamics/vector/issues/27
    return api.specific_angular_momentum(w.data)


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(w: gc.AbstractPhaseSpaceObject) -> gt.BtQuSz3:
    r"""Compute the specific angular momentum."""
    return api.specific_angular_momentum(w.q, w.p)


# ===================================================================
# omega


@dispatch
@partial(jax.jit, inline=True)
def omega(
    x: gt.LengthBtSz3, v: gt.SpeedBtSz3, /
) -> Shaped[u.Quantity["frequency"], "*batch"]:
    """Compute the orbital angular frequency about the origin.

    Arguments:
    ---------
    x: Quantity[Any, (3,), "length"]
        3d Cartesian position (x, y, z).
    v: Quantity[Any, (3,), "speed"]
        3d Cartesian velocity (v_x, v_y, v_z).

    Returns
    -------
    Quantity[Any, (3,), "frequency"]
        Angular velocity.

    Examples
    --------
    >>> import unxt as u

    >>> x = u.Quantity([8.0, 0.0, 0.0], "m")
    >>> v = u.Quantity([0.0, 8.0, 0.0], "m/s")
    >>> omega(x, v)
    Quantity['frequency'](Array(1., dtype=float64), unit='1 / s')
    """
    r = jnp.linalg.vector_norm(x, axis=-1, keepdims=True)
    omega = jnp.linalg.cross(x, v) / r**2
    return jnp.linalg.vector_norm(omega, axis=-1)


@dispatch
@partial(jax.jit, inline=True)
def omega(
    x: cx.vecs.AbstractPos3D, v: cx.vecs.AbstractVel3D, /
) -> Shaped[u.Quantity["frequency"], "*batch"]:
    """Compute the orbital angular frequency about the origin.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "m")
    >>> v = cx.CartesianVel3D.from_([0.0, 8.0, 0.0], "m/s")
    >>> omega(x, v)
    Quantity['frequency'](Array(1., dtype=float64), unit='1 / s')

    """
    # TODO: more directly using the vectors
    x = convert(x.vconvert(cx.CartesianPos3D), u.Quantity)
    v = convert(v.vconvert(cx.CartesianVel3D, x), u.Quantity)
    return omega(x, v)
