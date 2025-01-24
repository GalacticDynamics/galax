"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "specific_angular_momentum",
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

    Returns
    -------
    Quantity[Any, (3,), "angular momentum"]
        Specific angular momentum.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics as gd

    >>> x = u.Quantity([8.0, 0.0, 0.0], "m")
    >>> v = u.Quantity([0.0, 8.0, 0.0], "m/s")
    >>> gd.specific_angular_momentum(x, v)
    Quantity['diffusivity'](Array([ 0.,  0., 64.], dtype=float64), unit='m2 / s')

    """
    return jnp.linalg.cross(x, v)


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: cx.vecs.AbstractPos3D, v: cx.vecs.AbstractVel3D, /
) -> gt.BtQuSz3:
    """Compute the specific angular momentum.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.dynamics as gd

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "m")
    >>> v = cx.CartesianVel3D.from_([0.0, 8.0, 0.0], "m/s")
    >>> gd.specific_angular_momentum(x, v)
    Quantity['diffusivity'](Array([ 0.,  0., 64.], dtype=float64), unit='m2 / s')

    """
    # TODO: keep as a vector.
    #       https://github.com/GalacticDynamics/vector/issues/27
    x = convert(x.vconvert(cx.CartesianPos3D), u.Quantity)
    v = convert(v.vconvert(cx.CartesianVel3D, x), u.Quantity)
    return specific_angular_momentum(x, v)


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(w: cx.Space) -> gt.BtQuSz3:
    """Compute the specific angular momentum.

    Examples
    --------
    >>> import coordinax as cx
    >>> w = cx.Space(length=cx.CartesianPos3D.from_([[[7., 0, 0], [8, 0, 0]]], "m"),
    ...              speed=cx.CartesianVel3D.from_([[[0., 5, 0], [0, 6, 0]]], "m/s"))

    >>> specific_angular_momentum(w)
    Quantity['diffusivity'](Array([[[ 0.,  0., 35.], [ 0.,  0., 48.]]], dtype=float64), unit='m2 / s')

    """  # noqa: E501
    # TODO: keep as a vector.
    #       https://github.com/GalacticDynamics/vector/issues/27
    return specific_angular_momentum(w["length"], w["speed"])


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(w: gc.AbstractPhaseSpacePosition) -> gt.BtQuSz3:
    r"""Compute the specific angular momentum.

    .. math::

        \boldsymbol{{L}} = \boldsymbol{{q}} \times \boldsymbol{{p}}

    Returns
    -------
    L : Quantity[float, (*batch,3)]
        Array of angular momentum vectors in Cartesian coordinates.

    Examples
    --------
    We assume the following imports

    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd

    We can compute the angular momentum of a single object

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1., 0, 0], "au"),
    ...                           p=u.Quantity([0, 2., 0], "au/yr"),
    ...                           t=u.Quantity(0, "yr"))
    >>> gd.specific_angular_momentum(w)
    Quantity[...](Array([0., 0., 2.], dtype=float64), unit='AU2 / yr')
    """
    return specific_angular_momentum(w.q, w.p)
