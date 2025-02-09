"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "specific_angular_momentum",
    "omega",
]

from functools import partial
from typing import Any

import jax
from jaxtyping import Shaped
from plum import dispatch

import unxt as u

import galax.typing as gt


@dispatch.abstract
def specific_angular_momentum(
    *args: Any, **kwargs: Any
) -> Shaped[u.Quantity["angular momentum"], "*batch 3"]:
    """Compute the specific angular momentum.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd

    >>> x = u.Quantity([8.0, 0.0, 0.0], "m")
    >>> v = u.Quantity([0.0, 8.0, 0.0], "m/s")
    >>> gd.specific_angular_momentum(x, v)
    Quantity[...](Array([ 0.,  0., 64.], dtype=float64), unit='m2 / s')

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "m")
    >>> v = cx.CartesianVel3D.from_([0.0, 8.0, 0.0], "m/s")
    >>> gd.specific_angular_momentum(x, v)
    Quantity[...](Array([ 0.,  0., 64.], dtype=float64), unit='m2 / s')

    >>> space = cx.Space(length=cx.CartesianPos3D.from_([[[7., 0, 0], [8, 0, 0]]], "m"),
    ...                  speed=cx.CartesianVel3D.from_([[[0., 5, 0], [0, 6, 0]]], "m/s"))
    >>> gd.specific_angular_momentum(space)
    Quantity[...](Array([[[ 0., 0., 35.], [ 0., 0., 48.]]], dtype=float64), unit='m2 / s')

    >>> w = cx.frames.Coordinate(space, frame=gc.frames.SimulationFrame())
    >>> w
    Coordinate(
        data=Space({
            'length': CartesianPos3D( ... ),
            'speed': CartesianVel3D( ... )
        }),
        frame=SimulationFrame()
    )

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1., 0, 0], "au"),
    ...                             p=u.Quantity([0, 2., 0], "au/yr"),
    ...                             t=u.Quantity(0, "yr"))
    >>> gd.specific_angular_momentum(w)
    Quantity[...](Array([0., 0., 2.], dtype=float64), unit='AU2 / yr')

    """  # noqa: E501
    raise NotImplementedError  # pragma: no cover


# ===================================================================
# Omega


@dispatch
@partial(jax.jit, inline=True)
def omega(
    x: gt.LengthBtSz3, v: gt.SpeedBtSz3, /
) -> Shaped[u.Quantity["frequency"], "*batch"]:
    """Compute the orbital angular frequency about the origin.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = u.Quantity([8.0, 0.0, 0.0], "m")
    >>> v = u.Quantity([0.0, 8.0, 0.0], "m/s")
    >>> omega(x, v)
    Quantity['frequency'](Array(1., dtype=float64), unit='1 / s')

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "m")
    >>> v = cx.CartesianVel3D.from_([0.0, 8.0, 0.0], "m/s")
    >>> omega(x, v)
    Quantity['frequency'](Array(1., dtype=float64), unit='1 / s')

    """
    raise NotImplementedError  # pragma: no cover
