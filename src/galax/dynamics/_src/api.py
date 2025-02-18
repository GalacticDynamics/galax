"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "specific_angular_momentum",
    "omega",
]

from typing import Any

from plum import dispatch

import galax.typing as gt


@dispatch.abstract
def specific_angular_momentum(*args: Any, **kwargs: Any) -> Any:
    r"""Compute the specific angular momentum.

    The specific angular momentum $\mathbf{j}$ is given by:

    ..math::

        \mathbf{h} = \mathbf{r} \times \mathbf{v}


    where $\mathbf{r}$ is the position vector and $\mathbf{v}$ is the velocity
    vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd

    >>> x = jnp.asarray([8.0, 0.0, 0.0])
    >>> v = jnp.asarray([0.0, 8.0, 0.0])
    >>> gd.specific_angular_momentum(x, v)
    Array([ 0.,  0., 64.], dtype=float64)

    >>> x = u.Quantity([8.0, 0.0, 0.0], "m")
    >>> v = u.Quantity([0.0, 8.0, 0.0], "m/s")
    >>> gd.specific_angular_momentum(x, v)
    Quantity[...](Array([ 0.,  0., 64.], dtype=float64), unit='m2 / s')

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "m")
    >>> v = cx.CartesianVel3D.from_([0.0, 8.0, 0.0], "m/s")
    >>> h = gd.specific_angular_momentum(x, v)
    >>> print(h)
    <CartesianGeneric3D (x[m2 / s], y[m2 / s], z[m2 / s])
        [ 0.  0. 64.]>

    >>> space = cx.Space(length=cx.CartesianPos3D.from_([[[7., 0, 0], [8, 0, 0]]], "m"),
    ...                  speed=cx.CartesianVel3D.from_([[[0., 5, 0], [0, 6, 0]]], "m/s"))
    >>> h = gd.specific_angular_momentum(space)
    >>> print(h)
    <CartesianGeneric3D (x[m2 / s], y[m2 / s], z[m2 / s])
        [[[ 0.  0. 35.]
          [ 0.  0. 48.]]]>

    >>> w = cx.frames.Coordinate(space, frame=gc.frames.SimulationFrame())
    >>> h = gd.specific_angular_momentum(w)
    >>> print(h)
    <CartesianGeneric3D (x[m2 / s], y[m2 / s], z[m2 / s])
        [[[ 0.  0. 35.]
          [ 0.  0. 48.]]]>

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1., 0, 0], "au"),
    ...                             p=u.Quantity([0, 2., 0], "au/yr"),
    ...                             t=u.Quantity(0, "yr"))
    >>> h = gd.specific_angular_momentum(w)
    >>> print(h)
    <CartesianGeneric3D (x[AU2 / yr], y[AU2 / yr], z[AU2 / yr])
        [0. 0. 2.]>

    """  # noqa: E501
    raise NotImplementedError  # pragma: no cover


# ===================================================================
# Omega


@dispatch.abstract
def omega(x: Any, v: Any, /) -> gt.BBtRealQuSz0:
    """Compute the orbital angular frequency about the origin.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = jnp.asarray([8.0, 0.0, 0.0])
    >>> v = jnp.asarray([0.0, 8.0, 0.0])
    >>> omega(x, v)
    Array(1., dtype=float64)

    >>> x = u.Quantity([8.0, 0.0, 0.0], "m")
    >>> v = u.Quantity([0.0, 8.0, 0.0], "m/s")
    >>> omega(x, v)
    Quantity['frequency'](Array(1., dtype=float64), unit='1 / s')

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "m")
    >>> v = cx.CartesianVel3D.from_([0.0, 8.0, 0.0], "m/s")
    >>> omega(x, v)
    BareQuantity(Array(1., dtype=float64), unit='1 / s')

    >>> space = cx.Space(length=x, speed=v)
    >>> omega(space)
    BareQuantity(Array(1., dtype=float64), unit='1 / s')

    >>> w = cx.frames.Coordinate(space, frame=gc.frames.SimulationFrame())
    >>> omega(w)
    BareQuantity(Array(1., dtype=float64), unit='1 / s')

    >>> w = gc.PhaseSpaceCoordinate(q=x, p=v, t=u.Quantity(0, "yr"))
    >>> omega(w)
    BareQuantity(Array(1., dtype=float64), unit='1 / s')

    """
    raise NotImplementedError  # pragma: no cover
