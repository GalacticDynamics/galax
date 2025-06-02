"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "specific_angular_momentum",
    "omega",
]

from typing import Any

from plum import dispatch

import galax._custom_types as gt


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

    >>> x = jnp.array([8.0, 0.0, 0.0])  # [m]
    >>> v = jnp.array([0.0, 8.0, 0.0])  # [m/s]
    >>> gd.specific_angular_momentum(x, v)
    Array([ 0.,  0., 64.], dtype=float64)

    >>> x = u.Quantity(x, "m")
    >>> v = u.Quantity(v, "m/s")
    >>> gd.specific_angular_momentum(x, v)
    Quantity[...](Array([ 0.,  0., 64.], dtype=float64), unit='m2 / s')

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)
    >>> h = gd.specific_angular_momentum(q, p)
    >>> print(h)
    <Cartesian3D (x[m2 / s], y[m2 / s], z[m2 / s])
        [ 0.  0. 64.]>

    >>> space = cx.Space(length=cx.CartesianPos3D.from_([[[7., 0, 0], [8, 0, 0]]], "m"),
    ...                  speed=cx.CartesianVel3D.from_([[[0., 5, 0], [0, 6, 0]]], "m/s"))
    >>> h = gd.specific_angular_momentum(space)
    >>> print(h)
    <Cartesian3D (x[m2 / s], y[m2 / s], z[m2 / s])
        [[[ 0.  0. 35.]
          [ 0.  0. 48.]]]>

    >>> w = cx.frames.Coordinate(space, frame=gc.frames.simulation_frame)
    >>> h = gd.specific_angular_momentum(w)
    >>> print(h)
    <Cartesian3D (x[m2 / s], y[m2 / s], z[m2 / s])
        [[[ 0.  0. 35.]
          [ 0.  0. 48.]]]>

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1., 0, 0], "au"),
    ...                             p=u.Quantity([0, 2., 0], "au/yr"),
    ...                             t=u.Quantity(0, "yr"))
    >>> h = gd.specific_angular_momentum(w)
    >>> print(h)
    <Cartesian3D (x[AU2 / yr], y[AU2 / yr], z[AU2 / yr])
        [0. 0. 2.]>

    """  # noqa: E501
    raise NotImplementedError  # pragma: no cover


# ===================================================================
# Omega


@dispatch.abstract
def omega(x: Any, v: Any, /) -> gt.BBtQuSz0:
    """Compute the orbital angular frequency about the origin.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = jnp.array([8.0, 0.0, 0.0])  # [m]
    >>> v = jnp.array([0.0, 8.0, 0.0])  # [m/s]
    >>> omega(x, v)
    Array(1., dtype=float64)

    >>> x = u.Quantity(x, "m")
    >>> v = u.Quantity(v, "m/s")
    >>> omega(x, v)
    Quantity['frequency'](Array(1., dtype=float64), unit='1 / s')

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)
    >>> omega(q, p)
    BareQuantity(Array(1., dtype=float64), unit='1 / s')

    >>> space = cx.Space(length=q, speed=p)
    >>> omega(space)
    BareQuantity(Array(1., dtype=float64), unit='1 / s')

    >>> w = cx.frames.Coordinate(space, frame=gc.frames.simulation_frame)
    >>> omega(w)
    BareQuantity(Array(1., dtype=float64), unit='1 / s')

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=u.Quantity(0, "yr"))
    >>> omega(w)
    BareQuantity(Array(1., dtype=float64), unit='1 / s')

    """
    raise NotImplementedError  # pragma: no cover
