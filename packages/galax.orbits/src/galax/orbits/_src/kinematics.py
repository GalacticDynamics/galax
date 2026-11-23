"""Kinematics of phase-space objects: `omega` and `specific_angular_momentum`."""

__all__ = [
    "specific_angular_momentum",
    "omega",
]

from typing import Any

from plum import dispatch

import galax.orbits.custom_types as gt


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
    >>> import galax.orbits as go

    >>> x = jnp.array([8.0, 0.0, 0.0])  # [m]
    >>> v = jnp.array([0.0, 8.0, 0.0])  # [m/s]
    >>> go.specific_angular_momentum(x, v)
    Array([ 0.,  0., 64.], dtype=float64)

    >>> x = u.Q(x, "m")
    >>> v = u.Q(v, "m/s")
    >>> go.specific_angular_momentum(x, v)
    Q([ 0.,  0., 64.], 'm2 / s')

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)
    >>> h = go.specific_angular_momentum(q, p)
    >>> print(h)
    <Cartesian3D: (x, y, z) [m2 / s]
        [ 0.  0. 64.]>

    >>> space = cx.KinematicSpace(length=cx.CartesianPos3D.from_([[[7., 0, 0], [8, 0, 0]]], "m"),
    ...                  speed=cx.CartesianVel3D.from_([[[0., 5, 0], [0, 6, 0]]], "m/s"))
    >>> h = go.specific_angular_momentum(space)
    >>> print(h)
    <Cartesian3D: (x, y, z) [m2 / s]
        [[[ 0.  0. 35.]
          [ 0.  0. 48.]]]>

    >>> w = cx.frames.Coordinate(space, frame=gc.frames.simulation_frame)
    >>> h = go.specific_angular_momentum(w)
    >>> print(h)
    <Cartesian3D: (x, y, z) [m2 / s]
        [[[ 0.  0. 35.]
          [ 0.  0. 48.]]]>

    >>> w = gc.PhaseSpaceCoordinate(q=u.Q([1., 0, 0], "au"),
    ...                             p=u.Q([0, 2., 0], "au/yr"),
    ...                             t=u.Q(0, "yr"))
    >>> h = go.specific_angular_momentum(w)
    >>> print(h)
    <Cartesian3D: (x, y, z) [AU2 / yr]
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
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.orbits as go

    >>> x = jnp.array([8.0, 0.0, 0.0])  # [m]
    >>> v = jnp.array([0.0, 8.0, 0.0])  # [m/s]
    >>> go.omega(x, v)
    Array(1., dtype=float64)

    >>> x = u.Q(x, "m")
    >>> v = u.Q(v, "m/s")
    >>> go.omega(x, v)
    Q(1., '1 / s')

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)
    >>> go.omega(q, p)
    Q(1., '1 / s')

    >>> space = cx.KinematicSpace(length=q, speed=p)
    >>> go.omega(space)
    Q(1., '1 / s')

    >>> w = cx.frames.Coordinate(space, frame=gc.frames.simulation_frame)
    >>> go.omega(w)
    Q(1., '1 / s')

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=u.Q(0, "yr"))
    >>> go.omega(w)
    Q(1., '1 / s')

    """
    raise NotImplementedError  # pragma: no cover
