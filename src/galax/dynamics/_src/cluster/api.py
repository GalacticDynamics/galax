"""The primary functional API. Private module.

Also see:

- `galax.dynamics.cluster`: The public API.
- `galax.dynamics.cluster.radius`: more tidal radii functions.


"""

__all__ = [
    # Lagrange points
    "lagrange_points",
    "L1L2LagrangePoints",
    # Times
    "relaxation_time",
    # Radius
    "tidal_radius",
]

from typing import Any, Generic, NamedTuple, TypeVar

from plum import dispatch

import unxt as u

import galax._custom_types as gt
import galax.potential as gp

#########################################################################
# Lagrange points

T = TypeVar("T")


class L1L2LagrangePoints(NamedTuple, Generic[T]):
    l1: T
    l2: T


@dispatch.abstract
def lagrange_points(
    potential: gp.AbstractPotential, x: Any, v: Any, /, mass: Any, t: Any
) -> L1L2LagrangePoints[Any]:
    """Compute the L1, L2 lagrange points of a cluster in a host potential.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.MilkyWayPotential()

    - With `jax.Array`:

    >>> x, v = jnp.asarray([8.0, 0.0, 0.0]), jnp.asarray([0.0, 220.0, 0.0])
    >>> mass, t = jnp.asarray(1e4), jnp.asarray(0.0)

    >>> lpts = gd.cluster.lagrange_points(pot, x, v, mass=mass, t=t)
    >>> lpts
    L1L2LagrangePoints(l1=Array([7.99960964, 0. , 0. ], dtype=float64),
                       l2=Array([8.00039036, 0. , 0. ], dtype=float64))

    - With `unxt.Quantity`:

    >>> x = u.Quantity([8.0, 0.0, 0.0], "kpc")
    >>> v = u.Quantity([0.0, 220.0, 0.0], "km/s")
    >>> mass = u.Quantity(1e4, "Msun")
    >>> t = u.Quantity(0.0, "Gyr")

    >>> lpts = gd.cluster.lagrange_points(pot, x, v, mass=mass, t=t)
    >>> lpts
    L1L2LagrangePoints(l1=Quantity(Array([7.97070926, 0. , 0. ], dtype=float64), unit='kpc'),
                       l2=Quantity(Array([8.02929074, 0. , 0. ], dtype=float64), unit='kpc'))

    - With `coordinax.vecs.AbstractVector`:

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)

    >>> lpts = gd.cluster.lagrange_points(pot, q, p, mass=mass, t=t)
    >>> lpts
    L1L2LagrangePoints(l1=Quantity(Array([7.97070926, 0. , 0. ], dtype=float64), unit='kpc'),
                       l2=Quantity(Array([8.02929074, 0. , 0. ], dtype=float64), unit='kpc'))

    - With `coordinax.Space`:

    >>> space = cx.Space(length=q, speed=p)
    >>> lpts = gd.cluster.lagrange_points(pot, space, mass=mass, t=t)
    >>> lpts
    L1L2LagrangePoints(l1=Quantity(Array([7.97070926, 0. , 0. ], dtype=float64), unit='kpc'),
                       l2=Quantity(Array([8.02929074, 0. , 0. ], dtype=float64), unit='kpc'))

    - With `coordinax.Coordinate`:

    >>> coord = cx.Coordinate(space, frame=gc.frames.simulation_frame)
    >>> lpts = gd.cluster.lagrange_points(pot, coord, mass=mass, t=t)
    >>> lpts
    L1L2LagrangePoints(l1=Quantity(Array([7.97070926, 0. , 0. ], dtype=float64), unit='kpc'),
                       l2=Quantity(Array([8.02929074, 0. , 0. ], dtype=float64), unit='kpc'))

    - With `coordinax.PhaseSpacePosition`:

    >>> w = gc.PhaseSpacePosition(q=q, p=p)
    >>> lpts = gd.cluster.lagrange_points(pot, w, mass=mass, t=t)
    >>> lpts
    L1L2LagrangePoints(l1=Quantity(Array([7.97070926, 0. , 0. ], dtype=float64), unit='kpc'),
                       l2=Quantity(Array([8.02929074, 0. , 0. ], dtype=float64), unit='kpc'))

    - With `coordinax.PhaseSpaceCoordinate`:

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> lpts = gd.cluster.lagrange_points(pot, w, mass=mass)
    >>> lpts
    L1L2LagrangePoints(l1=Quantity(Array([7.97070926, 0. , 0. ], dtype=float64), unit='kpc'),
                       l2=Quantity(Array([8.02929074, 0. , 0. ], dtype=float64), unit='kpc'))

    """  # noqa: E501
    raise NotImplementedError  # pragma: no cover


#########################################################################
# relaxation time


@dispatch.abstract
def relaxation_time(*args: Any, **kwargs: Any) -> u.AbstractQuantity:
    """Compute the cluster's relaxation time.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> M = u.Quantity(1e4, "Msun")
    >>> r_hm = u.Quantity(2, "pc")
    >>> m_avg = u.Quantity(0.5, "Msun")

    >>> gdc.relaxation_time(M, r_hm, m_avg=m_avg).uconvert("Myr")
    Quantity(Array(129.50777927, dtype=float64), unit='Myr')

    There are many different definitions of the relaxation time.
    By passing a flag object you can choose the one you want.
    Let's work through the built-in options:

    >>> flags = gdc.relax_time  # (not only flags)

    - Baumgardt (1998) (the default):

    >>> gdc.relaxation_time(flags.Baumgardt1998, M, r_hm, m_avg=m_avg).uconvert("Myr")
    Quantity(Array(129.50777927, dtype=float64), unit='Myr')

    - Spitzer and Hart (1971):

    >>> gdc.relaxation_time(flags.SpitzerHart1971, M, r_hm, m_avg=m_avg).uconvert("Myr")
    Quantity(Array(151.23177551, dtype=float64), unit='Myr')

    - Spitzer (1987) half-mass:

    >>> lnLambda = 10  # very approximate
    >>> gdc.relaxation_time(flags.Spitzer1987HalfMass, M, r_hm, m_avg=m_avg, lnLambda=lnLambda).uconvert("Myr")
    Quantity(Array(143.38045171, dtype=float64), unit='Myr')

    - Spitzer (1987) core:

    >>> Mcore, r_c = M / 5, r_hm / 5  # very approximate
    >>> gdc.relaxation_time(flags.Spitzer1987Core, Mcore, r_c, m_avg=m_avg, lnLambda=lnLambda).uconvert("Myr")
    Quantity(Array(11.47043614, dtype=float64), unit='Myr')

    Using multiple-dispatch, you can register your own relaxation time
    definition.

    """  # noqa: E501
    raise NotImplementedError  # pragma: no cover


#####################################################################
# tidal radius


@dispatch.abstract
def tidal_radius(*args: Any, **kwargs: Any) -> gt.BBtQuSz0:
    """Compute the tidal radius of a cluster in the potential.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics.cluster as gdc

    >>> pot = gp.NFWPotential(m=1e12, r_s=20.0, units="galactic")

    >>> x = u.Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> v = u.Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc/Myr")
    >>> t = u.Quantity(0, "Myr")
    >>> mass = u.Quantity(1e4, "Msun")

    >>> gdc.tidal_radius(pot, x, v, mass=mass, t=t)
    Quantity(Array(0.06362008, dtype=float64), unit='kpc')

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)
    >>> gdc.tidal_radius(pot, q, p, mass=mass, t=t)
    Quantity(Array(0.06362008, dtype=float64), unit='kpc')

    >>> space = cx.Space(length=q, speed=p)
    >>> gdc.tidal_radius(pot, space, mass=mass, t=t)
    Quantity(Array(0.06362008, dtype=float64), unit='kpc')

    >>> coord = cx.Coordinate(space, frame=gc.frames.simulation_frame)
    >>> gdc.tidal_radius(pot, coord, mass=mass, t=t)
    Quantity(Array(0.06362008, dtype=float64), unit='kpc')

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> gdc.tidal_radius(pot, w, mass=mass)
    Quantity(Array(0.06362008, dtype=float64), unit='kpc')

    Now with different methods:

    - King (1962) (the default):

    >>> gdc.tidal_radius(gdc.radius.King1962, pot, x, v, mass=mass, t=t)
    Quantity(Array(0.06362008, dtype=float64), unit='kpc')

    - von Hoerner (1957):

    >>> gdc.tidal_radius(gdc.radius.Hoerner1957, pot, x, mass=mass, t=t).uconvert("pc")
    Quantity(Array(36.94695299, dtype=float64), unit='pc')

    - King (1962) with a point mass:

    >>> rperi = jnp.linalg.vector_norm(x, axis=-1)
    >>> gdc.tidal_radius(gdc.radius.King1962PointMass, pot,
    ...                  rperi=rperi, mass=mass, t=t, e=0.5).uconvert("pc")
    Quantity(Array(30.65956192, dtype=float64), unit='pc')

    >>> gdc.tidal_radius(gdc.radius.King1962PointMass, pot,
    ...                  rperi=q, mass=mass, t=t, e=0.5).uconvert("pc")
    Quantity(Array(30.65956192, dtype=float64), unit='pc')

    """
    raise NotImplementedError  # pragma: no cover
