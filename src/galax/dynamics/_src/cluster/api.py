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

from typing import Any, NamedTuple

from plum import dispatch

import unxt as u

import galax.potential as gp
import galax.typing as gt

#########################################################################
# Lagrange points


class L1L2LagrangePoints(NamedTuple):
    l1: gt.BBtRealQuSz3
    l2: gt.BBtRealQuSz3


@dispatch.abstract
def lagrange_points(
    potential: gp.AbstractPotential,
    x: gt.LengthSz3,
    v: gt.SpeedSz3,
    /,
    mass: gt.MassSz0,
    t: gt.TimeSz0,
) -> L1L2LagrangePoints:
    """Compute the lagrange points of a cluster in a host potential.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.MilkyWayPotential()
    >>> x = u.Quantity([8.0, 0.0, 0.0], "kpc")
    >>> v = u.Quantity([0.0, 220.0, 0.0], "km/s")
    >>> mass = u.Quantity(1e4, "Msun")
    >>> t = u.Quantity(0.0, "Gyr")

    >>> lpts = gd.cluster.lagrange_points(pot, x, v, mass=mass, t=t)
    >>> lpts.l1
    Quantity['length'](Array([7.97070926, 0. , 0. ], dtype=float64), unit='kpc')
    >>> lpts.l2
    Quantity['length'](Array([8.02929074, 0. , 0. ], dtype=float64), unit='kpc')

    """
    raise NotImplementedError  # pragma: no cover


#########################################################################
# relaxation time


@dispatch.abstract
def relaxation_time(*args: Any, **kwargs: Any) -> u.AbstractQuantity:
    """Compute the cluster's relaxation time.

    Baumgardt 1998 Equation 1.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> M = u.Quantity(1e4, "Msun")
    >>> r_hm = u.Quantity(2, "pc")
    >>> nstars = 2e4
    >>> G = u.Quantity(4.30091e-3, "pc km2 / (s2 Msun)")

    >>> gdc.relaxation_time(M, r_hm, nstars, G=G).uconvert("Myr")
    Quantity['time'](Array(129.50788873, dtype=float64, ...), unit='Myr')

    Now with different methods:

    - Baumgardt (1998) (the default):

    >>> gdc.relaxation_time(gdc.relax_time.Baumgardt1998, M, r_hm, nstars, G=G).uconvert("Myr")
    Quantity['time'](Array(129.50788873, dtype=float64, ...), unit='Myr')

    - Spitzer (1987) half-mass:

    >>> lnLambda = 10  # very approximate
    >>> gdc.relaxation_time(gdc.relax_time.Spitzer1987HalfMass, M, r_hm, nstars, lnLambda=lnLambda, G=G).uconvert("Myr")
    Quantity['time'](Array(143.38057289, dtype=float64, weak_type=True), unit='Myr')

    - Spitzer (1987) core:

    >>> Mcore, r_c = M / 5, r_hm / 5  # very approximate
    >>> gdc.relaxation_time(gdc.relax_time.Spitzer1987Core, Mcore, r_c, nstars, lnLambda=lnLambda, G=G).uconvert("Myr")
    Quantity['time'](Array(57.35222916, dtype=float64, weak_type=True), unit='Myr')

    """  # noqa: E501
    raise NotImplementedError  # pragma: no cover


#####################################################################
# tidal radius


@dispatch.abstract
def tidal_radius(*args: Any, **kwargs: Any) -> gt.BBtRealQuSz0:
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
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)
    >>> gdc.tidal_radius(pot, q, p, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> space = cx.Space(length=q, speed=p)
    >>> gdc.tidal_radius(pot, space, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> coord = cx.Coordinate(space, frame=gc.frames.SimulationFrame())
    >>> gdc.tidal_radius(pot, coord, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> gdc.tidal_radius(pot, w, mass=mass)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    Now with different methods:

    - King (1962) (the default):

    >>> gdc.tidal_radius(gdc.radius.King1962, pot, x, v, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    - von Hoerner (1957):

    >>> gdc.tidal_radius(gdc.radius.Hoerner1957, pot, x, mass=mass, t=t).uconvert("pc")
    Quantity['length'](Array(36.94695299, dtype=float64), unit='pc')

    - King (1962) with a point mass:

    >>> rperi = jnp.linalg.vector_norm(x, axis=-1)
    >>> gdc.tidal_radius(gdc.radius.King1962PointMass, pot,
    ...                  rperi=rperi, mass=mass, t=t, e=0.5).uconvert("pc")
    Quantity['length'](Array(30.65956192, dtype=float64), unit='pc')

    >>> gdc.tidal_radius(gdc.radius.King1962PointMass, pot,
    ...                  rperi=q, mass=mass, t=t, e=0.5).uconvert("pc")
    Quantity['length'](Array(30.65956192, dtype=float64), unit='pc')

    """
    raise NotImplementedError  # pragma: no cover
