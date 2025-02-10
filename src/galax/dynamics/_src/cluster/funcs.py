"""Cluster functions."""

__all__ = [
    "lagrange_points",
    "tidal_radius",
]

from functools import partial
from typing import Any, NamedTuple

import jax
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from galax.dynamics._src.api import omega

# ===================================================================


@dispatch.abstract
def tidal_radius(*args: Any, **kwargs: Any) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential.

    Examples
    --------
    >>> import unxt as u
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.NFWPotential(m=1e12, r_s=20.0, units="galactic")

    >>> x = u.Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> v = u.Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc/Myr")
    >>> t = u.Quantity(0, "Myr")
    >>> mass = u.Quantity(1e4, "Msun")

    >>> gd.cluster.tidal_radius(pot, x, v, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)
    >>> gd.cluster.tidal_radius(pot, q, p, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> space = cx.Space(length=q, speed=p)
    >>> gd.cluster.tidal_radius(pot, space, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> coord = cx.Coordinate(space, frame=gc.frames.SimulationFrame())
    >>> gd.cluster.tidal_radius(pot, coord, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> gd.cluster.tidal_radius(pot, w, mass=mass)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    """
    raise NotImplementedError  # pragma: no cover


@dispatch
@partial(jax.jit)
def tidal_radius(
    pot: gp.AbstractPotential,
    x: gt.BBtRealQuSz3 | cx.vecs.AbstractPos3D,
    v: gt.BBtRealQuSz3 | cx.vecs.AbstractVel3D,
    /,
    *,
    mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
) -> gt.BBtRealQuSz0:
    """Compute from `unxt.Quantity` or `coordinax.vecs.AbstractVector`s."""
    d2phi_dr2 = pot.d2potential_dr2(x, t)
    return jnp.cbrt(pot.constants["G"] * mass / (omega(x, v) ** 2 - d2phi_dr2))


@dispatch
def tidal_radius(
    pot: gp.AbstractPotential,
    space: cx.Space,
    /,
    *,
    mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    q, p = space["length"], space["speed"]
    return tidal_radius(pot, q, p, mass=mass, t=t)


@dispatch
def tidal_radius(
    pot: gp.AbstractPotential,
    coord: cx.frames.AbstractCoordinate,
    /,
    *,
    mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius(pot, coord.data, mass=mass, t=t)


@dispatch
def tidal_radius(
    pot: gp.AbstractPotential, w: gc.PhaseSpaceCoordinate, /, *, mass: gt.MassBBtSz0
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius(pot, w.q, w.p, mass=mass, t=w.t)


# ===================================================================


class L1L2LagrangePoints(NamedTuple):
    l1: gt.BBtRealQuSz3
    l2: gt.BBtRealQuSz3


# TODO: vec, space, coordinate, PSP I/O
@partial(jax.jit)
def lagrange_points(
    potential: gp.AbstractPotential,
    x: gt.LengthSz3,
    v: gt.SpeedSz3,
    /,
    mass: gt.MassSz0,
    t: gt.TimeSz0,
) -> L1L2LagrangePoints:
    """Compute the lagrange points of a cluster in a host potential.

    Parameters
    ----------
    potential : `galax.potential.AbstractPotential`
        The gravitational potential of the host.
    x: Quantity[float, (3,), "length"]
        Cartesian 3D position ($x$, $y$, $z$)
    v: Quantity[float, (3,), "speed"]
        Cartesian 3D velocity ($v_x$, $v_y$, $v_z$)
    mass: Quantity[float, (), "mass"]
        Cluster mass.
    t: Quantity[float, (), "time"]
        Time.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MilkyWayPotential()
    >>> x = u.Quantity([8.0, 0.0, 0.0], "kpc")
    >>> v = u.Quantity([0.0, 220.0, 0.0], "km/s")
    >>> mass = u.Quantity(1e4, "Msun")
    >>> t = u.Quantity(0.0, "Gyr")

    >>> lpts = lagrange_points(pot, x, v, mass=mass, t=t)
    >>> lpts.l1
    Quantity['length'](Array([7.97070926, 0. , 0. ], dtype=float64), unit='kpc')
    >>> lpts.l2
    Quantity['length'](Array([8.02929074, 0. , 0. ], dtype=float64), unit='kpc')

    """
    r_t = tidal_radius(potential, x, v, mass=mass, t=t)
    r_hat = cx.vecs.normalize_vector(x)
    l1 = x - r_hat * r_t  # close
    l2 = x + r_hat * r_t  # far
    return L1L2LagrangePoints(l1=l1, l2=l2)
