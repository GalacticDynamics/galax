"""Cluster functions."""

__all__: list[str] = []

from functools import partial

import equinox as eqx
import jax
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.coordinates as gc
import galax.potential as gp
from .api import L1L2LagrangePoints
from .radius import tidal_radius_king1962


@dispatch
@partial(jax.jit)
def lagrange_points(
    potential: gp.AbstractPotential, x: gt.Sz3, v: gt.Sz3, /, *, mass: gt.Sz0, t: gt.Sz0
) -> L1L2LagrangePoints:  # type: ignore[type-arg]  # TODO: when beartype permits
    """Compute the lagrange points of a cluster in a host potential."""
    r_t = tidal_radius_king1962(potential, x, v, mass=mass, t=t)
    r_hat = cx.vecs.normalize_vector(x)
    l1 = x - r_hat * r_t  # close
    l2 = x + r_hat * r_t  # far
    return L1L2LagrangePoints(l1=l1, l2=l2)


@dispatch
@partial(jax.jit)
def lagrange_points(
    potential: gp.AbstractPotential,
    x: gt.QuSz3 | cx.vecs.AbstractPos3D,
    v: gt.QuSz3 | cx.vecs.AbstractVel3D,
    /,
    *,
    mass: gt.QuSz0,
    t: gt.QuSz0,
) -> L1L2LagrangePoints:  # type: ignore[type-arg]  # TODO: when beartype permits
    """Compute the lagrange points of a cluster in a host potential."""
    x = convert(x, u.Quantity)
    v = convert(v, u.Quantity)
    r_t = tidal_radius_king1962(potential, x, v, mass=mass, t=t)
    r_hat = cx.vecs.normalize_vector(x)
    l1 = x - r_hat * r_t  # close
    l2 = x + r_hat * r_t  # far
    return L1L2LagrangePoints(l1=l1, l2=l2)


@dispatch
def lagrange_points(
    pot: gp.AbstractPotential,
    space: cx.Space,
    /,
    *,
    mass: gt.QuSz0,
    t: gt.QuSz0,
) -> L1L2LagrangePoints:  # type: ignore[type-arg]  # TODO: when beartype permits
    """Compute the lagrange points of a cluster in a host potential."""
    return lagrange_points(pot, space["length"], space["speed"], mass=mass, t=t)


@dispatch
def lagrange_points(
    pot: gp.AbstractPotential,
    coord: cx.frames.AbstractCoordinate,
    /,
    *,
    mass: gt.QuSz0,
    t: gt.QuSz0,
) -> L1L2LagrangePoints:  # type: ignore[type-arg]  # TODO: when beartype permits
    """Compute the lagrange points of a cluster in a host potential."""
    return lagrange_points(pot, coord.data, mass=mass, t=t)


@dispatch
def lagrange_points(
    pot: gp.AbstractPotential,
    wt: gc.AbstractPhaseSpaceCoordinate,
    /,
    *,
    mass: gt.QuSz0,
    t: gt.QuSz0 | None = None,
) -> L1L2LagrangePoints:  # type: ignore[type-arg]  # TODO: when beartype permits
    """Compute the lagrange points of a cluster in a host potential."""
    t = eqx.error_if(
        wt.t,
        t is not None and jnp.logical_not(jnp.array_equal(wt.t, t)),
        "t must be None or equal to the time of the phase space coordinate.",
    )
    return lagrange_points(pot, wt.q, wt.p, mass=mass, t=t.squeeze())


@dispatch
def lagrange_points(
    pot: gp.AbstractPotential,
    w: gc.PhaseSpacePosition,
    /,
    *,
    mass: gt.QuSz0,
    t: gt.QuSz0,
) -> L1L2LagrangePoints:  # type: ignore[type-arg]  # TODO: when beartype permits
    """Compute the lagrange points of a cluster in a host potential."""
    return lagrange_points(pot, w.q, w.p, mass=mass, t=t)
