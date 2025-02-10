"""Cluster functions."""

__all__ = [
    # Lagrange points
    "lagrange_points",
    "L1L2LagrangePoints",
]

from typing import NamedTuple

from plum import dispatch

import galax.potential as gp
import galax.typing as gt


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
