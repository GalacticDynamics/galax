"""Cluster functions."""

__all__ = [
    "lagrange_points",
    "tidal_radius",
]

from functools import partial

import jax
from jaxtyping import Float

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.typing as gt
from galax.dynamics._src.api import omega

# ===================================================================


@partial(jax.jit, inline=True)
def tidal_radius(
    pot: gp.AbstractPotential,
    x: gt.LengthBtSz3,
    v: gt.SpeedBtSz3,
    /,
    prog_mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
) -> Float[u.Quantity["length"], "*batch"]:
    """Compute the tidal radius of a cluster in the potential.

    Parameters
    ----------
    pot : `galax.potential.AbstractPotential`
        The gravitational potential of the host.
    x: Quantity[float, (3,), "length"]
        3d position (x, y, z).
    v: Quantity[float, (3,), "speed"]
        3d velocity (v_x, v_y, v_z).
    prog_mass : Quantity[float, (), "mass"]
        Cluster mass.
    t: Quantity[float, (), "time"]
        Time.

    Returns
    -------
    Quantity[float, (), "length"]
        Tidal radius of the cluster.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential(m=1e12, r_s=20.0, units="galactic")

    >>> x = u.Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> v = u.Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc/Myr")
    >>> prog_mass = u.Quantity(1e4, "Msun")

    >>> tidal_radius(pot, x, v, prog_mass=prog_mass, t=u.Quantity(0, "Myr"))
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')
    """
    d2phi_dr2 = pot.d2potential_dr2(x, t)
    return jnp.cbrt(pot.constants["G"] * prog_mass / (omega(x, v) ** 2 - d2phi_dr2))


# ===================================================================


@partial(jax.jit, inline=True)
def lagrange_points(
    potential: gp.AbstractPotential,
    x: gt.LengthSz3,
    v: gt.SpeedSz3,
    prog_mass: gt.MassSz0,
    t: gt.TimeSz0,
) -> tuple[gt.LengthSz3, gt.LengthSz3]:
    """Compute the lagrange points of a cluster in a host potential.

    Parameters
    ----------
    potential : `galax.potential.AbstractPotential`
        The gravitational potential of the host.
    x: Quantity[float, (3,), "length"]
        Cartesian 3D position ($x$, $y$, $z$)
    v: Quantity[float, (3,), "speed"]
        Cartesian 3D velocity ($v_x$, $v_y$, $v_z$)
    prog_mass: Quantity[float, (), "mass"]
        Cluster mass.
    t: Quantity[float, (), "time"]
        Time.

    Returns
    -------
    L_1, L_2: Quantity[float, (3,), "length"]
        The lagrange points L_1 and L_2.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MilkyWayPotential()
    >>> x = u.Quantity([8.0, 0.0, 0.0], "kpc")
    >>> v = u.Quantity([0.0, 220.0, 0.0], "km/s")
    >>> prog_mass = u.Quantity(1e4, "Msun")
    >>> t = u.Quantity(0.0, "Gyr")

    >>> L1, L2 = lagrange_points(pot, x, v, prog_mass, t)
    >>> L1
    Quantity['length'](Array([7.97070926, 0.        , 0.        ], dtype=float64), unit='kpc')
    >>> L2
    Quantity['length'](Array([8.02929074, 0.        , 0.        ], dtype=float64), unit='kpc')
    """  # noqa: E501
    r_hat = cx.vecs.normalize_vector(x)
    r_t = tidal_radius(potential, x, v, prog_mass, t)
    L_1 = x - r_hat * r_t  # close
    L_2 = x + r_hat * r_t  # far
    return L_1, L_2
