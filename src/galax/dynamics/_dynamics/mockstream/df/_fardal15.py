"""galax: Galactic Dynamix in Jax."""

__all__ = ["FardalStreamDF"]


from functools import partial
from typing import final

import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray, Shaped
from plum import convert

import quaxed.array_api as xp
import quaxed.numpy as qnp
from unxt import Quantity
from unxt.experimental import grad

import galax.potential as gp
import galax.typing as gt
from ._base import AbstractStreamDF

# ============================================================
# Constants

kr_bar = 2.0
kvphi_bar = 0.3

kz_bar = 0.0
kvz_bar = 0.0

sigma_kr = 0.5  # TODO: use actual Fardal values
sigma_kvphi = 0.5  # TODO: use actual Fardal values
sigma_kz = 0.5
sigma_kvz = 0.5

# ============================================================


@final
class FardalStreamDF(AbstractStreamDF):
    """Fardal Stream Distribution Function.

    A class for representing the Fardal+2015 distribution function for
    generating stellar streams based on Fardal et al. 2015
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract
    """

    @partial(jax.jit, inline=True)
    def _sample(
        self,
        key: PRNGKeyArray,
        potential: gp.AbstractPotentialBase,
        x: gt.LengthBatchableVec3,
        v: gt.SpeedBatchableVec3,
        prog_mass: gt.BatchableFloatQScalar,
        t: gt.BatchableFloatQScalar,
    ) -> tuple[
        gt.LengthBatchVec3, gt.SpeedBatchVec3, gt.LengthBatchVec3, gt.SpeedBatchVec3
    ]:
        """Generate stream particle initial conditions."""
        # Random number generation
        key1, key2, key3, key4 = jr.split(key, 4)

        omega_val = orbital_angular_velocity_mag(x, v)[..., None]

        # r-hat
        r = xp.linalg.vector_norm(x, axis=-1, keepdims=True)
        r_hat = x / r

        r_tidal = tidal_radius(potential, x, v, prog_mass, t)[..., None]
        v_circ = omega_val * r_tidal  # relative velocity

        # z-hat
        L_vec = qnp.cross(x, v)
        z_hat = L_vec / xp.linalg.vector_norm(L_vec, axis=-1, keepdims=True)

        # phi-hat
        phi_vec = v - xp.sum(v * r_hat, axis=-1, keepdims=True) * r_hat
        phi_hat = phi_vec / xp.linalg.vector_norm(phi_vec, axis=-1, keepdims=True)

        # k vals
        shape = r_tidal.shape
        kr_samp = kr_bar + jr.normal(key1, shape) * sigma_kr
        kvphi_samp = kr_samp * (kvphi_bar + jr.normal(key2, shape) * sigma_kvphi)
        kz_samp = kz_bar + jr.normal(key3, shape) * sigma_kz
        kvz_samp = kvz_bar + jr.normal(key4, shape) * sigma_kvz

        # Trailing arm
        x_trail = x + r_tidal * (kr_samp * r_hat + kz_samp * z_hat)
        v_trail = v + v_circ * (kvphi_samp * phi_hat + kvz_samp * z_hat)

        # Leading arm
        x_lead = x - r_tidal * (kr_samp * r_hat - kz_samp * z_hat)
        v_lead = v - v_circ * (kvphi_samp * phi_hat - kvz_samp * z_hat)

        return x_lead, v_lead, x_trail, v_trail


#####################################################################
# TODO: move this to a more general location.


def r_hat(x: gt.LengthBatchVec3, /) -> Shaped[Quantity[""], "*batch 3"]:
    """Compute the unit vector in the radial direction.

    Parameters
    ----------
    x: Quantity[float, (*batch, 3), "length"]
        3d position (x, y, z) in [kpc]

    Returns
    -------
    Quantity[float, (*batch, 3), ""]
        Unit vector in the radial direction.
    """
    return x / xp.linalg.vector_norm(x, axis=-1, keepdims=True)


@partial(jax.jit, inline=True)
def dphidr(
    potential: gp.AbstractPotentialBase,
    x: gt.LengthBatchVec3,
    t: Shaped[Quantity["time"], ""],
) -> Shaped[Quantity["acceleration"], "*batch"]:
    """Compute the derivative of the potential at a position x.

    Parameters
    ----------
    potential : `galax.potential.AbstractPotentialBase`
        The gravitational potential.
    x: Quantity[float, (3,), 'length']
        3d position (x, y, z)
    t: Quantity[float, (), 'time']
        Time in [Myr]

    Returns
    -------
    Quantity[float, (3,), 'acceleration']:
        Derivative of potential
    """
    gradient = convert(potential.gradient(x, t), Quantity)
    return xp.sum(gradient * r_hat(x), axis=-1)


@partial(jax.jit)
@partial(qnp.vectorize, excluded=(0,), signature="(3),()->()")
def d2phidr2(
    potential: gp.AbstractPotentialBase, x: gt.LengthVec3, t: gt.TimeScalar, /
) -> Shaped[Quantity["1/s^2"], ""]:
    """Compute the second derivative of the potential.

    At a position x (in the simulation frame).

    Parameters
    ----------
    potential : `galax.potential.AbstractPotentialBase`
        The gravitational potential.
    x: Quantity[Any, (3,), 'length']
        3d position (x, y, z) in [kpc]
    t: Quantity[Any, (), 'time']
        Time in [Myr]

    Returns
    -------
    Array:
        Second derivative of force (per unit mass) in [1/Myr^2]

    Examples
    --------
    >>> from unxt import Quantity
    >>> from galax.potential import NFWPotential
    >>> pot = NFWPotential(m=1e12, r_s=20.0, units="galactic")
    >>> q = Quantity(xp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> d2phidr2(pot, q, Quantity(0.0, "Myr"))
    Quantity['1'](Array(-0.0001747, dtype=float64), unit='1 / Myr2')
    """
    rhat = r_hat(x)
    # TODO: this isn't vectorized
    d2phidr2_func = grad(dphidr, argnums=1, units=(None, x.unit, t.unit))
    return xp.sum(d2phidr2_func(potential, x, t) * rhat)


@partial(jax.jit)
def orbital_angular_velocity(
    x: gt.LengthVec3, v: gt.SpeedVec3, /
) -> Shaped[Quantity["frequency"], ""]:
    """Compute the orbital angular velocity about the origin.

    Arguments:
    ---------
    x: Array[Any, (3,)]
        3d position (x, y, z) in [length]
    v: Array[Any, (3,)]
        3d velocity (v_x, v_y, v_z) in [length/time]

    Returns
    -------
    Array
        Angular velocity in [rad/time]

    Examples
    --------
    >>> x = Quantity(xp.asarray([8.0, 0.0, 0.0]), "m")
    >>> v = Quantity(xp.asarray([8.0, 0.0, 0.0]), "m/s")
    >>> orbital_angular_velocity(x, v)
    Quantity['frequency'](Array([0., 0., 0.], dtype=float64), unit='1 / s')
    """
    r = xp.linalg.vector_norm(x, axis=-1, keepdims=True)
    return qnp.cross(x, v) / r**2


@partial(jax.jit)
def orbital_angular_velocity_mag(
    x: gt.LengthVec3, v: gt.SpeedVec3, /
) -> Shaped[Quantity, "m^2/s"]:
    """Compute the magnitude of the angular momentum in the simulation frame.

    Arguments:
    ---------
    x: Quantity[float, (3,), "length"]
        3d position (x, y, z) in [kpc]
    v: Quantity[float, (3,), "speed"]
        3d velocity (v_x, v_y, v_z) in [kpc/Myr]

    Returns
    -------
    Quantity[float, (3,), "length^2/time"]
        Magnitude of specific angular momentum.

    Examples
    --------
    >>> x = Quantity(xp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> v = Quantity(xp.asarray([8.0, 0.0, 0.0]), "kpc/Myr")
    >>> orbital_angular_velocity_mag(x, v)
    Quantity['frequency'](Array(0., dtype=float64), unit='1 / Myr')
    """
    return xp.linalg.vector_norm(orbital_angular_velocity(x, v), axis=-1)


@partial(jax.jit)
def tidal_radius(
    potential: gp.AbstractPotentialBase,
    x: gt.LengthVec3,
    v: gt.SpeedVec3,
    /,
    prog_mass: gt.MassScalar,
    t: gt.TimeScalar,
) -> gt.LengthScalar:
    """Compute the tidal radius of a cluster in the potential.

    Parameters
    ----------
    potential : `galax.potential.AbstractPotentialBase`
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
    >>> from galax.potential import NFWPotential

    >>> pot = NFWPotential(m=1e12, r_s=20.0, units="galactic")

    >>> x = Quantity(xp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> v = Quantity(xp.asarray([8.0, 0.0, 0.0]), "kpc/Myr")
    >>> prog_mass = Quantity(1e4, "Msun")

    >>> tidal_radius(pot, x, v, prog_mass=prog_mass, t=Quantity(0, "Myr"))
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')
    """
    return qnp.cbrt(
        potential.constants["G"]
        * prog_mass
        / (orbital_angular_velocity_mag(x, v) ** 2 - d2phidr2(potential, x, t))
    )


@partial(jax.jit)
def lagrange_points(
    potential: gp.AbstractPotentialBase,
    x: gt.LengthVec3,
    v: gt.SpeedVec3,
    prog_mass: gt.MassScalar,
    t: gt.TimeScalar,
) -> tuple[gt.LengthVec3, gt.LengthVec3]:
    """Compute the lagrange points of a cluster in a host potential.

    Parameters
    ----------
    potential : `galax.potential.AbstractPotentialBase`
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
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.MilkyWayPotential()
    >>> x = Quantity(xp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> v = Quantity(xp.asarray([0.0, 220.0, 0.0]), "km/s")
    >>> prog_mass = Quantity(1e4, "Msun")
    >>> t = Quantity(0.0, "Gyr")

    >>> L1, L2 = lagrange_points(pot, x, v, prog_mass, t)
    >>> L1
    Quantity['length'](Array([7.97070926, 0.        , 0.        ], dtype=float64), unit='kpc')
    >>> L2
    Quantity['length'](Array([8.02929074, 0.        , 0.        ], dtype=float64), unit='kpc')
    """  # noqa: E501
    r_hat = x / xp.linalg.vector_norm(x, axis=-1, keepdims=True)
    r_t = tidal_radius(potential, x, v, prog_mass, t)
    L_1 = x - r_hat * r_t  # close
    L_2 = x + r_hat * r_t  # far
    return L_1, L_2
