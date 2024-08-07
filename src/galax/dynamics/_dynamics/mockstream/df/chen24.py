"""galax: Galactic Dynamix in Jax."""

__all__ = ["ChenStreamDF"]


import warnings
from functools import partial
from typing import final

import jax
import jax.random as jr
from jaxtyping import Float, PRNGKeyArray, Shaped

import quaxed.array_api as xp
import quaxed.numpy as qnp
from unxt import Quantity

import galax.potential as gp
import galax.typing as gt
from .base import AbstractStreamDF
from galax.potential._potential.funcs import d2potential_dr2

# ============================================================
# Constants

mean = qnp.array([1.6, -30, 0, 1, 20, 0])

cov = qnp.array(
    [
        [0.1225, 0, 0, 0, -4.9, 0],
        [0, 529, 0, 0, 0, 0],
        [0, 0, 144, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-4.9, 0, 0, 0, 400, 0],
        [0, 0, 0, 0, 0, 484],
    ]
)

# ============================================================


@final
class ChenStreamDF(AbstractStreamDF):
    """Chen Stream Distribution Function.

    A class for representing the Chen+2024 distribution function for
    generating stellar streams based on Chen et al. 2024
    https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract
    """

    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            'Currently only the "no progenitor" version '
            "of the Chen+24 model is supported!",
            RuntimeWarning,
            stacklevel=1,
        )

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

        # x_new-hat
        r = xp.linalg.vector_norm(x, axis=-1, keepdims=True)
        x_new_hat = x / r

        # z_new-hat
        L_vec = qnp.cross(x, v)
        z_new_hat = L_vec / xp.linalg.vector_norm(L_vec, axis=-1, keepdims=True)

        # y_new-hat
        phi_vec = v - xp.sum(v * x_new_hat, axis=-1, keepdims=True) * x_new_hat
        y_new_hat = phi_vec / xp.linalg.vector_norm(phi_vec, axis=-1, keepdims=True)

        r_tidal = tidal_radius(potential, x, v, prog_mass, t)

        # Bill Chen: method="cholesky" doesn't work here!
        posvel = jr.multivariate_normal(
            key, mean, cov, shape=r_tidal.shape, method="svd"
        )

        Dr = posvel[:, 0] * r_tidal

        v_esc = qnp.sqrt(2 * potential.constants["G"] * prog_mass / Dr)
        Dv = posvel[:, 3] * v_esc

        # convert degrees to radians
        phi = posvel[:, 1] * 0.017453292519943295
        theta = posvel[:, 2] * 0.017453292519943295
        alpha = posvel[:, 4] * 0.017453292519943295
        beta = posvel[:, 5] * 0.017453292519943295

        # Trailing arm
        x_trail = (
            x
            + (Dr * qnp.cos(theta) * qnp.cos(phi))[:, qnp.newaxis] * x_new_hat
            + (Dr * qnp.cos(theta) * qnp.sin(phi))[:, qnp.newaxis] * y_new_hat
            + (Dr * qnp.sin(theta))[:, qnp.newaxis] * z_new_hat
        )
        v_trail = (
            v
            + (Dv * qnp.cos(beta) * qnp.cos(alpha))[:, qnp.newaxis] * x_new_hat
            + (Dv * qnp.cos(beta) * qnp.sin(alpha))[:, qnp.newaxis] * y_new_hat
            + (Dv * qnp.sin(beta))[:, qnp.newaxis] * z_new_hat
        )

        # Leading arm
        x_lead = (
            x
            - (Dr * qnp.cos(theta) * qnp.cos(phi))[:, qnp.newaxis] * x_new_hat
            - (Dr * qnp.cos(theta) * qnp.sin(phi))[:, qnp.newaxis] * y_new_hat
            + (Dr * qnp.sin(theta))[:, qnp.newaxis] * z_new_hat
        )
        v_lead = (
            v
            - (Dv * qnp.cos(beta) * qnp.cos(alpha))[:, qnp.newaxis] * x_new_hat
            - (Dv * qnp.cos(beta) * qnp.sin(alpha))[:, qnp.newaxis] * y_new_hat
            + (Dv * qnp.sin(beta))[:, qnp.newaxis] * z_new_hat
        )

        return x_lead, v_lead, x_trail, v_trail


#####################################################################
# TODO: below is copied from fardal15.py
#       Change accordingly when moved to a more general location.

#####################################################################
# TODO: move this to a more general location.


@partial(jax.jit, inline=True)
def orbital_angular_velocity(
    x: gt.LengthBatchVec3, v: gt.SpeedBatchVec3, /
) -> Shaped[Quantity["frequency"], "*batch 3"]:  # TODO: rad/s
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


@partial(jax.jit, inline=True)
def orbital_angular_velocity_mag(
    x: gt.LengthBatchVec3, v: gt.SpeedBatchVec3, /
) -> Shaped[Quantity["frequency"], "*batch"]:  # TODO: rad/s
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


@partial(jax.jit, inline=True)
def tidal_radius(
    potential: gp.AbstractPotentialBase,
    x: gt.LengthBatchVec3,
    v: gt.SpeedBatchVec3,
    /,
    prog_mass: gt.MassBatchableScalar,
    t: gt.TimeBatchableScalar,
) -> Float[Quantity["length"], "*batch"]:
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
    omega = orbital_angular_velocity_mag(x, v)
    d2phi_dr2 = d2potential_dr2(potential, x, t)
    return qnp.cbrt(potential.constants["G"] * prog_mass / (omega**2 - d2phi_dr2))


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
