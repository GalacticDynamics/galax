"""galax: Galactic Dynamix in Jax."""


__all__ = ["FardalStreamDF"]


import jax.experimental.array_api as xp
import jax.numpy as jnp
from jax import grad, random

from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import (
    FloatOrIntScalarLike,
    FloatScalar,
    IntLike,
    Vec3,
    Vec6,
)
from galax.utils import partial_jit

from .base import AbstractStreamDF


class FardalStreamDF(AbstractStreamDF):
    """Fardal Stream Distribution Function.

    A class for representing the Fardal+2015 distribution function for
    generating stellar streams based on Fardal et al. 2015
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract
    """

    @partial_jit(static_argnums=(0,), static_argnames=("seed_num",))
    def _sample(
        self,
        potential: AbstractPotentialBase,
        qp: Vec6,
        prog_mass: FloatScalar,
        t: FloatScalar,
        *,
        i: IntLike,
        seed_num: int,
    ) -> tuple[Vec3, Vec3, Vec3, Vec3]:
        """Generate stream particle initial conditions."""
        # Random number generation
        # TODO: change random key handling... need to do all of the sampling up front...
        key_master = random.PRNGKey(seed_num)
        random_ints = random.randint(key=key_master, shape=(4,), minval=0, maxval=1000)
        keya = random.PRNGKey(i * random_ints[0])
        keyb = random.PRNGKey(i * random_ints[1])
        keyc = random.PRNGKey(i * random_ints[2])
        keyd = random.PRNGKey(i * random_ints[3])

        # ---------------------------

        x, v = qp[0:3], qp[3:6]

        omega_val = orbital_angular_velocity_mag(x, v)

        r = xp.linalg.vector_norm(x)
        r_hat = x / r
        r_tidal = tidal_radius(potential, x, v, prog_mass, t)
        rel_v = omega_val * r_tidal  # relative velocity

        # circlar_velocity
        v_circ = rel_v

        L_vec = jnp.cross(x, v)
        z_hat = L_vec / xp.linalg.vector_norm(L_vec)

        phi_vec = v - xp.sum(v * r_hat) * r_hat
        phi_hat = phi_vec / xp.linalg.vector_norm(phi_vec)

        kr_bar = 2.0
        kvphi_bar = 0.3

        kz_bar = 0.0
        kvz_bar = 0.0

        sigma_kr = 0.5
        sigma_kvphi = 0.5
        sigma_kz = 0.5
        sigma_kvz = 0.5

        kr_samp = kr_bar + random.normal(keya, shape=(1,)) * sigma_kr
        kvphi_samp = kr_samp * (
            kvphi_bar + random.normal(keyb, shape=(1,)) * sigma_kvphi
        )
        kz_samp = kz_bar + random.normal(keyc, shape=(1,)) * sigma_kz
        kvz_samp = kvz_bar + random.normal(keyd, shape=(1,)) * sigma_kvz

        # Trailing arm
        x_trail = (
            x + (kr_samp * r_hat * (r_tidal)) + (z_hat * kz_samp * (r_tidal / 1.0))
        )
        v_trail = (
            v
            + (0.0 + kvphi_samp * v_circ * (1.0)) * phi_hat
            + (kvz_samp * v_circ * (1.0)) * z_hat
        )

        # Leading arm
        x_lead = (
            x + (kr_samp * r_hat * (-r_tidal)) + (z_hat * kz_samp * (-r_tidal / 1.0))
        )
        v_lead = (
            v
            + (0.0 + kvphi_samp * v_circ * (-1.0)) * phi_hat
            + (kvz_samp * v_circ * (-1.0)) * z_hat
        )

        return x_lead, x_trail, v_lead, v_trail


#####################################################################
# TODO: move this to a more general location.


@partial_jit()
def dphidr(potential: AbstractPotentialBase, x: Vec3, t: FloatScalar) -> Vec3:
    """Compute the derivative of the potential at a position x.

    Parameters
    ----------
    potential: AbstractPotentialBase
        The gravitational potential.
    x: Array[Any, (3,)]
        3d position (x, y, z) in [kpc]
    t: Numeric
        Time in [Myr]

    Returns
    -------
    Array:
        Derivative of potential in [1/Myr]
    """
    r_hat = x / xp.linalg.vector_norm(x)
    return xp.sum(potential.gradient(x, t) * r_hat)


@partial_jit()
def d2phidr2(
    potential: AbstractPotentialBase, x: Vec3, /, t: FloatOrIntScalarLike
) -> FloatScalar:
    """Compute the second derivative of the potential.

    At a position x (in the simulation frame).

    Parameters
    ----------
    potential: AbstractPotentialBase
        The gravitational potential.
    x: Array[Any, (3,)]
        3d position (x, y, z) in [kpc]
    t: Numeric
        Time in [Myr]

    Returns
    -------
    Array:
        Second derivative of force (per unit mass) in [1/Myr^2]

    Examples
    --------
    >>> from galax.potential import NFWPotential
    >>> from galax.units import galactic
    >>> pot = NFWPotential(m=1e12, r_s=20.0, units=galactic)
    >>> d2phidr2(pot, xp.asarray([8.0, 0.0, 0.0]), t=0)
    Array(-0.00017469, dtype=float64)
    """
    r_hat = x / xp.linalg.vector_norm(x)
    dphi_dr_func = lambda x: xp.sum(potential.gradient(x, t) * r_hat)  # noqa: E731
    return xp.sum(grad(dphi_dr_func)(x) * r_hat)


@partial_jit()
def orbital_angular_velocity(x: Vec3, v: Vec3, /) -> Vec3:
    """Compute the orbital angular velocity about the origin.

    Arguments:
    ---------
    x: Array[Any, (3,)]
        3d position (x, y, z) in [length]
    v: Array[Any, (3,)]
        3d velocity (v_x, v_y, v_z) in [length/time]

    Returns:
    -------
    Array
        Angular velocity in [rad/time]

    Examples:
    --------
    >>> x = xp.asarray([8.0, 0.0, 0.0])
    >>> v = xp.asarray([8.0, 0.0, 0.0])
    >>> orbital_angular_velocity(x, v)
    Array([0., 0., 0.], dtype=float64)
    """
    r = xp.linalg.vector_norm(x)
    return jnp.cross(x, v) / r**2


@partial_jit()
def orbital_angular_velocity_mag(x: Vec3, v: Vec3, /) -> FloatScalar:
    """Compute the magnitude of the angular momentum in the simulation frame.

    Arguments:
    ---------
    x: Array[Any, (3,)]
        3d position (x, y, z) in [kpc]
    v: Array[Any, (3,)]
        3d velocity (v_x, v_y, v_z) in [kpc/Myr]

    Returns:
    -------
    Array
        Magnitude of angular momentum in [rad/Myr]

    Examples:
    --------
    >>> x = xp.asarray([8.0, 0.0, 0.0])
    >>> v = xp.asarray([8.0, 0.0, 0.0])
    >>> orbital_angular_velocity_mag(x, v)
    Array(0., dtype=float64)
    """
    return xp.linalg.vector_norm(orbital_angular_velocity(x, v))


@partial_jit()
def tidal_radius(
    potential: AbstractPotentialBase,
    x: Vec3,
    v: Vec3,
    /,
    prog_mass: FloatScalar,
    t: FloatOrIntScalarLike,
) -> FloatScalar:
    """Compute the tidal radius of a cluster in the potential.

    Parameters
    ----------
    potential: AbstractPotentialBase
        The gravitational potential of the host.
    x: Array[float, (3,)]
        3d position (x, y, z) in [kpc]
    v: Array[float, (3,)]
        3d velocity (v_x, v_y, v_z) in [kpc/Myr]
    prog_mass : Array[float, ()]
        Cluster mass in [Msol]
    t: Array[float | int, ()] | float | int
        Time in [Myr]

    Returns
    -------
    Array[float, ""]] :
        Tidal radius of the cluster in [kpc]

    Examples
    --------
    >>> from galax.potential import NFWPotential
    >>> from galax.units import galactic
    >>> pot = NFWPotential(m=1e12, r_s=20.0, units=galactic)
    >>> x=xp.asarray([8.0, 0.0, 0.0])
    >>> v=xp.asarray([8.0, 0.0, 0.0])
    >>> tidal_radius(pot, x, v, prog_mass=1e4, t=0)
    Array(0.06362136, dtype=float64)
    """
    return (
        potential._G  # noqa: SLF001
        * prog_mass
        / (orbital_angular_velocity_mag(x, v) ** 2 - d2phidr2(potential, x, t))
    ) ** (1.0 / 3.0)


@partial_jit()
def lagrange_points(
    potential: AbstractPotentialBase,
    x: Vec3,
    v: Vec3,
    prog_mass: FloatScalar,
    t: FloatScalar,
) -> tuple[Vec3, Vec3]:
    """Compute the lagrange points of a cluster in a host potential.

    Parameters
    ----------
    potential: AbstractPotentialBase
        The gravitational potential of the host.
    x: Array
        3d position (x, y, z)
    v: Array
        3d velocity (v_x, v_y, v_z)
    prog_mass: Array
        Cluster mass.
    t: Array
        Time.
    """
    r_t = tidal_radius(potential, x, v, prog_mass, t)
    r_hat = x / xp.linalg.vector_norm(x)
    L_1 = x - r_hat * r_t  # close
    L_2 = x + r_hat * r_t  # far
    return L_1, L_2
