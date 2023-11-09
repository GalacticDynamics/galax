"""galdynamix: Galactic Dynamix in Jax"""
# ruff: noqa: F403

from __future__ import annotations

__all__ = ["FardalStreamDF"]


import jax
import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.utils import partial_jit

from .base import AbstractStreamDF


class FardalStreamDF(AbstractStreamDF):
    @partial_jit(static_argnums=(0,), static_argnames=("seed_num",))
    def _sample(
        self,
        # parts of gala's ``prog_orbit``
        potential: AbstractPotentialBase,
        x: jt.Array,
        v: jt.Array,
        prog_mass: jt.Array,
        i: int,
        t: jt.Array,
        *,
        seed_num: int,
    ) -> tuple[jt.Array, jt.Array, jt.Array, jt.Array]:
        """
        Simplification of particle spray: just release particles in gaussian blob at each lagrange point.
        User sets the spatial and velocity dispersion for the "leaking" of particles
        """
        # Random number generation
        # TODO: change random key handling... need to do all of the sampling up front...
        key_master = jax.random.PRNGKey(seed_num)
        random_ints = jax.random.randint(
            key=key_master, shape=(4,), minval=0, maxval=1000
        )
        keya = jax.random.PRNGKey(i * random_ints[0])
        keyb = jax.random.PRNGKey(i * random_ints[1])
        keyc = jax.random.PRNGKey(i * random_ints[2])
        keyd = jax.random.PRNGKey(i * random_ints[3])

        # ---------------------------

        omega_val = orbital_angular_velocity_mag(x, v)

        r = xp.linalg.norm(x)
        r_hat = x / r
        r_tidal = tidal_radius(potential, x, v, prog_mass, t)
        rel_v = omega_val * r_tidal  # relative velocity

        # circlar_velocity
        v_circ = rel_v  ##xp.sqrt( r*dphi_dr )

        L_vec = xp.cross(x, v)
        z_hat = L_vec / xp.linalg.norm(L_vec)

        phi_vec = v - xp.sum(v * r_hat) * r_hat
        phi_hat = phi_vec / xp.linalg.norm(phi_vec)

        kr_bar = 2.0
        kvphi_bar = 0.3
        ####################kvt_bar = 0.3 ## FROM GALA

        kz_bar = 0.0
        kvz_bar = 0.0

        sigma_kr = 0.5
        sigma_kvphi = 0.5
        sigma_kz = 0.5
        sigma_kvz = 0.5
        ##############sigma_kvt = 0.5 ##FROM GALA

        kr_samp = kr_bar + jax.random.normal(keya, shape=(1,)) * sigma_kr
        kvphi_samp = kr_samp * (
            kvphi_bar + jax.random.normal(keyb, shape=(1,)) * sigma_kvphi
        )
        kz_samp = kz_bar + jax.random.normal(keyc, shape=(1,)) * sigma_kz
        kvz_samp = kvz_bar + jax.random.normal(keyd, shape=(1,)) * sigma_kvz
        ########kvt_samp = kvt_bar + jax.random.normal(keye,shape=(1,))*sigma_kvt

        ## Trailing arm
        pos_trail = x + kr_samp * r_hat * (r_tidal)  # nudge out
        pos_trail = pos_trail + z_hat * kz_samp * (
            r_tidal / 1.0
        )  # r #nudge above/below orbital plane
        v_trail = (
            v + (0.0 + kvphi_samp * v_circ * (1.0)) * phi_hat
        )  # v + (0.0 + kvphi_samp*v_circ*(-r_tidal/r))*phi_hat #nudge velocity along tangential direction
        v_trail = (
            v_trail + (kvz_samp * v_circ * (1.0)) * z_hat
        )  # v_trail + (kvz_samp*v_circ*(-r_tidal/r))*z_hat #nudge velocity along vertical direction

        ## Leading arm
        pos_lead = x + kr_samp * r_hat * (-r_tidal)  # nudge in
        pos_lead = pos_lead + z_hat * kz_samp * (
            -r_tidal / 1.0
        )  # r #nudge above/below orbital plane
        v_lead = (
            v + (0.0 + kvphi_samp * v_circ * (-1.0)) * phi_hat
        )  # v + (0.0 + kvphi_samp*v_circ*(r_tidal/r))*phi_hat #nudge velocity along tangential direction
        v_lead = (
            v_lead + (kvz_samp * v_circ * (-1.0)) * z_hat
        )  # v_lead + (kvz_samp*v_circ*(r_tidal/r))*z_hat #nudge velocity against vertical direction

        return pos_lead, pos_trail, v_lead, v_trail


#####################################################################
# TODO: move this to a more general location.


@partial_jit()
def dphidr(potential: AbstractPotentialBase, x: jt.Array, t: jt.Numeric) -> jt.Array:
    """Computes the derivative of the potential at a position x.

    Parameters
    ----------
    potential: AbstractPotentialBase
        The gravitational potential.
    x: Array[(3,), Any]
        3d position (x, y, z) in [kpc]
    t: Numeric
        Time in [Myr]

    Returns
    -------
    Array:
        Derivative of potential in [1/Myr]
    """
    r_hat = x / xp.linalg.norm(x)
    return xp.sum(potential.gradient(x, t) * r_hat)


@partial_jit()
def d2phidr2(potential: AbstractPotentialBase, x: jt.Array, t: jt.Numeric) -> jt.Array:
    """Computes the second derivative of the potential at a position x (in the simulation frame).

    Parameters
    ----------
    potential: AbstractPotentialBase
        The gravitational potential.
    x: Array[(3,), Any]
        3d position (x, y, z) in [kpc]
    t: Numeric
        Time in [Myr]

    Returns
    -------
    Array:
        Second derivative of force (per unit mass) in [1/Myr^2]

    Examples
    --------
    >>> d2phidr2(x=xp.array([8.0, 0.0, 0.0]))
    """
    r_hat = x / xp.linalg.norm(x)
    dphi_dr_func = lambda x: xp.sum(potential.gradient(x, t) * r_hat)  # noqa: E731
    return xp.sum(jax.grad(dphi_dr_func)(x) * r_hat)


@partial_jit()
def orbital_angular_velocity(x: jt.Array, v: jt.Array, /) -> jt.Array:
    """Computes the orbital angular velocity about the origin.

    Arguments
    ---------
    x: Array[(3,), Any]
        3d position (x, y, z) in [length]
    v: Array[(3,), Any]
        3d velocity (v_x, v_y, v_z) in [length/time]

    Returns
    -------
    Array
        Angular velocity in [rad/time]

    Examples
    --------
    >>> orbital_angular_velocity(x=xp.array([8.0, 0.0, 0.0]), v=xp.array([8.0, 0.0, 0.0]))
    """
    r = xp.linalg.norm(x)
    return xp.cross(x, v) / r**2


@partial_jit()
def orbital_angular_velocity_mag(x: jt.Array, v: jt.Array, /) -> jt.Array:
    """Computes the magnitude of the angular momentum in the simulation frame.

    Arguments
    ---------
    x: Array[(3,), Any]
        3d position (x, y, z) in [kpc]
    v: Array[(3,), Any]
        3d velocity (v_x, v_y, v_z) in [kpc/Myr]

    Returns
    -------
    Array
        Magnitude of angular momentum in [rad/Myr]

    Examples
    --------
    >>> orbital_angular_velocity_mag(x=xp.array([8.0, 0.0, 0.0]), v=xp.array([8.0, 0.0, 0.0]))
    """
    return xp.linalg.norm(orbital_angular_velocity(x, v))


@partial_jit()
def tidal_radius(
    potential: AbstractPotentialBase,
    x: jt.Array,
    v: jt.Array,
    /,
    prog_mass: jt.Array,
    t: jt.Array,
) -> jt.Array:
    """Computes the tidal radius of a cluster in the potential.

    Parameters
    ----------
    x: Array
        3d position (x, y, z) in [kpc]
    v: Array
        3d velocity (v_x, v_y, v_z) in [kpc/Myr]
    prog_mass:
        Cluster mass in [Msol]
    t: Array
        Time in [Myr]

    Returns
    -------
    Array:
        Tidal radius of the cluster in [kpc]

    Examples
    --------
    >>> tidal_radius(x=xp.array([8.0, 0.0, 0.0]), v=xp.array([8.0, 0.0, 0.0]), prog_mass=1e4)
    """
    return (
        potential._G
        * prog_mass
        / (orbital_angular_velocity_mag(x, v) ** 2 - d2phidr2(potential, x, t))
    ) ** (1.0 / 3.0)


@partial_jit()
def lagrange_points(
    potential: AbstractPotentialBase,
    x: jt.Array,
    v: jt.Array,
    prog_mass: jt.Array,
    t: jt.Array,
) -> tuple[jt.Array, jt.Array]:
    """Computes the lagrange points of a cluster in a host potential.

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
    r_hat = x / xp.linalg.norm(x)
    L_1 = x - r_hat * r_t  # close
    L_2 = x + r_hat * r_t  # far
    return L_1, L_2
