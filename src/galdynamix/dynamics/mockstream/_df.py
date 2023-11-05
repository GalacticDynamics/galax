"""galdynamix: Galactic Dynamix in Jax"""
# ruff: noqa: F403

from __future__ import annotations

__all__ = ["BaseStreamDF", "FardalStreamDF"]

import abc

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._hamiltonian import Hamiltonian
from galdynamix.utils import jit_method


class BaseStreamDF(eqx.Module):
    lead: bool = eqx.field(default=True, static=True)
    trail: bool = eqx.field(default=True, static=True)

    def __post_init__(self) -> None:
        if not self.lead and not self.trail:
            msg = "You must generate either leading or trailing tails (or both!)"
            raise ValueError(msg)

    @abc.abstractmethod
    def sample(
        self,
        x: jt.Array,
        v: jt.Array,
        prog_mass: jt.Array,
        i: int,
        t: jt.Array,
        *,
        hamiltonian: Hamiltonian,
        seed_num: int,
    ) -> tuple[jt.Array, jt.Array, jt.Array, jt.Array]:
        """Sample the DF."""
        raise NotImplementedError


# ==========================================================================


class FardalStreamDF(BaseStreamDF):
    @jit_method(static_argnames=("seed_num",))
    def sample(
        self,
        x: jt.Array,
        v: jt.Array,
        prog_mass: jt.Array,
        i: int,
        t: jt.Array,
        *,
        hamiltonian: Hamiltonian,
        seed_num: int,
    ) -> tuple[jt.Array, jt.Array, jt.Array, jt.Array]:
        """
        Simplification of particle spray: just release particles in gaussian blob at each lagrange point.
        User sets the spatial and velocity dispersion for the "leaking" of particles
        TODO: change random key handling... need to do all of the sampling up front...
        """
        key_master = jax.random.PRNGKey(seed_num)
        random_ints = jax.random.randint(
            key=key_master, shape=(5,), minval=0, maxval=1000
        )

        keya = jax.random.PRNGKey(i * random_ints[0])  # jax.random.PRNGKey(i*13)
        keyb = jax.random.PRNGKey(i * random_ints[1])  # jax.random.PRNGKey(i*23)

        keyc = jax.random.PRNGKey(i * random_ints[2])  # jax.random.PRNGKey(i*27)
        keyd = jax.random.PRNGKey(i * random_ints[3])  # jax.random.PRNGKey(i*3)
        keye = jax.random.PRNGKey(i * random_ints[4])  # jax.random.PRNGKey(i*17)

        L_close, L_far = self._lagrange_pts(
            x, v, prog_mass, t, hamiltonian=hamiltonian
        )  # each is an xyz array

        omega_val = self._omega(x, v)

        r = xp.linalg.norm(x)
        r_hat = x / r
        r_tidal = self._tidalr_mw(x, v, prog_mass, t, hamiltonian=hamiltonian)
        rel_v = omega_val * r_tidal  # relative velocity

        # circlar_velocity
        dphi_dr = xp.sum(hamiltonian.potential.gradient(x, t) * r_hat)
        v_circ = rel_v  ##xp.sqrt( r*dphi_dr )

        L_vec = xp.cross(x, v)
        z_hat = L_vec / xp.linalg.norm(L_vec)

        phi_vec = v - xp.sum(v * r_hat) * r_hat
        phi_hat = phi_vec / xp.linalg.norm(phi_vec)
        vt_sat = xp.sum(v * phi_hat)

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

    @jit_method()
    def _lagrange_pts(
        self,
        x: jt.Array,
        v: jt.Array,
        Msat: jt.Array,
        t: jt.Array,
        *,
        hamiltonian: Hamiltonian,
    ) -> tuple[jt.Array, jt.Array]:
        r_tidal = self._tidalr_mw(x, v, Msat, t, hamiltonian=hamiltonian)
        r_hat = x / xp.linalg.norm(x)
        L_close = x - r_hat * r_tidal
        L_far = x + r_hat * r_tidal
        return L_close, L_far

    @jit_method()
    def _d2phidr2_mw(
        self, x: jt.Array, /, t: jt.Array, *, hamiltonian: Hamiltonian
    ) -> jt.Array:
        """
        Computes the second derivative of the potential at a position x (in the simulation frame)

        Parameters
        ----------
          x: 3d position (x, y, z) in [kpc]

        Returns
        -------
        Array:
          Second derivative of force (per unit mass) in [1/Myr^2]

        Examples
        --------
        >>> _d2phidr2_mw(x=xp.array([8.0, 0.0, 0.0]))
        """
        rad = xp.linalg.norm(x)
        r_hat = x / rad
        dphi_dr_func = lambda x: xp.sum(hamiltonian.potential.gradient(x, t) * r_hat)  # noqa: E731
        return xp.sum(jax.grad(dphi_dr_func)(x) * r_hat)

    @jit_method()
    def _omega(self, x: jt.Array, v: jt.Array) -> jt.Array:
        """
        Computes the magnitude of the angular momentum in the simulation frame

        Arguments
        ---------
        Array
            3d position (x, y, z) in [kpc]
        Array
            3d velocity (v_x, v_y, v_z) in [kpc/Myr]

        Returns
        -------
        Array
            Magnitude of angular momentum in [rad/Myr]

        Examples
        --------
        >>> _omega(x=xp.array([8.0, 0.0, 0.0]), v=xp.array([8.0, 0.0, 0.0]))
        """
        rad = xp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        omega_vec = xp.cross(x, v) / (rad**2)
        return xp.linalg.norm(omega_vec)

    @jit_method()
    def _tidalr_mw(
        self,
        x: jt.Array,
        v: jt.Array,
        /,
        Msat: jt.Array,
        t: jt.Array,
        *,
        hamiltonian: Hamiltonian,
    ) -> jt.Array:
        """Computes the tidal radius of a cluster in the potential.

        Parameters
        ----------
          x: 3d position (x, y, z) in [kpc]
          v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
          Msat: Cluster mass in [Msol]

        Returns
        -------
        Array:
          Tidal radius of the cluster in [kpc]

        Examples
        --------
        >>> _tidalr_mw(x=xp.array([8.0, 0.0, 0.0]), v=xp.array([8.0, 0.0, 0.0]), Msat=1e4)
        """
        return (
            hamiltonian.potential._G
            * Msat
            / (
                self._omega(x, v) ** 2
                - self._d2phidr2_mw(x, t, hamiltonian=hamiltonian)
            )
        ) ** (1.0 / 3.0)
