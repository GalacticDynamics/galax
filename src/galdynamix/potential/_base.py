from __future__ import annotations

__all__ = ["PotentialBase"]

import abc
from typing import Any

import jax
import jax.numpy as xp
import jax.typing as jt
from astropy.constants import G
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from gala.units import UnitSystem, dimensionless

from galdynamix.utils import jit_method


class PotentialBase(metaclass=abc.ABCMeta):
    """Potential Class."""

    units: UnitSystem
    _G: float

    def __init__(self, units: UnitSystem | None, params: dict[str, Any]) -> None:
        if units is None:
            units = dimensionless

        self.units = UnitSystem(units)

        if self.units == dimensionless:
            self._G = 1
        else:
            self._G = G.decompose(self.units).value

        for name, param in params.items():
            if hasattr(param, "unit"):
                param = param.decompose(self.units).value  # noqa: PLW2901
            setattr(self, name, param)

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the potential energy at the given position(s)."""
        raise NotImplementedError

    ###########################################################################
    # Core methods that use the above implemented functions
    #

    @jit_method()
    def gradient(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the gradient."""
        return jax.grad(self.energy)(q, t)

    @jit_method()
    def density(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        lap = xp.trace(jax.hessian(self.energy)(q, t))
        return lap / (4 * xp.pi * self._G)

    @jit_method()
    def hessian(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        return jax.hessian(self.energy)(q, t)

    @jit_method()
    def acceleration(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        return -self.gradient(q, t)

    ###########################################################################

    @jit_method()
    def _jacobian_force_mw(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        return jax.jacfwd(self.gradient)(q, t)

    @jit_method()
    def d2phidr2_mw(self, x: jt.Array, /, t: jt.Array) -> jt.Array:
        """
        Computes the second derivative of the Milky Way potential at a position x (in the simulation frame)

        Parameters
        ----------
          x: 3d position (x, y, z) in [kpc]

        Returns
        -------
        Array:
          Second derivative of force (per unit mass) in [1/Myr^2]

        Examples
        --------
        >>> d2phidr2_mw(x=xp.array([8.0, 0.0, 0.0]))
        """
        rad = xp.linalg.norm(x)
        r_hat = x / rad
        dphi_dr_func = lambda x: xp.sum(self.gradient(x, t) * r_hat)  # noqa: E731
        return xp.sum(jax.grad(dphi_dr_func)(x) * r_hat)

        ##return xp.matmul(xp.transpose(x), xp.matmul(self._jacobian_force_mw(x, t), x)) / rad**2

    @jit_method()
    def omega(self, x: jt.Array, v: jt.Array) -> jt.Array:
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
        >>> omega(x=xp.array([8.0, 0.0, 0.0]), v=xp.array([8.0, 0.0, 0.0]))
        """
        rad = xp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        omega_vec = xp.cross(x, v) / (rad**2)
        return xp.linalg.norm(omega_vec)

    @jit_method()
    def tidalr_mw(
        self, x: jt.Array, v: jt.Array, /, Msat: jt.Array, t: jt.Array
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
        >>> tidalr_mw(x=xp.array([8.0, 0.0, 0.0]), v=xp.array([8.0, 0.0, 0.0]), Msat=1e4)
        """
        return (self._G * Msat / (self.omega(x, v) ** 2 - self.d2phidr2_mw(x, t))) ** (
            1.0 / 3.0
        )

    @jit_method()
    def lagrange_pts(
        self, x: jt.Array, v: jt.Array, Msat: jt.Array, t: jt.Array
    ) -> tuple[jt.Array, jt.Array]:
        r_tidal = self.tidalr_mw(x, v, Msat, t)
        r_hat = x / xp.linalg.norm(x)
        L_close = x - r_hat * r_tidal
        L_far = x + r_hat * r_tidal
        return L_close, L_far

    @jit_method()
    def velocity_acceleration(self, t: jt.Array, xv: jt.Array, args: Any) -> jt.Array:
        x, v = xv[:3], xv[3:]
        acceleration = -self.gradient(x, t)
        return xp.hstack([v, acceleration])

    @jit_method()
    def orbit_integrator_run(
        self, w0: jt.Array, t0: jt.Array, t1: jt.Array, ts: jt.Array
    ) -> jt.Array:
        term = ODETerm(self.velocity_acceleration)
        solver = Dopri5()
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=False)
        rtol: float = 1e-7
        atol: float = 1e-7
        stepsize_controller = PIDController(rtol=rtol, atol=atol)
        max_steps: int = 16**3
        # t0 = t0  # 0.0
        # t1 = t1  # 4000.
        dense = False
        # y0= w_init

        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=None,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,
            max_steps=(max_steps if dense else None),
        )
        return solution.ys

    @jit_method()
    def release_model(
        self,
        x: jt.Array,
        v: jt.Array,
        Msat: jt.Array,
        i: int,
        t: jt.Array,
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

        L_close, L_far = self.lagrange_pts(x, v, Msat, t)  # each is an xyz array

        omega_val = self.omega(x, v)

        r = xp.linalg.norm(x)
        r_hat = x / r
        r_tidal = self.tidalr_mw(x, v, Msat, t)
        rel_v = omega_val * r_tidal  # relative velocity

        # circlar_velocity
        dphi_dr = xp.sum(self.gradient(x, t) * r_hat)
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
    def gen_stream_ics(
        self, ts: jt.Array, prog_w0: jt.Array, Msat: jt.Array, seed_num: int
    ) -> jt.Array:
        ws_jax = self.orbit_integrator_run(prog_w0, xp.min(ts), xp.max(ts), ts)

        def scan_fun(carry: Any, t: Any) -> Any:
            i, pos_close, pos_far, vel_close, vel_far = carry
            pos_close_new, pos_far_new, vel_close_new, vel_far_new = self.release_model(
                ws_jax[i, :3], ws_jax[i, 3:], Msat, i, t, seed_num
            )
            return [i + 1, pos_close_new, pos_far_new, vel_close_new, vel_far_new], [
                pos_close_new,
                pos_far_new,
                vel_close_new,
                vel_far_new,
            ]  # [i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new]

        # init_carry = [0, 0, 0, 0, 0]
        init_carry = [
            0,
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
        ]
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, ts[1:])
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
        return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr

    @jit_method()
    def gen_stream_scan(
        self, ts: jt.Array, prog_w0: jt.Array, Msat: jt.Array, seed_num: int
    ) -> tuple[jt.Array, jt.Array]:
        """
        Generate stellar stream by scanning over the release model/integration. Better for CPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(
            ts, prog_w0, Msat, seed_num
        )

        @jax.jit  # type: ignore[misc]
        def scan_fun(carry: Any, particle_idx: Any) -> Any:
            i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            curr_particle_w0_close = xp.hstack([pos_close_curr, vel_close_curr])
            curr_particle_w0_far = xp.hstack([pos_far_curr, vel_far_curr])
            w0_lead_trail = xp.vstack([curr_particle_w0_close, curr_particle_w0_far])

            minval, maxval = ts[i], ts[-1]

            def integrate_different_ics(ics: jt.Array) -> jt.Array:
                return self.orbit_integrator_run(ics, minval, maxval, None)[0]

            w_particle_close, w_particle_far = jax.vmap(
                integrate_different_ics, in_axes=(0,)
            )(
                w0_lead_trail
            )  # vmap over leading and trailing arm

            return [
                i + 1,
                pos_close_arr[i + 1, :],
                pos_far_arr[i + 1, :],
                vel_close_arr[i + 1, :],
                vel_far_arr[i + 1, :],
            ], [w_particle_close, w_particle_far]

        init_carry = [
            0,
            pos_close_arr[0, :],
            pos_far_arr[0, :],
            vel_close_arr[0, :],
            vel_far_arr[0, :],
        ]
        particle_ids = xp.arange(len(pos_close_arr))
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        lead_arm, trail_arm = all_states
        return lead_arm, trail_arm

    @jit_method()
    def gen_stream_vmapped(
        self, ts: jt.Array, prog_w0: jt.Array, Msat: jt.Array, *, seed_num: int
    ) -> jt.Array:
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(
            ts, prog_w0, Msat, seed_num
        )

        @jax.jit  # type: ignore[misc]
        def single_particle_integrate(
            particle_number: int,
            pos_close_curr: jt.Array,
            pos_far_curr: jt.Array,
            vel_close_curr: jt.Array,
            vel_far_curr: jt.Array,
        ) -> tuple[jt.Array, jt.Array]:
            curr_particle_w0_close = xp.hstack([pos_close_curr, vel_close_curr])
            curr_particle_w0_far = xp.hstack([pos_far_curr, vel_far_curr])
            t_release = ts[particle_number]
            t_final = ts[-1] + 0.01

            w_particle_close = self.orbit_integrator_run(
                curr_particle_w0_close, t_release, t_final, None
            )[0]
            w_particle_far = self.orbit_integrator_run(
                curr_particle_w0_far, t_release, t_final, None
            )[0]

            return w_particle_close, w_particle_far

        particle_ids = xp.arange(len(pos_close_arr))

        return jax.vmap(
            single_particle_integrate,
            in_axes=(
                0,
                0,
                0,
                0,
                0,
            ),
        )(particle_ids, pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)
