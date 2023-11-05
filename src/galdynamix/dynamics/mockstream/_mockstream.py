"""galdynamix: Galactic Dynamix in Jax"""
# ruff: noqa: F403

from __future__ import annotations

__all__ = ["MockStreamGenerator"]

from typing import Any

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import PotentialBase
from galdynamix.utils import jit_method

from ._df import BaseStreamDF


class MockStreamGenerator(eqx.Module):
    df: BaseStreamDF
    potential: PotentialBase
    progenitor_potential: PotentialBase | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "self_gravity", self.progenitor_potential is not None)

    # ==========================================================================

    @jit_method(static_argnames=("seed_num",))
    def _gen_stream_ics(
        self, ts: jt.Array, prog_w0: jt.Array, prog_mass: jt.Array, *, seed_num: int
    ) -> jt.Array:
        ws_jax = self.potential.integrate_orbit(
            prog_w0, t0=xp.min(ts), t1=xp.max(ts), ts=ts
        )

        def scan_fun(carry: Any, t: Any) -> Any:
            i, pos_close, pos_far, vel_close, vel_far = carry
            sample_outputs = self.df.sample(
                self.potential,
                ws_jax[i, :3],
                ws_jax[i, 3:],
                prog_mass,
                i,
                t,
                seed_num=seed_num,
            )
            return [i + 1, *sample_outputs], list(sample_outputs)

        init_carry = [
            0,
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
        ]
        # final_state, all_states = jax.lax.scan(scan_fun, init_carry, ts[1:])
        # pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
        return jax.lax.scan(scan_fun, init_carry, ts[1:])[1]

    @jit_method(static_argnames=("seed_num",))
    def _run_scan(
        self, ts: jt.Array, prog_w0: jt.Array, prog_mass: jt.Array, *, seed_num: int
    ) -> tuple[jt.Array, jt.Array]:
        """
        Generate stellar stream by scanning over the release model/integration. Better for CPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self._gen_stream_ics(
            ts, prog_w0, prog_mass, seed_num=seed_num
        )

        @jax.jit  # type: ignore[misc]
        def scan_fun(carry: Any, particle_idx: Any) -> Any:
            i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            curr_particle_w0_close = xp.hstack([pos_close_curr, vel_close_curr])
            curr_particle_w0_far = xp.hstack([pos_far_curr, vel_far_curr])
            w0_lead_trail = xp.vstack([curr_particle_w0_close, curr_particle_w0_far])

            minval, maxval = ts[i], ts[-1]

            def integrate_different_ics(ics: jt.Array) -> jt.Array:
                return self.potential.integrate_orbit(ics, minval, maxval, None)[0]

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

    @jit_method(static_argnames=("seed_num",))
    def _run_vmap(
        self, ts: jt.Array, prog_w0: jt.Array, prog_mass: jt.Array, *, seed_num: int
    ) -> tuple[jt.Array, jt.Array]:
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self._gen_stream_ics(
            ts, prog_w0, prog_mass, seed_num
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

            w_particle_close = self.integrate_orbit(
                curr_particle_w0_close, t_release, t_final, None
            )[0]
            w_particle_far = self.integrate_orbit(
                curr_particle_w0_far, t_release, t_final, None
            )[0]

            return w_particle_close, w_particle_far

        particle_ids = xp.arange(len(pos_close_arr))

        return jax.vmap(
            single_particle_integrate,
            in_axes=(0, 0, 0, 0, 0),
        )(particle_ids, pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)

    @jit_method(static_argnames=("seed_num", "vmapped"))
    def run(
        self,
        ts: jt.Array,
        prog_w0: jt.Array,
        prog_mass: jt.Array,
        *,
        seed_num: int,
        vmapped: bool = False,
    ) -> tuple[jt.Array, jt.Array]:
        if vmapped:
            return self._run_vmap(ts, prog_w0, prog_mass, seed_num=seed_num)
        else:
            return self._run_scan(ts, prog_w0, prog_mass, seed_num=seed_num)

    # ==========================================================================
