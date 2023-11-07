"""galdynamix: Galactic Dynamix in Jax"""
# ruff: noqa: F403

from __future__ import annotations

__all__ = ["MockStreamGenerator"]

from typing import TYPE_CHECKING, Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.utils import partial_jit

from ._df import BaseStreamDF

if TYPE_CHECKING:
    _wifT: TypeAlias = tuple[jt.Array, jt.Array, jt.Array, jt.Array]
    _carryT: TypeAlias = tuple[int, jt.Array, jt.Array, jt.Array, jt.Array]


class MockStreamGenerator(eqx.Module):  # type: ignore[misc]
    df: BaseStreamDF
    potential: AbstractPotentialBase
    progenitor_potential: AbstractPotentialBase | None = None

    @property
    def self_gravity(self) -> bool:
        return self.progenitor_potential is not None

    # ==========================================================================

    @partial_jit(static_argnames=("seed_num",))
    def _stream_ics(
        self, ts: jt.Array, w0: jt.Array, mass: jt.Array, *, seed_num: int
    ) -> jt.Array:
        """Stream Initial Conditions.

        Parameters
        ----------
        ts : array_like
            Array of times to release particles.
        w0 : array_like
            q, p of the progenitor.
        mass : array_like
            Mass of the progenitor.
        """
        ws = self.potential.integrate_orbit(w0, xp.min(ts), xp.max(ts), ts)

        def scan_fun(carry: _carryT, t: Any) -> tuple[_carryT, _wifT]:
            i = carry[0]
            output = self.df.sample(
                self.potential, ws[i, :3], ws[i, 3:], mass, i, t, seed_num=seed_num
            )
            return (i + 1, *output), tuple(output)  # type: ignore[return-value]

        init_carry = (
            0,
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
        )
        return jax.lax.scan(scan_fun, init_carry, ts[1:])[1]

    @partial_jit(static_argnames=("seed_num",))
    def _run_scan(
        self, ts: jt.Array, prog_w0: jt.Array, prog_mass: jt.Array, *, seed_num: int
    ) -> tuple[jt.Array, jt.Array]:
        """
        Generate stellar stream by scanning over the release model/integration. Better for CPU usage.
        """
        q_close, q_far, p_close, p_far = self._stream_ics(
            ts, prog_w0, prog_mass, seed_num=seed_num
        )

        # TODO: make this a separated method
        @jax.jit  # type: ignore[misc]
        def scan_fun(
            carry: _carryT, particle_idx: int
        ) -> tuple[_carryT, tuple[jt.Array, jt.Array]]:
            i, q_close_i, q_far_i, p_close_i, p_far_i = carry
            w0_close_i = xp.hstack([q_close_i, p_close_i])
            w0_far_i = xp.hstack([q_far_i, p_far_i])
            w0_lead_trail = xp.vstack([w0_close_i, w0_far_i])

            minval, maxval = ts[i], ts[-1]
            integ_ics = lambda ics: self.potential.integrate_orbit(  # noqa: E731
                ics, minval, maxval, None
            )[0]
            # vmap over leading and trailing arm
            w_close, w_far = jax.vmap(integ_ics, in_axes=(0,))(w0_lead_trail)
            carry_out = (
                i + 1,
                q_close[i + 1, :],
                q_far[i + 1, :],
                p_close[i + 1, :],
                p_far[i + 1, :],
            )
            return carry_out, (w_close, w_far)

        carry_init = (0, q_close[0, :], q_far[0, :], p_close[0, :], p_far[0, :])
        particle_ids = xp.arange(len(q_close))
        lead_arm, trail_arm = jax.lax.scan(scan_fun, carry_init, particle_ids)[1]
        return lead_arm, trail_arm

    @partial_jit(static_argnames=("seed_num",))
    def _run_vmap(
        self, ts: jt.Array, prog_w0: jt.Array, prog_mass: jt.Array, *, seed_num: int
    ) -> tuple[jt.Array, jt.Array]:
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        """
        q_close_arr, q_far_arr, p_close_arr, p_far_arr = self._stream_ics(
            ts, prog_w0, prog_mass, seed_num=seed_num
        )

        # TODO: make this a separated method
        @jax.jit  # type: ignore[misc]
        def single_particle_integrate(
            i: int,
            q_close_i: jt.Array,
            q_far_i: jt.Array,
            p_close_i: jt.Array,
            p_far_i: jt.Array,
        ) -> tuple[jt.Array, jt.Array]:
            w0_close_i = xp.hstack([q_close_i, p_close_i])
            w0_far_i = xp.hstack([q_far_i, p_far_i])
            t_i = ts[i]
            t_f = ts[-1] + 0.01

            w_close = self.integrate_orbit(w0_close_i, t_i, t_f, None)[0]
            w_far = self.integrate_orbit(w0_far_i, t_i, t_f, None)[0]

            return w_close, w_far

        particle_ids = xp.arange(len(q_close_arr))

        integrator = jax.vmap(single_particle_integrate, in_axes=(0, 0, 0, 0, 0))
        w_close, w_far = integrator(
            particle_ids, q_close_arr, q_far_arr, p_close_arr, p_far_arr
        )
        return w_close, w_far

    @partial_jit(static_argnames=("seed_num", "vmapped"))
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
        return self._run_scan(ts, prog_w0, prog_mass, seed_num=seed_num)

    # ==========================================================================
