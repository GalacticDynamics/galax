"""galdynamix: Galactic Dynamix in Jax"""
# ruff: noqa: F403

from __future__ import annotations

__all__ = ["MockStreamGenerator"]

from typing import TYPE_CHECKING, TypeAlias

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
    # progenitor_potential: AbstractPotentialBase | None = None

    # @property
    # def self_gravity(self) -> bool:
    #     return self.progenitor_potential is not None

    # ==========================================================================

    @partial_jit(static_argnames=("seed_num",))
    def _run_scan(
        self, ts: jt.Array, prog_w0: jt.Array, prog_mass: jt.Array, *, seed_num: int
    ) -> tuple[tuple[jt.Array, jt.Array], jt.Array]:
        """Generate stellar stream by scanning over the release model/integration.

        Better for CPU usage.
        """
        # Integrate the progenitor orbit
        prog_ws = self.potential.integrate_orbit(prog_w0, xp.min(ts), xp.max(ts), ts)

        # Generate stream initial conditions along the integrated progenitor orbit
        x_lead, x_trail, v_lead, v_trail = self.df.sample(
            self.potential, prog_ws, ts, prog_mass, seed_num=seed_num
        )

        def scan_fn(
            carry: _carryT, particle_idx: int
        ) -> tuple[_carryT, tuple[jt.Array, jt.Array]]:
            i, x_lead_i, x_trail_i, v_lead_i, v_trail_i = carry
            w0_lead_i = xp.hstack([x_lead_i, v_lead_i])
            w0_trail_i = xp.hstack([x_trail_i, v_trail_i])
            w0_lead_trail = xp.vstack([w0_lead_i, w0_trail_i])

            minval, maxval = ts[i], ts[-1]
            integ_ics = lambda ics: self.potential.integrate_orbit(  # noqa: E731
                ics, minval, maxval, None
            )[0]
            # vmap over leading and trailing arm
            w_lead, w_trail = jax.vmap(integ_ics, in_axes=(0,))(w0_lead_trail)
            carry_out = (
                i + 1,
                x_lead[i + 1, :],
                x_trail[i + 1, :],
                v_lead[i + 1, :],
                v_trail[i + 1, :],
            )
            return carry_out, (w_lead, w_trail)

        carry_init = (0, x_lead[0, :], x_trail[0, :], v_lead[0, :], v_trail[0, :])
        particle_ids = xp.arange(len(x_lead))
        lead_arm, trail_arm = jax.lax.scan(scan_fn, carry_init, particle_ids)[1]
        return (lead_arm, trail_arm), prog_ws

    @partial_jit(static_argnames=("seed_num",))
    def _run_vmap(
        self, ts: jt.Array, prog_w0: jt.Array, prog_mass: jt.Array, *, seed_num: int
    ) -> tuple[tuple[jt.Array, jt.Array], jt.Array]:
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        """
        # Integrate the progenitor orbit
        prog_ws = self.potential.integrate_orbit(prog_w0, xp.min(ts), xp.max(ts), ts)

        # Generate stream initial conditions along the integrated progenitor orbit
        x_lead, x_trail, v_lead, v_trail = self.df.sample(
            self.potential, prog_ws, ts, prog_mass, seed_num=seed_num
        )

        # TODO: make this a separated method
        @jax.jit  # type: ignore[misc]
        def single_particle_integrate(
            i: int,
            x_lead_i: jt.Array,
            x_trail_i: jt.Array,
            v_lead_i: jt.Array,
            v_trail_i: jt.Array,
        ) -> tuple[jt.Array, jt.Array]:
            w0_lead_i = xp.hstack([x_lead_i, v_lead_i])
            w0_trail_i = xp.hstack([x_trail_i, v_trail_i])
            t_i = ts[i]
            t_f = ts[-1] + 0.01

            w_lead = self.integrate_orbit(w0_lead_i, t_i, t_f, None)[0]
            w_trail = self.integrate_orbit(w0_trail_i, t_i, t_f, None)[0]

            return w_lead, w_trail

        particle_ids = xp.arange(len(x_lead))

        integrator = jax.vmap(single_particle_integrate, in_axes=(0, 0, 0, 0, 0))
        w_lead, w_trail = integrator(particle_ids, x_lead, x_trail, v_lead, v_trail)
        return (w_lead, w_trail), prog_ws

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
