"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = ["MockStreamGenerator"]

from typing import TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.utils import partial_jit

from ._df import AbstractStreamDF

if TYPE_CHECKING:
    Carry: TypeAlias = tuple[int, jt.Array, jt.Array]

    from galdynamix.dynamics._orbit import Orbit


class MockStreamGenerator(eqx.Module):  # type: ignore[misc]
    df: AbstractStreamDF
    potential: AbstractPotentialBase

    # ==========================================================================

    @partial_jit(static_argnames=("seed_num",))
    def _run_scan(
        self,
        ts: jt.Array,
        prog_w0: jt.Array,
        prog_mass: jt.Array,
        *,
        seed_num: int,
    ) -> tuple[tuple[jt.Array, jt.Array], Orbit]:
        """Generate stellar stream by scanning over the release model/integration.

        Better for CPU usage.
        """
        # Integrate the progenitor orbit
        prog_o = self.potential.integrate_orbit(prog_w0, xp.min(ts), xp.max(ts), ts)

        # Generate stream initial conditions along the integrated progenitor orbit
        mock0_lead, mock0_trail = self.df.sample(
            self.potential, prog_o, prog_mass, seed_num=seed_num
        )
        qp0_lead = mock0_lead.qp
        qp0_trail = mock0_trail.qp

        def scan_fn(carry: Carry, idx: int) -> tuple[Carry, tuple[jt.Array, jt.Array]]:
            i, qp0_lead_i, qp0_trail_i = carry
            qp0_lead_trail = xp.vstack([qp0_lead_i, qp0_trail_i])
            t_i, t_f = ts[i], ts[-1]

            def integ_ics(ics: jt.Array) -> jt.Array:
                return self.potential.integrate_orbit(ics, t_i, t_f, None).qp[0]

            # vmap over leading and trailing arm
            qp_lead, qp_trail = jax.vmap(integ_ics, in_axes=(0,))(qp0_lead_trail)
            carry_out = (i + 1, qp0_lead[i + 1, :], qp0_trail[i + 1, :])
            return carry_out, (qp_lead, qp_trail)

        carry_init = (0, qp0_lead[0, :], qp0_trail[0, :])
        particle_ids = xp.arange(len(qp0_lead))
        lead_arm, trail_arm = jax.lax.scan(scan_fn, carry_init, particle_ids)[1]
        return (lead_arm, trail_arm), prog_o

    @partial_jit(static_argnames=("seed_num",))
    def _run_vmap(
        self, ts: jt.Array, prog_w0: jt.Array, prog_mass: jt.Array, *, seed_num: int
    ) -> tuple[tuple[jt.Array, jt.Array], Orbit]:
        """Generate stellar stream by vmapping over the release model/integration.

        Better for GPU usage.
        """
        # Integrate the progenitor orbit
        prog_o = self.potential.integrate_orbit(prog_w0, xp.min(ts), xp.max(ts), ts)

        # Generate stream initial conditions along the integrated progenitor orbit
        mock_lead, mock_trail = self.df.sample(
            self.potential, prog_o, prog_mass, seed_num=seed_num
        )
        qp0_lead = mock_lead.qp
        qp0_trail = mock_trail.qp
        t_f = ts[-1] + 0.01

        # TODO: make this a separated method
        @jax.jit  # type: ignore[misc]
        def single_particle_integrate(
            i: int, qp0_lead_i: jt.Array, qp0_trail_i: jt.Array
        ) -> tuple[jt.Array, jt.Array]:
            t_i = ts[i]
            qp_lead = self.integrate_orbit(qp0_lead_i, t_i, t_f, None).qp[0]
            qp_trail = self.integrate_orbit(qp0_trail_i, t_i, t_f, None).qp[0]
            return qp_lead, qp_trail

        particle_ids = xp.arange(len(qp0_lead))
        integrator = jax.vmap(single_particle_integrate, in_axes=(0, 0, 0))
        qp_lead, qp_trail = integrator(particle_ids, qp0_lead, qp0_trail)
        return (qp_lead, qp_trail), prog_o

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
        # TODO: figure out better return type: MockStream?
        if vmapped:
            return self._run_vmap(ts, prog_w0, prog_mass, seed_num=seed_num)
        return self._run_scan(ts, prog_w0, prog_mass, seed_num=seed_num)

    # ==========================================================================
