"""galdynamix: Galactic Dynamix in Jax."""

__all__ = ["MockStreamGenerator"]

from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as xp
from jaxtyping import Float

from galdynamix.dynamics._orbit import Orbit
from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.typing import (
    FloatScalar,
    IntegerScalar,
    TimeVector,
    Vector6,
    VectorN,
)
from galdynamix.utils import partial_jit

from ._df import AbstractStreamDF

Carry: TypeAlias = tuple[IntegerScalar, VectorN, VectorN]


class MockStreamGenerator(eqx.Module):  # type: ignore[misc]
    df: AbstractStreamDF
    potential: AbstractPotentialBase

    # ==========================================================================

    @partial_jit(static_argnames=("seed_num",))
    def _run_scan(
        self,
        ts: TimeVector,
        prog_w0: Vector6,
        prog_mass: FloatScalar,
        *,
        seed_num: int,
    ) -> tuple[tuple[Float[Vector6, "time"], Float[Vector6, "time"]], Orbit]:
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

        def scan_fn(
            carry: Carry, idx: IntegerScalar
        ) -> tuple[Carry, tuple[VectorN, VectorN]]:
            i, qp0_lead_i, qp0_trail_i = carry
            qp0_lead_trail = xp.vstack([qp0_lead_i, qp0_trail_i])
            t_i, t_f = ts[i], ts[-1]

            def integ_ics(ics: Vector6) -> VectorN:
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
        self, ts: TimeVector, prog_w0: Vector6, prog_mass: FloatScalar, *, seed_num: int
    ) -> tuple[tuple[Float[Vector6, "time"], Float[Vector6, "time"]], Orbit]:
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
            i: int, qp0_lead_i: Vector6, qp0_trail_i: Vector6
        ) -> tuple[Vector6, Vector6]:
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
        ts: TimeVector,
        prog_w0: Vector6,
        prog_mass: FloatScalar,
        *,
        seed_num: int,
        vmapped: bool = False,
    ) -> tuple[tuple[Float[Vector6, "time"], Float[Vector6, "time"]], Orbit]:
        # TODO: figure out better return type: MockStream?
        if vmapped:
            return self._run_vmap(ts, prog_w0, prog_mass, seed_num=seed_num)
        return self._run_scan(ts, prog_w0, prog_mass, seed_num=seed_num)

    # ==========================================================================
