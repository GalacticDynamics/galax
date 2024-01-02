"""galax: Galactic Dynamix in Jax."""

__all__ = ["MockStreamGenerator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as xp
from jaxtyping import Float

from galax.dynamics._orbit import Orbit
from galax.integrate._base import AbstractIntegrator
from galax.integrate._builtin import DiffraxIntegrator
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import (
    FloatScalar,
    IntScalar,
    TimeVector,
    Vec6,
    VecN,
)
from galax.utils import partial_jit
from galax.utils._collections import ImmutableDict

from ._df import AbstractStreamDF

Carry: TypeAlias = tuple[IntScalar, VecN, VecN]


def _converter_immutabledict_or_none(x: Any) -> ImmutableDict[Any] | None:
    return None if x is None else ImmutableDict(x)


class MockStreamGenerator(eqx.Module):  # type: ignore[misc]
    df: AbstractStreamDF
    """Distribution function for generating mock streams.

    E.g. ``galax.dynamics.mockstream.FardalStreamDF``.
    """

    potential: AbstractPotentialBase
    """Potential in which the progenitor orbits and creates a stream."""

    _: KW_ONLY
    progenitor_integrator: AbstractIntegrator = eqx.field(
        default=DiffraxIntegrator(), static=True
    )
    """Integrator for the progenitor orbit."""

    stream_integrator: AbstractIntegrator = eqx.field(
        default=DiffraxIntegrator(), static=True
    )

    stream_integrator_kw: Mapping[str, Any] | None = eqx.field(
        default=None, static=True, converter=_converter_immutabledict_or_none
    )
    """Keyword arguments for the stream integrator."""

    # ==========================================================================

    @partial_jit(static_argnames=("seed_num",))
    def _run_scan(
        self, ts: TimeVector, prog_w0: Vec6, prog_mass: FloatScalar, *, seed_num: int
    ) -> tuple[tuple[Float[Vec6, "time"], Float[Vec6, "time"]], Orbit]:
        """Generate stellar stream by scanning over the release model/integration.

        Better for CPU usage.
        """
        # Integrate the progenitor orbit
        prog_o = self.potential.integrate_orbit(
            prog_w0, xp.min(ts), xp.max(ts), ts, integrator=self.progenitor_integrator
        )

        # Generate stream initial conditions along the integrated progenitor orbit
        mock0_lead, mock0_trail = self.df.sample(
            self.potential, prog_o, prog_mass, seed_num=seed_num
        )
        qp0_lead = mock0_lead.qp
        qp0_trail = mock0_trail.qp

        def scan_fn(carry: Carry, idx: IntScalar) -> tuple[Carry, tuple[VecN, VecN]]:
            i, qp0_lead_i, qp0_trail_i = carry
            qp0_lead_trail = xp.vstack([qp0_lead_i, qp0_trail_i])
            t_i, t_f = ts[i], ts[-1]

            def integ_ics(ics: Vec6) -> VecN:
                return self.potential.integrate_orbit(
                    ics, t_i, t_f, None, integrator=self.stream_integrator
                ).qp[0]

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
        self, ts: TimeVector, prog_w0: Vec6, prog_mass: FloatScalar, *, seed_num: int
    ) -> tuple[tuple[Float[Vec6, "time"], Float[Vec6, "time"]], Orbit]:
        """Generate stellar stream by vmapping over the release model/integration.

        Better for GPU usage.
        """
        # Integrate the progenitor orbit
        prog_o = self.potential.integrate_orbit(
            prog_w0, xp.min(ts), xp.max(ts), ts, integrator=self.progenitor_integrator
        )

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
            i: int, qp0_lead_i: Vec6, qp0_trail_i: Vec6
        ) -> tuple[Vec6, Vec6]:
            t_i = ts[i]
            qp_lead = self.potential.integrate_orbit(
                qp0_lead_i, t_i, t_f, None, integrator=self.stream_integrator
            ).qp[0]
            qp_trail = self.potential.integrate_orbit(
                qp0_trail_i, t_i, t_f, None, integrator=self.stream_integrator
            ).qp[0]
            return qp_lead, qp_trail

        particle_ids = xp.arange(len(qp0_lead))
        integrator = jax.vmap(single_particle_integrate, in_axes=(0, 0, 0))
        qp_lead, qp_trail = integrator(particle_ids, qp0_lead, qp0_trail)
        return (qp_lead, qp_trail), prog_o

    @partial_jit(static_argnames=("seed_num", "vmapped"))
    def run(
        self,
        ts: TimeVector,
        prog_w0: Vec6,
        prog_mass: FloatScalar,
        *,
        seed_num: int,
        vmapped: bool = False,
    ) -> tuple[tuple[Float[Vec6, "time"], Float[Vec6, "time"]], Orbit]:
        # TODO: figure out better return type: MockStream?
        if vmapped:
            return self._run_vmap(ts, prog_w0, prog_mass, seed_num=seed_num)
        return self._run_scan(ts, prog_w0, prog_mass, seed_num=seed_num)

    # ==========================================================================
