"""galax: Galactic Dynamix in Jax."""

__all__ = ["MockStreamGenerator"]

from dataclasses import KW_ONLY
from functools import partial
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.experimental.array_api as xp
import jax.numpy as jnp
from jax.lib.xla_bridge import get_backend

from galax.dynamics._orbit import Orbit
from galax.integrate._base import Integrator
from galax.integrate._builtin import DiffraxIntegrator
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import (
    BatchVec6,
    FloatScalar,
    IntScalar,
    TimeVector,
    Vec6,
    VecN,
)
from galax.utils._collections import ImmutableDict

from ._core import MockStream
from ._df import AbstractStreamDF

Carry: TypeAlias = tuple[IntScalar, VecN, VecN]


def _converter_immutabledict_or_none(x: Any) -> ImmutableDict[Any] | None:
    return None if x is None else ImmutableDict(x)


class MockStreamGenerator(eqx.Module):  # type: ignore[misc]
    """Generate a mock stellar stream in the specified external potential."""

    df: AbstractStreamDF
    """Distribution function for generating mock streams.

    E.g. ``galax.dynamics.mockstream.FardalStreamDF``.
    """

    potential: AbstractPotentialBase
    """Potential in which the progenitor orbits and creates a stream."""

    _: KW_ONLY
    progenitor_integrator: Integrator = eqx.field(
        default=DiffraxIntegrator(), static=True
    )
    """Integrator for the progenitor orbit."""

    stream_integrator: Integrator = eqx.field(default=DiffraxIntegrator(), static=True)
    """Integrator for the stream."""

    # ==========================================================================

    @partial(jax.jit)
    def _run_scan(  # TODO: output shape depends on the input shape
        self, ts: TimeVector, mock0_lead: MockStream, mock0_trail: MockStream
    ) -> tuple[BatchVec6, BatchVec6]:
        """Generate stellar stream by scanning over the release model/integration.

        Better for CPU usage.
        """
        qp0_lead = mock0_lead.qp
        qp0_trail = mock0_trail.qp

        def scan_fn(carry: Carry, _: IntScalar) -> tuple[Carry, tuple[VecN, VecN]]:
            i, qp0_lead_i, qp0_trail_i = carry
            qp0_lead_trail = jnp.vstack([qp0_lead_i, qp0_trail_i])  # TODO: xp.stack
            tstep = xp.asarray([ts[i], ts[-1]])

            def integ_ics(ics: Vec6) -> VecN:
                # TODO: only return the final state
                return self.potential.integrate_orbit(
                    ics, tstep, integrator=self.stream_integrator
                ).qp[-1]

            # vmap over leading and trailing arm
            qp_lead, qp_trail = jax.vmap(integ_ics, in_axes=(0,))(qp0_lead_trail)
            carry_out = (i + 1, qp0_lead[i + 1, :], qp0_trail[i + 1, :])
            return carry_out, (qp_lead, qp_trail)

        carry_init = (0, qp0_lead[0, :], qp0_trail[0, :])
        particle_ids = xp.arange(len(qp0_lead))
        lead_arm_qp, trail_arm_qp = jax.lax.scan(scan_fn, carry_init, particle_ids)[1]

        return lead_arm_qp, trail_arm_qp

    @partial(jax.jit)
    def _run_vmap(  # TODO: output shape depends on the input shape
        self, ts: TimeVector, mock0_lead: MockStream, mock0_trail: MockStream
    ) -> tuple[BatchVec6, BatchVec6]:
        """Generate stellar stream by vmapping over the release model/integration.

        Better for GPU usage.
        """
        t_f = ts[-1] + 0.01  # TODO: not have the bump in the final time.

        # TODO: make this a separated method
        @jax.jit  # type: ignore[misc]
        def single_particle_integrate(
            i: IntScalar, qp0_lead_i: Vec6, qp0_trail_i: Vec6
        ) -> tuple[Vec6, Vec6]:
            tstep = xp.asarray([ts[i], t_f])
            # TODO: only return the final state
            qp_lead = self.potential.integrate_orbit(
                qp0_lead_i, tstep, integrator=self.stream_integrator
            ).qp[-1]
            qp_trail = self.potential.integrate_orbit(
                qp0_trail_i, tstep, integrator=self.stream_integrator
            ).qp[-1]
            return qp_lead, qp_trail

        qp0_lead = mock0_lead.qp
        particle_ids = xp.arange(len(qp0_lead))
        integrator = jax.vmap(single_particle_integrate, in_axes=(0, 0, 0))
        lead_arm_qp, trail_arm_qp = integrator(particle_ids, qp0_lead, mock0_trail.qp)
        return lead_arm_qp, trail_arm_qp

    @partial(jax.jit, static_argnames=("seed_num", "vmapped"))
    def run(
        self,
        ts: TimeVector,
        prog_w0: Vec6,
        prog_mass: FloatScalar,
        *,
        seed_num: int,
        vmapped: bool | None = None,
    ) -> tuple[tuple[MockStream, MockStream], Orbit]:
        """Generate mock stellar stream.

        Parameters
        ----------
        ts : Array[float, (time,)]
            Stripping times.
        prog_w0 : Array[float, (6,)]
            Initial conditions of the progenitor.
        prog_mass : float
            Mass of the progenitor.

        seed_num : int, keyword-only
            Seed number for the random number generator.

            :todo: a better way to handle PRNG

        vmapped : bool | None, optional keyword-only
            Whether to use `jax.vmap` (`True`) or `jax.lax.scan` (`False`) to
            parallelize the integration. ``vmapped=True`` is recommended for GPU
            usage, while ``vmapped=False`` is recommended for CPU usage.  If
            `None` (default), then `jax.vmap` is used on GPU and `jax.lax.scan`
            otherwise.

        Returns
        -------
        lead_arm, trail_arm : tuple[MockStream, MockStream]
            Leading and trailing arms of the mock stream.
        prog_o : Orbit
            Orbit of the progenitor.
        """
        # TODO: ꜛ a discussion about the stripping times
        # Parse vmapped
        use_vmap = get_backend().platform == "gpu" if vmapped is None else vmapped

        # Integrate the progenitor orbit, evaluating at the stripping times
        prog_o = self.potential.integrate_orbit(
            prog_w0, ts, integrator=self.progenitor_integrator
        )

        # Generate stream initial conditions along the integrated progenitor
        # orbit. The release times are stripping times.
        mock0_lead, mock0_trail = self.df.sample(
            self.potential, prog_o, prog_mass, seed_num=seed_num
        )

        if use_vmap:
            lead_arm_qp, trail_arm_qp = self._run_vmap(ts, mock0_lead, mock0_trail)
        else:
            lead_arm_qp, trail_arm_qp = self._run_scan(ts, mock0_lead, mock0_trail)

        lead_arm = MockStream(
            q=lead_arm_qp[:, 0:3],
            p=lead_arm_qp[:, 3:6],
            release_time=mock0_lead.release_time,
        )
        trail_arm = MockStream(
            q=trail_arm_qp[:, 0:3],
            p=trail_arm_qp[:, 3:6],
            release_time=mock0_trail.release_time,
        )

        return (lead_arm, trail_arm), prog_o
