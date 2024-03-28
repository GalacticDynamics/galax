"""galax: Galactic Dynamix in Jax."""

__all__ = ["MockStreamGenerator"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import TypeAlias, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import quax.examples.prng as jr
from jax.lib.xla_bridge import get_backend

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity

import galax.typing as gt
from .core import MockStream
from .df import AbstractStreamDF, ProgenitorMassCallable
from .utils import cond_reverse, interleave_concat
from galax.coordinates import PhaseSpacePosition
from galax.dynamics._dynamics.integrate._api import Integrator
from galax.dynamics._dynamics.integrate._builtin import DiffraxIntegrator
from galax.dynamics._dynamics.integrate._funcs import evaluate_orbit
from galax.potential._potential.base import AbstractPotentialBase

Carry: TypeAlias = tuple[gt.IntScalar, gt.VecN, gt.VecN]


@final
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

    @property
    def units(self) -> AbstractUnitSystem:
        """Units of the potential."""
        return cast(AbstractUnitSystem, self.potential.units)

    # ==========================================================================

    @partial(jax.jit)
    def _run_scan(  # TODO: output shape depends on the input shape
        self, ts: gt.QVecTime, mock0_lead: MockStream, mock0_trail: MockStream
    ) -> tuple[gt.BatchVec6, gt.BatchVec6]:
        """Generate stellar stream by scanning over the release model/integration.

        Better for CPU usage.
        """
        w0_lead = mock0_lead.w(units=self.units)
        w0_trail = mock0_trail.w(units=self.units)
        t_f = ts[-1] + Quantity(1e-3, ts.unit)  # TODO: not bump in the final time.

        def one_pt_intg(
            carry: Carry, _: gt.IntScalar
        ) -> tuple[Carry, tuple[gt.VecN, gt.VecN]]:
            """Integrate one point along the stream.

            Parameters
            ----------
            carry : tuple[int, VecN, VecN]
                Initial state of the particle at index `idx`.
            idx : int
                Index of the current point.
            """
            i, w0_l_i, w0_t_i = carry
            tstep = xp.asarray([ts[i], t_f])

            def integ_ics(ics: gt.Vec6) -> gt.VecN:
                # TODO: only return the final state
                return evaluate_orbit(
                    self.potential, ics, tstep, integrator=self.stream_integrator
                ).w(units=self.units)[-1]

            # vmap integration over leading and trailing arm
            w0_lt_i = jnp.vstack([w0_l_i, w0_t_i])  # TODO: xp.stack
            w_l, w_t = jax.vmap(integ_ics, in_axes=(0,))(w0_lt_i)
            # Prepare for next iteration
            carry_out = (i + 1, w0_lead[i + 1, :], w0_trail[i + 1, :])
            return carry_out, (w_l, w_t)

        carry_init = (0, w0_lead[0, :], w0_trail[0, :])
        pt_ids = xp.arange(len(w0_lead))
        lead_arm_w, trail_arm_w = jax.lax.scan(one_pt_intg, carry_init, pt_ids)[1]

        return lead_arm_w, trail_arm_w

    @partial(jax.jit)
    def _run_vmap(  # TODO: output shape depends on the input shape
        self, ts: gt.QVecTime, mock0_lead: MockStream, mock0_trail: MockStream
    ) -> tuple[gt.BatchVec6, gt.BatchVec6]:
        """Generate stellar stream by vmapping over the release model/integration.

        Better for GPU usage.
        """
        t_f = ts[-1] + Quantity(1e-3, ts.unit)  # TODO: not bump in the final time.

        @partial(jax.jit, inline=True)
        def one_pt_intg(
            i: gt.IntScalar, w0_l_i: gt.Vec6, w0_t_i: gt.Vec6
        ) -> tuple[gt.Vec6, gt.Vec6]:
            tstep = xp.asarray([ts[i], t_f])
            w_lead = evaluate_orbit(
                self.potential, w0_l_i, tstep, integrator=self.stream_integrator
            ).w(units=self.potential.units)[-1]
            w_trail = evaluate_orbit(
                self.potential, w0_t_i, tstep, integrator=self.stream_integrator
            ).w(units=self.potential.units)[-1]
            return w_lead, w_trail

        w0_lead = mock0_lead.w(units=self.units)
        w0_trail = mock0_trail.w(units=self.units)
        pt_ids = xp.arange(len(w0_lead))
        lead_arm_w, trail_arm_w = jax.vmap(one_pt_intg)(pt_ids, w0_lead, w0_trail)
        return lead_arm_w, trail_arm_w

    @partial(jax.jit, static_argnames=("vmapped",))
    def run(
        self,
        rng: jr.PRNG,
        ts: gt.QVecTime,
        prog_w0: PhaseSpacePosition | gt.Vec6,
        prog_mass: gt.FloatQScalar | ProgenitorMassCallable,
        *,
        vmapped: bool | None = None,
    ) -> tuple[MockStream, PhaseSpacePosition]:
        """Generate mock stellar stream.

        Parameters
        ----------
        rng : :class:`quax.examples.prng.PRNG`
            Random number generator.
        ts : Quantity[float, (time,), "time"]
            Stripping times.
        prog_w0 : PhaseSpacePosition[float, ()]
            Initial conditions of the progenitor.
        prog_mass : Quantity[float, (), "mass"]
            Mass of the progenitor.

        vmapped : bool | None, optional keyword-only
            Whether to use `jax.vmap` (`True`) or `jax.lax.scan` (`False`) to
            parallelize the integration. ``vmapped=True`` is recommended for GPU
            usage, while ``vmapped=False`` is recommended for CPU usage.  If
            `None` (default), then `jax.vmap` is used on GPU and `jax.lax.scan`
            otherwise.

        Returns
        -------
        mockstream : :class:`galax.dynamcis.MockStream`
            Leading and/or trailing arms of the mock stream.
        prog_o : :class:`galax.coordinates.PhaseSpacePosition`
            The final phase-space(+time) position of the progenitor.
        """
        # TODO: ꜛ a discussion about the stripping times
        # Parse vmapped
        use_vmap = get_backend().platform == "gpu" if vmapped is None else vmapped

        # Ensure w0 is a PhaseSpacePosition
        w0: PhaseSpacePosition
        if isinstance(prog_w0, PhaseSpacePosition):
            w0 = prog_w0 if prog_w0.t is not None else replace(prog_w0, t=ts[0])
        else:
            w0 = PhaseSpacePosition(
                q=Quantity(prog_w0[0:3], self.units["length"]),
                p=Quantity(prog_w0[3:6], self.units["speed"]),
                t=ts[0].to(self.potential.units["time"]),
            )
        w0 = eqx.error_if(w0, w0.ndim > 0, "prog_w0 must be scalar")

        # If the time stepping passed in is negative, assume this means that all
        # of the initial conditions are at *end time*, and we need to reverse
        # them before treating them as initial conditions
        ts = cond_reverse(ts[1] < ts[0], ts)

        # Integrate the progenitor orbit, evaluating at the stripping times
        prog_o = evaluate_orbit(
            self.potential, w0, ts, integrator=self.progenitor_integrator
        )

        # Generate initial conditions from the DF, along the integrated
        # progenitor orbit. The release times are the stripping times.
        mock0_lead, mock0_trail = self.df.sample(rng, self.potential, prog_o, prog_mass)

        if use_vmap:
            lead_arm_w, trail_arm_w = self._run_vmap(ts, mock0_lead, mock0_trail)
        else:
            lead_arm_w, trail_arm_w = self._run_scan(ts, mock0_lead, mock0_trail)

        t = xp.ones_like(ts) * ts.value[-1]  # TODO: ensure this time is correct

        # TODO: move the leading vs trailing logic to the DF
        if self.df.lead and self.df.trail:
            axis = len(trail_arm_w.shape) - 2
            q = interleave_concat(trail_arm_w[:, 0:3], lead_arm_w[:, 0:3], axis=axis)
            p = interleave_concat(trail_arm_w[:, 3:6], lead_arm_w[:, 3:6], axis=axis)
            t = interleave_concat(t, t, axis=0)
            release_time = interleave_concat(
                mock0_lead.release_time, mock0_trail.release_time, axis=0
            )
        elif self.df.lead:
            q = lead_arm_w[:, 0:3]
            p = lead_arm_w[:, 3:6]
            release_time = mock0_lead.release_time
        elif self.df.trail:
            q = trail_arm_w[:, 0:3]
            p = trail_arm_w[:, 3:6]
            release_time = mock0_trail.release_time
        else:
            msg = "You must generate either leading or trailing tails (or both!)"
            raise ValueError(msg)

        mockstream = MockStream(
            q=Quantity(q, self.units["length"]),
            p=Quantity(p, self.units["speed"]),
            t=t,
            release_time=release_time,
        )

        return mockstream, prog_o[-1]
