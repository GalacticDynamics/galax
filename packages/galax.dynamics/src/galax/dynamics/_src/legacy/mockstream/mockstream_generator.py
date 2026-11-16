"""Generator for mock streams."""

__all__ = ["MockStreamGenerator"]

import functools as ft
import warnings
from dataclasses import KW_ONLY

from jaxtyping import PRNGKeyArray
from typing import TypeAlias, cast, final

import equinox as eqx
import jax
import jax.extend as jex

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import Quantity as FastQ

import galax.coordinates as gc
import galax.dynamics.custom_types as gt
import galax.potential as gp
from .df import AbstractStreamDF, ChenStreamDF, ProgenitorMassCallable
from galax.dynamics._src.legacy.funcs import default_integrator, evaluate_orbit
from galax.dynamics._src.legacy.integrator import Integrator
from galax.dynamics._src.mockstream.arm import MockStreamArm
from galax.dynamics._src.mockstream.core import MockStream
from galax.dynamics._src.orbit import HamiltonianField, Orbit
from galax.potential import AbstractPotential
from galax.potential.utils import cond_reverse

Carry: TypeAlias = tuple[gt.IntSz0, gt.SzN, gt.SzN]

# Knots of the progenitor orbit that the stream particles see, per release
# time and at least: ~0.5 Myr apart for 1000 releases over 5 Gyr, and <= 0.9
# Myr for any ``ts`` spanning <= 14 Gyr.
KNOTS_PER_RELEASE = 20
MIN_KNOTS = 2**14


@final
class _HermitePath(eqx.Module):
    """Cubic Hermite interpolant of (x, v) sampled on a uniform time grid.

    The error is O(h^4); outside the grid it extrapolates the end cubics.
    """

    t0: gt.Sz0
    dt: gt.Sz0
    x: gt.SzN  # (n, 3)
    v: gt.SzN  # (n, 3)

    def __call__(self, t: gt.Sz0) -> gt.Sz3:
        s = (t - self.t0) / self.dt
        i = jnp.clip(jnp.floor(s).astype(int), 0, self.x.shape[0] - 2)
        a = s - i
        b = 1 - a
        x: gt.Sz3 = (
            (b * b * (1 + 2 * a)) * self.x[i]
            + (a * b * b * self.dt) * self.v[i]
            + (a * a * (3 - 2 * a)) * self.x[i + 1]
            - (a * a * b * self.dt) * self.v[i + 1]
        )
        return x


@final
class MockStreamGenerator(eqx.Module):
    """Generate a mock stellar stream in the specified external potential."""

    df: AbstractStreamDF
    """Distribution function for generating mock streams.

    E.g. ``galax.dynamics.mockstream.FardalStreamDF``.
    """

    potential: AbstractPotential
    """Potential in which the progenitor orbits and creates a stream."""

    _: KW_ONLY
    progenitor_potential: AbstractPotential | None = None
    """Potential of the progenitor, centred on the origin, or `None`.

    If given, the stream particles move in `potential` plus this potential
    centred on the progenitor as it orbits (the progenitor itself orbits in
    `potential` alone). This is the default model of Chen et al. 2024
    (arXiv:2408.01496), with a `~galax.potential.KeplerPotential` of the
    progenitor's mass, and gala's ``progenitor_potential``. Its mass is its own:
    the ``prog_mass`` given to `run` only sets the release distribution.

    The particles see the progenitor's orbit as a cubic Hermite interpolant of
    its position and velocity on ``max(20 len(ts), 2**14)`` uniformly spaced
    times.
    """

    progenitor_integrator: Integrator = eqx.field(
        default=default_integrator, static=True
    )
    """Integrator for the progenitor orbit."""

    stream_integrator: Integrator = eqx.field(default=default_integrator, static=True)
    """Integrator for the stream."""

    @property
    def units(self) -> u.AbstractUnitSystem:
        """Units of the potential."""
        return cast("u.AbstractUnitSystem", self.potential.units)

    # ==========================================================================

    def _progenitor_trajectory(
        self, w0: gc.AbstractPhaseSpaceObject, ts: gt.QuSzTime
    ) -> Orbit:
        """Integrate the progenitor orbit."""
        return cast(
            "Orbit",
            evaluate_orbit(
                self.potential, w0, ts, integrator=self.progenitor_integrator
            ),
        )

    def _stream_potential(
        self, w0: gc.PhaseSpaceCoordinate, ts: gt.QuSzTime
    ) -> AbstractPotential:
        """Return the potential the stream particles move in."""
        if self.progenitor_potential is None:
            return self.potential
        # ponytail: with a fixed-step progenitor integrator, knots between its
        # steps are its linear dense output, O(dt^2) off its own orbit (far
        # below the particles' own step error); align knots with steps if needed.
        n = max(KNOTS_PER_RELEASE * len(ts), MIN_KNOTS)
        tk = jnp.linspace(ts[0], ts[-1], n)
        wk = self._progenitor_trajectory(w0, tk).w(units=self.units)
        tk = u.ustrip(self.units["time"], tk)
        path = _HermitePath(tk[0], tk[1] - tk[0], wk[:, :3], wk[:, 3:])
        moving = gp.TranslatedPotential(
            self.progenitor_potential,
            translation=gp.params.TimeDependentTranslationParameter(
                path, units=self.units
            ),
        )
        return gp.CompositePotential(host=self.potential, progenitor=moving)

    def _integrate_to(
        self,
        w0: gt.BtSz6,
        t0: gt.QuSz0,
        t1: gt.QuSz0,
        potential: AbstractPotential | None = None,
    ) -> tuple[gt.Sz6, gt.Sz6]:
        """Integrate a leading and a trailing particle ``(2, 6)`` from t0 to t1.

        Each gets its own solve, so an adaptive solver picks its steps per particle.
        ``potential`` defaults to the host, ``self.potential``.
        """
        pot = self.potential if potential is None else potential
        # `units` is an `eqx.AbstractVar` on the base class, which mypy reads as
        # an init field; on `HamiltonianField` it is a property.
        field = HamiltonianField(pot)  # type: ignore[call-arg]
        # `SaveAt(t1=True)` is ~2.4x slower here on the adaptive scan path (an
        # XLA whole-program effect), so save at the time `t1` instead.
        saveat = jnp.asarray([t1])

        def final_state(w: gt.Sz6) -> gt.Sz6:
            out = self.stream_integrator(field, w, t0, t1, saveat=saveat)
            w1: gt.Sz6 = out.w(units=self.units)[-1]
            return w1

        w = jax.vmap(final_state)(w0)
        return w[0], w[1]

    # ==========================================================================

    @ft.partial(jax.jit)
    def _run_scan(  # TODO: output shape depends on the input shape
        self,
        ts: gt.QuSzTime,
        mock0_lead: MockStreamArm,
        mock0_trail: MockStreamArm,
        potential: AbstractPotential,
    ) -> tuple[gt.BtSz6, gt.BtSz6]:
        """Generate stellar stream by scanning over the release model/integration.

        Better for CPU usage.
        """
        w0_lead = mock0_lead.w(units=self.units)
        w0_trail = mock0_trail.w(units=self.units)
        t_f = ts[-1] + u.Q(1e-3, "Myr")  # TODO: not bump in the final time.

        def one_pt_intg(
            carry: Carry, _: gt.IntSz0
        ) -> tuple[Carry, tuple[gt.SzN, gt.SzN]]:
            """Integrate one point along the stream.

            Parameters
            ----------
            carry : tuple[int, SzN, SzN]
                Initial state of the particle at index `idx`.
            idx : int
                Index of the current point.
            """
            i, w0_l_i, w0_t_i = carry
            w_l, w_t = self._integrate_to(
                jnp.vstack([w0_l_i, w0_t_i]), ts[i], t_f, potential
            )
            # Prepare for next iteration
            carry_out = (i + 1, w0_lead[i + 1, :], w0_trail[i + 1, :])
            return carry_out, (w_l, w_t)

        carry_init = (0, w0_lead[0, :], w0_trail[0, :])
        pt_ids = jnp.arange(len(w0_lead))
        lead_arm_w, trail_arm_w = jax.lax.scan(
            one_pt_intg,  # type: ignore[arg-type]
            carry_init,
            pt_ids,
        )[1]

        return lead_arm_w, trail_arm_w

    @ft.partial(jax.jit)
    def _run_vmap(  # TODO: output shape depends on the input shape
        self,
        ts: gt.QuSzTime,
        mock0_lead: MockStreamArm,
        mock0_trail: MockStreamArm,
        potential: AbstractPotential,
    ) -> tuple[gt.BtSz6, gt.BtSz6]:
        """Generate stellar stream by vmapping over the release model/integration.

        Better for GPU usage.
        """
        t_f = ts[-1] + u.Q(1e-3, "Myr")  # TODO: not bump in the final time.

        @ft.partial(jax.jit, inline=True)
        def one_pt_intg(
            i: gt.IntSz0, w0_l_i: gt.Sz6, w0_t_i: gt.Sz6
        ) -> tuple[gt.Sz6, gt.Sz6]:
            return self._integrate_to(
                jnp.vstack([w0_l_i, w0_t_i]), ts[i], t_f, potential
            )

        w0_lead = mock0_lead.w(units=self.units)
        w0_trail = mock0_trail.w(units=self.units)
        pt_ids = jnp.arange(len(w0_lead))
        lead_arm_w, trail_arm_w = jax.vmap(one_pt_intg)(pt_ids, w0_lead, w0_trail)
        return lead_arm_w, trail_arm_w

    @ft.partial(jax.jit, static_argnames=("vmapped",))
    def run(
        self,
        rng: PRNGKeyArray,
        ts: gt.QuSzTime,
        prog_w0: gc.AbstractPhaseSpaceObject | gt.Sz6,
        prog_mass: gt.FloatQuSz0 | ProgenitorMassCallable,
        *,
        vmapped: bool | None = None,
    ) -> tuple[MockStream, gc.PhaseSpaceCoordinate]:
        """Generate mock stellar stream.

        Parameters
        ----------
        rng : :class:`jaxtyping.PRNGKeyArray`
            Random number generator.
        ts : Quantity[float, (time,), "time"]
            Stripping times.
        prog_w0 : PhaseSpaceCoordinate[float, ()]
            Initial conditions of the progenitor.

            The recommended way to pass in the progenitor's initial conditions
            is as a :class:`~galax.coordinates.PhaseSpaceCoordinate` object with
            a set time. This is the most explicit and is guaranteed to have the
            correct units and no surprises about the progenitor.

            .. note::

                If the time is not set, it is assumed to be the first stripping
                time.

            Alternatively, you can pass in a 6-element array of the Cartesian
            phase-space coordinates (x, y, z, vx, vy, vz) in the same units as
            the potential.

        prog_mass : Quantity[float, (), "mass"] | `ProgenitorMassCallable`
            Mass of the progenitor. May be a Quantity or a callable that returns
            the progenitor mass at the given times.

        vmapped : bool | None, optional keyword-only
            Whether to use `jax.vmap` (`True`) or `jax.lax.scan` (`False`) to
            vectorize the integration. ``vmapped=True`` is recommended for GPU
            usage, while ``vmapped=False`` is recommended for CPU usage.  If
            `None` (default), then `jax.vmap` is used on GPU and `jax.lax.scan`
            otherwise.

        Returns
        -------
        mockstream : :class:`galax.dynamcis.MockStreamArm`
            Leading and/or trailing arms of the mock stream.
        prog_o : :class:`galax.coordinates.PhaseSpaceCoordinate`
            The final phase-space(+time) position of the progenitor.
        """
        if self.progenitor_potential is None and isinstance(self.df, ChenStreamDF):
            warnings.warn(
                "ChenStreamDF without a progenitor_potential is the "
                '"no progenitor" variant of the Chen+24 model; pass e.g. '
                "progenitor_potential=KeplerPotential(m_tot=prog_mass) for its "
                "default model.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Parse vmapped
        use_vmap = (
            jex.backend.get_backend().platform == "gpu" if vmapped is None else vmapped
        )

        # Ensure w0 is a PhaseSpacePosition
        w0: gc.PhaseSpaceCoordinate
        if isinstance(prog_w0, gc.PhaseSpaceCoordinate):
            w0 = prog_w0
        elif isinstance(prog_w0, gc.PhaseSpacePosition):
            w0 = gc.PhaseSpaceCoordinate(q=prog_w0.q, p=prog_w0.p, t=ts[0])
        else:
            w0 = gc.PhaseSpaceCoordinate(
                q=u.Q(prog_w0[0:3], self.units["length"]),
                p=u.Q(prog_w0[3:6], self.units["speed"]),
                t=u.uconvert(self.potential.units["time"], ts[0]),
            )
        w0 = eqx.error_if(w0, w0.ndim > 0, "prog_w0 must be scalar")
        # TODO: allow for multiple progenitors

        # If the time stepping passed in is negative, assume this means that all
        # of the initial conditions are at *end time*, and we need to reverse
        # them before treating them as initial conditions
        ts = cond_reverse(ts[1] < ts[0], ts)

        # Integrate the progenitor orbit, evaluating at the stripping times
        prog_o = self._progenitor_trajectory(w0, ts)

        # TODO: here sep out lead vs trailing
        # Generate initial conditions from the DF, along the integrated
        # progenitor orbit. The release times are the stripping times.
        mock0 = self.df.sample(rng, self.potential, prog_o, prog_mass)

        pot = self._stream_potential(w0, ts)
        run = self._run_vmap if use_vmap else self._run_scan
        lead_arm_w, trail_arm_w = run(ts, mock0["lead"], mock0["trail"], pot)

        t = jnp.ones_like(ts) * ts.value[-1]  # TODO: ensure this time is correct

        frame = (
            prog_w0.frame
            if isinstance(prog_w0, gc.AbstractPhaseSpaceObject)
            else gc.frames.simulation_frame
        )

        comps = {}
        comps["lead"] = MockStreamArm(
            q=FastQ(lead_arm_w[:, 0:3], self.units["length"]),
            p=FastQ(lead_arm_w[:, 3:6], self.units["speed"]),
            t=t,
            release_time=mock0["lead"].release_time,
            frame=frame,
        )
        comps["trail"] = MockStreamArm(
            q=FastQ(trail_arm_w[:, 0:3], self.units["length"]),
            p=FastQ(trail_arm_w[:, 3:6], self.units["speed"]),
            t=t,
            release_time=mock0["trail"].release_time,
            frame=frame,
        )

        return MockStream(comps), prog_o[-1]
