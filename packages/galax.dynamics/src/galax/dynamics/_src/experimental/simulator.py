"""The spray-stream simulator: run modes, result and `SpraySimulator`."""

__all__ = [
    "Copies",
    "PerParticle",
    "SharedTrack",
    "SprayResult",
    "SpraySimulator",
    "WholeStream",
    "to_mockstream",
]

from dataclasses import KW_ONLY

from collections.abc import Callable
from jaxtyping import Array, PRNGKeyArray
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxmore.loop_strategies as lstrat
from plum import dispatch

import diffraxtra as dfxtra
import unxt as u

import galax.coordinates as gc
import galax.fields as gf
import galax.potential as gp
from .df import AbstractKinematicDF, sample_releases
from .fields import _WholeStreamField
from .schedule import Uniform
from .solve import default_solver, progenitor_states, solve_on
from galax.dynamics._src.mockstream.arm import MockStreamArm
from galax.dynamics._src.mockstream.core import MockStream

ProgenitorPotential: TypeAlias = (
    gp.AbstractPotential | Callable[[Array], gp.AbstractPotential] | None
)


class WholeStream(eqx.Module):
    """Integrate the whole stream and the progenitor as one system.

    The state holds the progenitor and every tracer. A tracer is frozen at its
    release state until its release time, then moves in the host plus the
    progenitor potential centred on the progenitor row. Each stretch between
    consecutive release times is its own solve, so a tracer's first step uses
    the force at its release state.

    With an adaptive controller, diffrax's default error norm is the RMS over
    the whole state, frozen tracers included. So one particle's error can reach
    about ``sqrt(N)`` times the tolerance for ``N`` rows. Pass a max norm to the
    controller (``diffrax.PIDController(..., norm=optimistix.max_norm)``) if the
    tolerance should hold per particle.
    """


class SharedTrack(eqx.Module):
    """Per-particle progenitor model: one shared track of the progenitor."""


class Copies(eqx.Module):
    """Per-particle progenitor model: each particle carries its own copy."""


class PerParticle(eqx.Module):
    """Integrate each particle with its own solve, from its release to ``t1``."""

    progenitor: SharedTrack | Copies
    loop: type[lstrat.Scan | lstrat.VMap] = eqx.field(default=lstrat.VMap, static=True)


class SprayResult(eqx.Module):
    """The stream at ``t1``, as raw arrays in the host potential's units.

    ``lead[i]`` and ``trail[i]`` were both released at ``release_times[i]``.
    """

    lead: Array  # (n_release, 6)
    trail: Array  # (n_release, 6)
    release_times: Array  # (n_release,)
    progenitor: Array  # (6,), the progenitor at t1
    t1: Array  # ()


class SpraySimulator(eqx.Module):
    """Spray a stellar stream from a progenitor's state today.

    Parameters
    ----------
    df : AbstractKinematicDF
        Release distribution, e.g. `Fardal2015DF` or `Chen2024DF`.
    mode : WholeStream | PerParticle
        How the particles are integrated.
    progenitor_potential : AbstractPotential | Callable | None
        The progenitor's potential, centred on it, felt by the particles. A
        callable is given ``Msat`` (as passed to the call, traced) and returns
        the potential. It must use the host's unit system.
    solver : diffraxtra.DiffEqSolver
        Settings of the stream's solves. A `diffrax.ConstantStepSize`
        controller steps on the schedule's lattice; any other is adaptive.
    progenitor_solver : diffraxtra.DiffEqSolver | None
        Settings of the progenitor's orbit; `None` means ``solver``.

    Calling it with ``(key, pot, w_today, Msat, schedule)`` returns a
    `SprayResult`: ``w_today`` is the progenitor's state at ``schedule.t1``,
    ``Msat`` a scalar or one mass per release. ``lead[i]`` and ``trail[i]``
    share ``release_times[i]``. All inputs and outputs are raw arrays in the
    host potential's units.
    """

    df: AbstractKinematicDF
    mode: WholeStream | PerParticle
    _: KW_ONLY
    progenitor_potential: ProgenitorPotential = None
    solver: dfxtra.DiffEqSolver = eqx.field(default_factory=default_solver)
    progenitor_solver: dfxtra.DiffEqSolver | None = None

    @eqx.filter_jit
    def __call__(
        self,
        key: PRNGKeyArray,
        pot: gp.AbstractPotential,
        w_today: Array,
        Msat: Array,
        schedule: Uniform,
        /,
    ) -> SprayResult:
        return simulate(self, key, pot, w_today, Msat, schedule)


def simulate(
    sim: SpraySimulator,
    key: PRNGKeyArray,
    pot: gp.AbstractPotential,
    w_today: Array,
    Msat: Array,
    schedule: Uniform,
    /,
) -> SprayResult:
    """Sample the releases on the progenitor's orbit and integrate them."""
    psolver = sim.solver if sim.progenitor_solver is None else sim.progenitor_solver
    ws = progenitor_states(pot, w_today, schedule, psolver)
    idx = schedule.release_index()
    ts = schedule.release_times()
    Ms = jnp.broadcast_to(jnp.asarray(Msat, dtype=float), ts.shape)
    xl, vl, xt, vt = sample_releases(sim.df, key, pot, ts, ws[idx, :3], ws[idx, 3:], Ms)
    pp = sim.progenitor_potential
    if pp is not None and not isinstance(pp, gp.AbstractPotential):
        pp = pp(Msat)
    lead, trail = _evolve(
        sim.mode,
        sim,
        pot,
        pp,
        schedule,
        ws,
        jnp.concat([xl, vl], axis=-1),
        jnp.concat([xt, vt], axis=-1),
    )
    return SprayResult(lead, trail, ts, ws[-1], jnp.asarray(schedule.t1, dtype=float))


def evolve_whole_stream(
    field: _WholeStreamField,
    solver: dfxtra.DiffEqSolver,
    w0: Array,
    grids: Array,
    active: Array,
) -> Array:
    """Evolve the rows ``w0`` (R, 6) over the segments ``grids`` (S, L + 1).

    Segment ``s`` is one solve along ``grids[s]`` with the tracers ``active[s]``
    (S, R - 1) moving and the rest frozen.
    """
    terms = field.terms(solver)
    split = isinstance(solver.solver, gf.Leapfrog)

    def segment(y: Any, xs: tuple[Array, Array]) -> tuple[Any, None]:
        grid, on = xs
        return solve_on(solver, terms, y, grid, args={"active": on}), None

    y0: Any = (w0[:, :3], w0[:, 3:]) if split else w0
    y: Any
    y, _ = jax.lax.scan(segment, y0, (grids, active))
    return jnp.concat(y, axis=-1) if split else y


@dispatch.abstract
def _evolve(mode: Any, *args: Any) -> tuple[Array, Array]:
    """Integrate the release states ``(lead0, trail0)`` to ``t1``."""


@dispatch
def _evolve(
    mode: WholeStream,  # noqa: ARG001
    sim: SpraySimulator,
    pot: gp.AbstractPotential,
    prog_pot: gp.AbstractPotential | None,
    schedule: Uniform,
    ws: Array,
    lead0: Array,
    trail0: Array,
) -> tuple[Array, Array]:
    # One segment per stretch between releases (one for a single release); the
    # tracers released by a segment's start move in it.
    n = schedule.n_release
    n_seg = max(n - 1, 1)
    L = schedule.n_steps() // n_seg
    starts = jnp.arange(n_seg) * L
    grids = schedule.lattice()[starts[:, None] + jnp.arange(L + 1)]
    on = schedule.release_index()[None, :] <= starts[:, None]
    w = evolve_whole_stream(
        _WholeStreamField(pot, prog_pot),
        sim.solver,
        jnp.concat([ws[:1], lead0, trail0]),
        grids,
        jnp.concat([on, on], axis=1),
    )
    return w[1 : n + 1], w[n + 1 :]


@dispatch
def _evolve(mode: PerParticle, *args: Any) -> tuple[Array, Array]:
    msg = f"per-particle mode with {type(mode.progenitor).__name__}"
    raise NotImplementedError(msg)


def to_mockstream(result: SprayResult, units: Any, /) -> MockStream:
    """Return ``result`` as a legacy `MockStream` with arms "lead" and "trail".

    ``units`` is the host potential's unit system (the result's units).
    """
    usys = u.unitsystem(units)
    rt = u.Q(result.release_times, usys["time"])
    t = u.Q(jnp.full_like(result.release_times, result.t1), usys["time"])

    def arm(w: Array) -> MockStreamArm:
        return MockStreamArm(
            q=u.Q(w[:, :3], usys["length"]),
            p=u.Q(w[:, 3:], usys["speed"]),
            t=t,
            release_time=rt,
            frame=gc.frames.simulation_frame,
        )

    return MockStream({"lead": arm(result.lead), "trail": arm(result.trail)})
