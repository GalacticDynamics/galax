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

import diffrax as dfx
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
from .fields import _CopiesField, _TrackField, _WholeStreamField
from .schedule import Uniform
from .solve import (
    default_solver,
    host_terms,
    is_lattice,
    progenitor_states,
    solve_on,
)
from .track import make_track
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
    """Per-particle progenitor model: one shared track of the progenitor.

    Every particle feels the progenitor potential centred on the same track:
    the progenitor's orbit from ``progenitor_solver``, Hermite-interpolated
    between knots (its fixed-step lattice, or ``max(20 n_release, 2**14)``
    uniform knots of an adaptive solve). Gradients use the knots' sensitivity
    ``S`` to ``theta``, the inexact array leaves of ``(w_today, pot,
    schedule)`` that ``wrt`` selects, computed once with `jax.jacfwd`. Under
    `jax.jit` a forward-only run never computes ``S``: nothing uses it, so XLA
    drops it as dead code.

    ``S`` takes one tangent progenitor solve per entry of ``theta`` and holds
    ``(n_knots, 6, len(theta))`` floats, so the gradient's cost and memory grow
    with ``len(theta)``. A host with many parameters (a multipole expansion
    has thousands) can restrict ``theta`` with ``wrt``.

    First derivatives are exact. Second derivatives through the track are not:
    its sensitivity is held constant, so the track's own curvature in those
    inputs is dropped.

    Parameters
    ----------
    wrt : filter spec
        An equinox filter spec (as for `equinox.partition`) over ``(w_today,
        pot, schedule)``: a callable applied to each leaf, or a pytree prefix
        of bools and callables (built with `equinox.tree_at` on array leaves,
        e.g. ``pot["halo"].m.value.value``, not on parameter nodes, which
        can't hold a bool). Default: every inexact array leaf. Leaves
        outside ``theta`` get no gradient through the track: the part of their
        gradient from moving the progenitor that the particles orbit is
        dropped, not approximated. Their effect on the release states and on
        the host's force on the particles is kept. Left out, ``t_age`` and
        ``t1`` still move the knot times; only the knots' values are held.
    """

    wrt: Any = eqx.is_inexact_array


class Copies(eqx.Module):
    """Per-particle progenitor model: each particle carries its own copy.

    A copy starts from the progenitor's state at the release (from
    ``progenitor_solver``) and moves with ``solver``. With the same fixed-step
    solver it equals the progenitor to round-off; with different solvers it
    drifts from it by their difference.
    """


class PerParticle(eqx.Module):
    """Integrate each particle with its own solve, from its release to ``t1``.

    The lead and trail of a release are vmapped; ``loop`` (`jaxmore` `VMap` or
    `Scan`) maps over the releases. A failed solve gives NaN for its particle.
    """

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
        callable is given ``Msat`` (a scalar, traced) and returns the potential.
        It must use the host's unit system.
    solver : diffraxtra.DiffEqSolver
        Settings of the stream's solves. A `diffrax.ConstantStepSize`
        controller steps on the schedule's lattice; any other is adaptive.
    progenitor_solver : diffraxtra.DiffEqSolver | None
        Settings of the progenitor's orbit in the per-particle modes; `None`
        means ``solver``. `WholeStream` integrates the progenitor with the
        stream, so there it must be `None` or equal to ``solver``.

    Calling it with ``(key, pot, w_today, Msat, schedule)`` returns a
    `SprayResult`: ``w_today`` is the progenitor's state at ``schedule.t1``,
    ``Msat`` a scalar or, with a fixed or no progenitor potential, one mass per
    release. A failed solve gives NaN, for the whole result in `WholeStream`
    and for that particle in `PerParticle`; it never raises. ``lead[i]`` and
    ``trail[i]`` share ``release_times[i]``. All inputs and outputs are raw
    arrays in the host potential's units.
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
        if jnp.ndim(Msat) != 0:
            msg = (
                "a callable progenitor_potential needs a scalar Msat; pass a "
                "time-dependent potential for a varying mass"
            )
            raise ValueError(msg)
        pp = pp(Msat)
    lead, trail, prog = _evolve(
        sim.mode,
        sim,
        pot,
        pp,
        schedule,
        w_today,
        ws,
        jnp.concat([xl, vl], axis=-1),
        jnp.concat([xt, vt], axis=-1),
    )
    return SprayResult(lead, trail, ts, prog, jnp.asarray(schedule.t1, dtype=float))


def evolve_whole_stream(
    field: _WholeStreamField,
    solver: dfxtra.DiffEqSolver,
    w0: Array,
    grids: Array,
    active: Array,
) -> Array:
    """Evolve the rows ``w0`` (R, 6) over the segments ``grids`` (S, L + 1).

    Segment ``s`` is one solve along ``grids[s]`` with the tracers ``active[s]``
    (S, R - 1) moving and the rest frozen. If a segment fails, the later ones
    go on from the state before it (a NaN state would make an adaptive solve
    run to ``max_steps``) and every row comes out NaN.
    """
    terms = field.terms(solver)
    split = isinstance(solver.solver, gf.Leapfrog)

    def segment(
        carry: tuple[Any, Array], xs: tuple[Array, Array]
    ) -> tuple[tuple[Any, Array], None]:
        y, failed = carry
        grid, on = xs
        y1 = solve_on(solver, terms, y, grid, args={"active": on})
        bad = jnp.any(jnp.isnan(jnp.concat(jax.tree.leaves(y1), axis=None)))
        y1 = jax.tree.map(lambda a, b: jnp.where(bad, a, b), y, y1)
        return (y1, failed | bad), None

    y0: Any = (w0[:, :3], w0[:, 3:]) if split else w0
    no = jnp.zeros((), bool)
    (y, failed), _ = jax.lax.scan(segment, (y0, no), (grids, active))
    w = jnp.concat(y, axis=-1) if split else y
    return jnp.where(failed, jnp.nan, w)


class _FromRelease(dfx.AbstractTerm):  # type: ignore[misc]
    """``term`` with time held at ``t_release`` before it.

    Every step that ends by ``t_release`` has a zero control, so it leaves the
    state exactly as it is; later steps are the plain ``term``'s.
    """

    term: dfx.AbstractTerm
    t_release: Array

    def vf(self, t: Array, y: Any, args: Any) -> Any:
        return self.term.vf(jnp.maximum(t, self.t_release), y, args)

    def contr(self, t0: Array, t1: Array, **kwargs: Any) -> Array:  # noqa: ARG002
        return jnp.maximum(t1, self.t_release) - jnp.maximum(t0, self.t_release)

    def prod(self, vf: Any, control: Array) -> Any:
        return self.term.prod(vf, control)


def evolve_per_particle(
    terms: Any,
    solver: dfxtra.DiffEqSolver,
    w0: Array,
    t0: Array,
    lattice: Array,
    loop: type[lstrat.Scan | lstrat.VMap],
) -> Array:
    """Evolve each ``w0[i, j]`` (..., 6) alone, from ``t0[i]`` to ``lattice[-1]``.

    ``j`` (lead, trail) is vmapped; ``loop`` vmaps or scans over ``i``. A
    fixed-step solve steps over the whole ``lattice`` (static shapes), with
    no-op steps up to ``t0[i]``, so it takes exactly the lattice steps from
    ``t0[i]``. An adaptive one starts at ``t0[i]``. A failed solve gives NaN
    for its particle only.
    """
    split = isinstance(solver.solver, gf.Leapfrog)
    on_lattice = is_lattice(solver)

    def one(w: Array, t: Array) -> Array:
        if on_lattice:
            tm = jax.tree.map(
                lambda x: _FromRelease(x, t),
                terms,
                is_leaf=lambda x: isinstance(x, dfx.AbstractTerm),
            )
            ts = lattice
        else:
            tm, ts = terms, jnp.stack([t, lattice[-1]])
        y = solve_on(solver, tm, (w[..., :3], w[..., 3:]) if split else w, ts)
        return jnp.concat(y, axis=-1) if split else y

    pair = jax.vmap(one, in_axes=(0, None))
    if loop is lstrat.Scan:
        w: Array = jax.lax.map(lambda x: pair(*x), (w0, t0))
        return w
    return jax.vmap(pair)(w0, t0)


@dispatch.abstract
def _evolve(mode: Any, *args: Any) -> tuple[Array, Array, Array]:
    """Integrate the release states to ``t1``: ``(lead, trail, progenitor)``."""


@dispatch
def _evolve(
    mode: WholeStream,  # noqa: ARG001
    sim: SpraySimulator,
    pot: gp.AbstractPotential,
    prog_pot: gp.AbstractPotential | None,
    schedule: Uniform,
    w_today: Array,  # noqa: ARG001
    ws: Array,
    lead0: Array,
    trail0: Array,
) -> tuple[Array, Array, Array]:
    # The tracers orbit the progenitor row, so it must start from (and its
    # release states come from) the same solver.
    psolver = sim.progenitor_solver
    if psolver is not None and eqx.tree_equal(psolver, sim.solver) is not True:
        msg = (
            "WholeStream integrates the progenitor with the stream: "
            "progenitor_solver must be None or equal to solver"
        )
        raise ValueError(msg)
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
    return w[1 : n + 1], w[n + 1 :], w[0]


@dispatch
def _evolve(
    mode: PerParticle,
    sim: SpraySimulator,
    pot: gp.AbstractPotential,
    prog_pot: gp.AbstractPotential | None,
    schedule: Uniform,
    w_today: Array,
    ws: Array,
    lead0: Array,
    trail0: Array,
) -> tuple[Array, Array, Array]:
    idx = schedule.release_index()
    lattice = schedule.lattice()
    w0 = jnp.stack([lead0, trail0], axis=1)  # (n, 2, 6)
    if prog_pot is None:  # no progenitor model needed: the host alone
        terms = host_terms(pot, sim.solver)
    elif isinstance(mode.progenitor, Copies):  # rows [particle, copy]
        terms = _CopiesField(pot, prog_pot).terms(sim.solver)
        w0 = jnp.stack([w0, jnp.broadcast_to(ws[idx, None], w0.shape)], axis=2)
    else:  # one row about the shared track
        psolver = sim.progenitor_solver
        psolver = sim.solver if psolver is None else psolver
        track = make_track(pot, w_today, schedule, psolver, ws, mode.progenitor.wrt)
        terms = _TrackField(pot, prog_pot, track).terms(sim.solver)
        w0 = w0[:, :, None]
    w = evolve_per_particle(terms, sim.solver, w0, lattice[idx], lattice, mode.loop)
    if prog_pot is not None:
        w = w[:, :, 0]
    return w[:, 0], w[:, 1], ws[-1]


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
