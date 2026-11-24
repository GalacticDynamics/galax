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

import functools as ft
import warnings
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
from .df import AbstractKinematicDF, Chen2024DF, sample_releases
from .fields import _CopiesField, _TrackField, _WholeStreamField
from .schedule import Times, Uniform
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

    The fastest mode, with one progenitor shared by every tracer; the default
    choice for fits. Its gradient needs the most memory of the modes.

    The state holds the progenitor and every tracer. A tracer is frozen at its
    release state until its release time, then moves in the host plus the
    progenitor potential centred on the progenitor row. With a `Uniform`
    schedule, or with `Times` and an adaptive controller, each stretch between
    consecutive release times is its own solve, so a tracer's first step uses
    the force at its release state. With `Times` and fixed steps, it is one
    solve over the merged grid, in which each tracer's steps have zero length
    until its release; the step that ends at the release computes the force at
    the release state, so the first step after it uses that force too.

    With an adaptive controller, diffrax's default error norm is the RMS over
    the whole state, frozen tracers included. So one particle's error can reach
    about ``sqrt(N)`` times the tolerance for ``N`` rows. Pass a max norm to the
    controller (``diffrax.PIDController(..., norm=optimistix.max_norm)``) if the
    tolerance should hold per particle.
    """


class SharedTrack(eqx.Module):
    """Per-particle progenitor model: one shared track of the progenitor.

    One progenitor shared by every particle, like `WholeStream`: exact at the
    lattice points with `Leapfrog` (a fixed-step Runge-Kutta solver's stages
    read the Hermite interpolant between knots), Hermite-interpolated between
    knots with an adaptive progenitor solver. The gradient's cost grows with
    ``len(theta)`` (below).

    Every particle feels the progenitor potential centred on the same track:
    the progenitor's orbit from ``progenitor_solver``, Hermite-interpolated
    between knots (its fixed-step lattice, or ``max(20 n_release, 2**14)``
    uniform knots of an adaptive solve; with `Times`, ``n_release`` there is
    ``steps + 1``). Gradients use the knots' sensitivity
    ``S`` to ``theta``, the inexact array leaves of ``(w_today, pot,
    schedule)`` that ``wrt`` selects, computed once with `jax.jacfwd`. Under
    `jax.jit` a forward-only run never computes ``S``: nothing uses it, so XLA
    drops it as dead code.

    ``S`` takes one tangent progenitor solve per entry of ``theta`` and holds
    ``(n_knots, 6, len(theta))`` floats, so the gradient's cost and memory grow
    with ``len(theta)``. A host with many parameters (a multipole expansion
    has thousands) can restrict ``theta`` with ``wrt``.

    With a `Times` schedule the knots are on its uniform lattice
    (`Times.uniform`), not on the merged grid of the release times, so
    ``theta`` holds that schedule's ``t_age`` and ``t1`` and ``wrt`` is given
    it as ``schedule``. With fixed steps that orbit is a separate solve along
    the lattice, which differs from the merged-grid orbit the releases come
    from by the two discretizations' difference.

    An adaptive ``progenitor_solver`` makes ``S`` costly: it takes
    ``len(theta)`` tangent solves through `jax.jacfwd`, each running the
    adaptive solver over every knot, so its cost is many adaptive progenitor
    solves and grows with the knot count. A fixed-step ``progenitor_solver``
    avoids this.

    First derivatives are exact. Second derivatives through the track are not:
    its sensitivity is held constant, so the track's own curvature in those
    inputs is dropped.

    Parameters
    ----------
    wrt : Callable | None
        Like `equinox.tree_at`'s ``where``: called with the call's
        ``(w_today, pot, schedule)``, it returns the leaves or subtrees to
        differentiate, e.g. ``lambda w, pot, s: (w, pot["halo"].m, s.t_age)``.
        Every inexact array under them goes into ``theta``. `None` (the
        default) means every inexact array leaf. The schedule's inexact
        leaves (``t_age``, ``t1``) are always in ``theta``, whatever ``wrt``
        returns: they move the knot times, so holding the knots' values would
        shift the track spuriously in time. Leaves outside ``theta`` get no
        gradient through the track: the part of their gradient from moving the
        progenitor that the particles orbit is dropped, not approximated.
        Their effect on the release states and on the host's force on the
        particles is kept. The dropped part can dominate: leaving out a host
        mass scale can make its gradient wrong by a large factor, so leave a
        leaf out only if the progenitor's orbit doesn't depend on it.
    """

    wrt: Callable[..., Any] | None = None


class Copies(eqx.Module):
    """Per-particle progenitor model: each particle carries its own copy.

    Its gradient's cost, unlike `SharedTrack`'s, doesn't grow with the number
    of inputs differentiated, and its forward run needs the least memory.

    A copy starts from the progenitor's state at the release (from
    ``progenitor_solver``) and moves with ``solver``. With the same fixed-step
    solver it equals the progenitor to round-off; with different solvers it
    drifts from it by their difference.
    """


class PerParticle(eqx.Module):
    """Integrate each particle with its own solve, from its release to ``t1``.

    The lead and trail of a release are vmapped; ``loop`` (`jaxmore` `VMap` or
    `Scan`) maps over the releases. A failed solve gives NaN for its particle.
    With fixed steps, a particle steps on the schedule's grid from its release
    (for `Times`, the merged grid, like `WholeStream`); an adaptive solve
    starts at its release time.
    Pick it for adaptive steps chosen per particle, or for a gradient with less
    memory than `WholeStream`'s; `Scan` cuts that memory further, at about 10
    times the gradient's run time with `VMap`.
    """

    progenitor: SharedTrack | Copies
    loop: type[lstrat.Scan | lstrat.VMap] = eqx.field(default=lstrat.VMap, static=True)

    def __check_init__(self) -> None:
        if self.loop not in (lstrat.Scan, lstrat.VMap):
            msg = f"loop must be jaxmore's VMap or Scan, not {self.loop!r}"
            raise TypeError(msg)


class SprayResult(eqx.Module):
    """The stream at ``t1``, as raw arrays in the host potential's units.

    ``lead[i]`` and ``trail[i]`` were both released at ``release_times[i]``,
    in the schedule's order (for `Times`, the order of its ``ts``).
    """

    lead: Array  # (n_release, 6)
    trail: Array  # (n_release, 6)
    release_times: Array  # (n_release,)
    progenitor: Array  # (6,), the progenitor at t1
    t1: Array  # ()


class SpraySimulator(eqx.Module):
    """Spray a stellar stream from a progenitor's state today.

    Calling it with ``(key, pot, w_today, Msat, schedule)`` returns a
    `SprayResult`. ``schedule`` is a `Uniform` (releases on a lattice) or a
    `Times` (releases at any times).
    ``w_today`` is the progenitor's state at ``schedule.t1``,
    ``Msat`` a scalar or, with a fixed or no progenitor potential, one mass per
    release. A failed solve gives NaN, for the whole result in `WholeStream`
    and for that particle in `PerParticle`; it never raises. Gradients at a
    failed point are not meaningful: some come out NaN and some finite, so
    test the outputs, not the gradients, for failure. ``lead[i]`` and
    ``trail[i]`` share ``release_times[i]``. All inputs and outputs are raw
    arrays in the host potential's units. The call is jitted: pass ``Msat`` as
    a JAX array, since a Python float is static and retraces for each value.

    Gradients: the default adjoint, `diffrax.RecursiveCheckpointAdjoint`,
    supports reverse mode (`jax.grad`). For `jax.jvp` or `jax.jacfwd` use
    `diffrax.ForwardMode` in ``solver`` (and ``progenitor_solver``); it can't
    be reverse-differentiated, and the default adjoint can't be
    forward-differentiated. `diffrax.DirectAdjoint` supports both, e.g. for
    `jax.hessian`, at a higher cost. Second derivatives through a
    `SharedTrack` are not exact.

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
        controller steps on the schedule's grid (`Uniform`'s lattice, or
        `Times`'s merged grid of lattice points and release times); any other
        is adaptive.
    progenitor_solver : diffraxtra.DiffEqSolver | None
        Settings of the progenitor's orbit in the per-particle modes; `None`
        means ``solver``. `WholeStream` integrates the progenitor with the
        stream, so there it must be `None` or equal to ``solver``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    A whole stream of 4 releases over 300 Myr, 5 Leapfrog steps apart, in
    kpc, Myr and Msun:

    >>> pot = gp.NFWPotential(m=1e12, r_s=15.0, units="galactic")
    >>> w_today = jnp.array([30.0, 10.0, 20.0, 0.01, -0.15, -0.02])
    >>> schedule = gd.experimental.Uniform(jnp.asarray(300.0), 4, 5)
    >>> sim = gd.experimental.SpraySimulator(
    ...     gd.experimental.Fardal2015DF(), gd.experimental.WholeStream()
    ... )
    >>> res = sim(jr.key(0), pot, w_today, jnp.asarray(1e5), schedule)
    >>> res.lead.shape, res.trail.shape
    ((4, 6), (4, 6))
    >>> res.release_times
    Array([-300., -200., -100.,    0.], dtype=float64)

    """

    df: AbstractKinematicDF
    mode: WholeStream | PerParticle
    _: KW_ONLY
    progenitor_potential: ProgenitorPotential = None
    solver: dfxtra.DiffEqSolver = eqx.field(default_factory=default_solver)
    progenitor_solver: dfxtra.DiffEqSolver | None = None

    def __check_init__(self) -> None:
        if self.progenitor_potential is None and isinstance(self.df, Chen2024DF):
            warnings.warn(
                "Chen2024DF without a progenitor_potential is the "
                '"no progenitor" variant of the Chen+24 model; pass e.g. '
                "progenitor_potential=lambda m: PlummerPotential(m_tot=m, "
                'r_s=u.Q(4, "pc"), units=...) for its default model.',
                RuntimeWarning,
                stacklevel=3,  # past equinox's __call__, to the constructor call
            )

    @eqx.filter_jit
    def __call__(
        self,
        key: PRNGKeyArray,
        pot: gp.AbstractPotential,
        w_today: Array,
        Msat: Array,
        schedule: Uniform | Times,
        /,
    ) -> SprayResult:
        return simulate(self, key, pot, w_today, Msat, schedule)


def simulate(
    sim: SpraySimulator,
    key: PRNGKeyArray,
    pot: gp.AbstractPotential,
    w_today: Array,
    Msat: Array,
    schedule: Uniform | Times,
    /,
) -> SprayResult:
    """Sample the releases on the progenitor's orbit and integrate them."""
    w_today = jnp.asarray(w_today, dtype=float)
    # Times has 1-d ts, Uniform a scalar t_age.
    times = schedule.ts if isinstance(schedule, Times) else schedule.t_age
    if (
        jnp.shape(w_today) != (6,)
        or jnp.ndim(times) != int(isinstance(schedule, Times))
        or jnp.ndim(schedule.t1) != 0
    ):
        msg = (
            "w_today must have shape (6,), t1 and t_age be scalars and ts be "
            f"1-d, got w_today {jnp.shape(w_today)}, t1 {jnp.shape(schedule.t1)} "
            f"and t_age or ts {jnp.shape(times)}; vmap over a batch"
        )
        raise ValueError(msg)
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
    terms: Any,
    solver: dfxtra.DiffEqSolver,
    w0: Array,
    grids: Array,
    active: Array,
    gate: Callable[[dfx.AbstractTerm], Any] | None = None,
) -> Array:
    """Evolve the rows ``w0`` (R, 6) with ``terms`` over the segments ``grids``.

    Segment ``s`` is one solve along ``grids[s]`` (S, L + 1) with the tracers
    ``active[s]`` (S, R - 1) moving and the rest frozen. ``gate`` goes to
    `solve_on`. The fixed-step `Times` path is one segment with every tracer in
    ``active``: its ``gate``, `_Gate`, freezes each row until its release, not
    ``active``. If ``w0`` is not finite
    or a segment fails, every row comes out NaN. After that, an adaptive
    solver's later segments span zero time: a NaN state would make it run to
    ``max_steps``.
    """
    split = isinstance(solver.solver, gf.Leapfrog)
    adaptive = not is_lattice(solver)

    def segment(
        carry: tuple[Any, Array], xs: tuple[Array, Array]
    ) -> tuple[tuple[Any, Array], None]:
        y, failed = carry
        grid, on = xs
        if adaptive:
            grid = jnp.where(failed, grid[0], grid)
        y1 = solve_on(solver, terms, y, grid, args={"active": on}, gate=gate)
        bad = jnp.any(~jnp.isfinite(jnp.concat(jax.tree.leaves(y1), axis=None)))
        y1 = jax.tree.map(lambda a, b: jnp.where(bad, a, b), y, y1)
        return (y1, failed | bad), None

    y0: Any = (w0[:, :3], w0[:, 3:]) if split else w0
    bad0 = jnp.any(~jnp.isfinite(w0))
    (y, failed), _ = jax.lax.scan(segment, (y0, bad0), (grids, active))
    w = jnp.concat(y, axis=-1) if split else y
    return jnp.where(failed, jnp.nan, w)


class _FromRelease(dfx.AbstractTerm):  # type: ignore[misc]
    """An index-time ``term`` held at the release's grid index ``start``.

    Every step ``k < start`` has a zero control, in value and tangent, so it
    leaves the state exactly as it is; later steps are the plain ``term``'s,
    and before ``start`` the vector field is read at the release. Held by
    index, not time, a release tied with the grid point before it never
    travels the zero step between them, so only the particle's own steps
    carry tangents.
    """

    term: dfx.AbstractTerm
    start: Array

    def vf(self, t: Array, y: Any, args: Any) -> Any:
        return self.term.vf(jnp.maximum(t, self.start), y, args)

    def contr(self, t0: Array, t1: Array, **kwargs: Any) -> Any:  # noqa: ARG002
        s = self.start
        return self.term.contr(jnp.maximum(t0, s), jnp.maximum(t1, s))

    def prod(self, vf: Any, control: Array) -> Any:
        return self.term.prod(vf, control)


def evolve_per_particle(
    terms: Any,
    solver: dfxtra.DiffEqSolver,
    w0: Array,
    t0: Array,
    i0: Array,
    lattice: Array,
    loop: type[lstrat.Scan | lstrat.VMap],
) -> Array:
    """Evolve each ``w0[i, j]`` (..., 6) alone, from ``t0[i]`` to ``lattice[-1]``.

    ``j`` (lead, trail) is vmapped; ``loop`` vmaps or scans over ``i``. A
    fixed-step solve steps over the whole ``lattice`` (static shapes), with
    no-op steps up to ``t0[i] = lattice[i0[i]]``, so it takes exactly the
    lattice steps from index ``i0[i]``. An adaptive one starts at ``t0[i]``.
    A failed solve gives NaN for its particle only. A non-finite ``lattice``
    or ``t0`` gives NaN for every particle and never raises: the solves run
    on stand-in times (``arange``, or zero time when adaptive).
    """
    split = isinstance(solver.solver, gf.Leapfrog)
    on_lattice = is_lattice(solver)
    ok = jnp.all(jnp.isfinite(lattice)) & jnp.all(jnp.isfinite(t0))
    stand_in = jnp.arange(lattice.shape[0], dtype=float) if on_lattice else 0.0
    lattice = jnp.where(ok, lattice, stand_in)
    t0 = jnp.where(ok, t0, lattice[0])

    def one(w: Array, t: Array, i: Array) -> Array:
        y: Any = (w[..., :3], w[..., 3:]) if split else w
        if on_lattice:
            y = solve_on(solver, terms, y, lattice, gate=lambda x: _FromRelease(x, i))
        else:
            y = solve_on(solver, terms, y, jnp.stack([t, lattice[-1]]))
        return jnp.concat(y, axis=-1) if split else y

    pair = jax.vmap(one, in_axes=(0, None, None))
    if loop is lstrat.Scan:
        w: Array = jax.lax.map(lambda x: pair(*x), (w0, t0, i0))
    else:
        w = jax.vmap(pair)(w0, t0, i0)
    return jnp.where(ok, w, jnp.nan)


class _Gate(dfx.AbstractTerm):  # type: ignore[misc]
    """An index-time whole-stream term with each row frozen until its release.

    Row ``r`` has a zero control on every step ``k < start[r]``, its release's
    grid index (0 for the progenitor row), and its derivatives are switched off
    (``args["active"]``) before it. The step that ends at a release evaluates
    the row's force at its release state, with the progenitor row at the
    release time, so `Leapfrog`'s cached force is right for the next step: no
    kick leaks across the release. Gated by index, not time, a zero step at a
    tie (two releases, or a release and a lattice point) carries its tangent
    only for the rows that travel it, those released by its start.
    """

    term: dfx.AbstractTerm
    start: Array  # (R,)

    def vf(self, t: Array, y: Any, args: Any) -> Any:  # noqa: ARG002
        return self.term.vf(t, y, {"active": (t >= self.start)[1:]})

    def contr(self, t0: Array, t1: Array, **kwargs: Any) -> Array:  # noqa: ARG002
        return jnp.where(t0 >= self.start, self.term.contr(t0, t1), 0.0)

    def prod(self, vf: Any, control: Array) -> Any:
        return jax.tree.map(lambda v: v * control[:, None], vf)


@dispatch.abstract
def _evolve(mode: Any, *args: Any) -> tuple[Array, Array, Array]:
    """Integrate the release states to ``t1``: ``(lead, trail, progenitor)``."""


@dispatch
def _evolve(
    mode: WholeStream,  # noqa: ARG001
    sim: SpraySimulator,
    pot: gp.AbstractPotential,
    prog_pot: gp.AbstractPotential | None,
    schedule: Uniform | Times,
    w_today: Array,  # noqa: ARG001
    ws: Array,
    lead0: Array,
    trail0: Array,
) -> tuple[Array, Array, Array]:
    # The tracers orbit the progenitor row, so it must start from (and its
    # release states come from) the same solver.
    psolver = sim.progenitor_solver
    if (
        psolver is not None
        and psolver is not sim.solver
        and eqx.tree_equal(psolver, sim.solver) is not True
    ):
        msg = (
            "WholeStream integrates the progenitor with the stream: "
            "progenitor_solver must be None or equal to solver"
        )
        raise ValueError(msg)
    n = lead0.shape[0]
    terms = _WholeStreamField(pot, prog_pot).terms(sim.solver)
    ts = schedule.release_times()
    gate = None
    if isinstance(schedule, Times) and is_lattice(sim.solver):
        # One solve over the merged grid, each row gated by its release's grid
        # index; the progenitor row starts at the grid's first point.
        idx = schedule.release_index()
        first = jnp.concat([jnp.zeros(1, dtype=idx.dtype), idx, idx])
        gate = ft.partial(_Gate, start=first)
        grids, on = schedule.grid()[None], jnp.ones((1, n), dtype=bool)
        start = ws[0]
    else:
        # One solve per segment; the tracers released by its start move in it.
        grids, on = schedule.segments()
        # The progenitor at min(ts). Not ws[0]: with `Times`, lattice round-off
        # can put the grid's first point an ulp before min(ts) (a release
        # equal to it sorts first).
        start = ws[schedule.release_index()[jnp.argmin(ts)]]
    w = evolve_whole_stream(
        terms,
        sim.solver,
        jnp.concat([start[None], lead0, trail0]),
        grids,
        jnp.concat([on, on], axis=1),
        gate,
    )
    return w[1 : n + 1], w[n + 1 :], w[0]


@dispatch
def _evolve(
    mode: PerParticle,
    sim: SpraySimulator,
    pot: gp.AbstractPotential,
    prog_pot: gp.AbstractPotential | None,
    schedule: Uniform | Times,
    w_today: Array,
    ws: Array,
    lead0: Array,
    trail0: Array,
) -> tuple[Array, Array, Array]:
    idx = schedule.release_index()
    w0 = jnp.stack([lead0, trail0], axis=1)  # (n, 2, 6)
    if prog_pot is None:  # no progenitor model needed: the host alone
        terms = host_terms(pot, sim.solver)
    elif isinstance(mode.progenitor, Copies):  # rows [particle, copy]
        terms = _CopiesField(pot, prog_pot).terms(sim.solver)
        w0 = jnp.stack([w0, jnp.broadcast_to(ws[idx, None], w0.shape)], axis=2)
    else:  # one row about the shared track
        psolver = sim.progenitor_solver
        psolver = sim.solver if psolver is None else psolver
        # With `Times`, the knots are on its uniform lattice, so theta holds
        # that schedule's t_age and t1, not every release time.
        ksched = schedule.uniform() if isinstance(schedule, Times) else schedule
        kws = ws
        if isinstance(schedule, Times) and is_lattice(psolver):
            kws = progenitor_states(pot, w_today, ksched, psolver)
        track = make_track(pot, w_today, ksched, psolver, kws, mode.progenitor.wrt)
        terms = _TrackField(pot, prog_pot, track).terms(sim.solver)
        w0 = w0[:, :, None]
    # A fixed-step solve steps on the grid (for `Times`, the merged grid, like
    # `WholeStream`): a copy then takes the progenitor's own steps.
    t0, steps = schedule.release_times(), schedule.grid()
    w = evolve_per_particle(terms, sim.solver, w0, t0, idx, steps, mode.loop)
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
