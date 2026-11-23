"""Integration settings and the progenitor's orbit on the release lattice."""

__all__ = [
    "default_solver",
    "grid_time",
    "host_terms",
    "is_lattice",
    "map_terms",
    "progenitor_states",
    "solve_on",
]

from collections.abc import Callable
from jaxtyping import Array
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp

import diffraxtra as dfxtra

import galax.fields as gf
import galax.potential as gp
from .schedule import Times, Uniform


def default_solver() -> dfxtra.DiffEqSolver:
    """Return the default solver: `Leapfrog` on the schedule's lattice.

    ``max_steps=None`` means "take it from the lattice".
    """
    return dfxtra.DiffEqSolver(
        solver=gf.Leapfrog(),
        stepsize_controller=dfx.ConstantStepSize(),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
        max_steps=None,
    )


def is_lattice(solver: dfxtra.DiffEqSolver) -> bool:
    """Whether the solver steps on the schedule's lattice (fixed step)."""
    return isinstance(solver.stepsize_controller, dfx.ConstantStepSize)


def host_terms(pot: gp.AbstractPotential, solver: dfxtra.DiffEqSolver) -> Any:
    """Return the terms of a test particle in ``pot``, in the potential's units.

    `Leapfrog` takes two terms on ``y = (q, p)``; any other solver takes one
    term on ``y = w`` of shape ``(6,)``.
    """

    def acc(t: Array, q: Array) -> Array:
        return -gp.gradient(pot, q, t)  # type: ignore[no-any-return,operator]

    if isinstance(solver.solver, gf.Leapfrog):
        return (
            dfx.ODETerm(lambda t, p, args: p),  # noqa: ARG005
            dfx.ODETerm(lambda t, q, args: acc(t, q)),  # noqa: ARG005
        )
    return dfx.ODETerm(
        lambda t, w, args: jnp.concat([w[3:], acc(t, w[:3])])  # noqa: ARG005
    )


def grid_time(grid: Array, s: Array) -> Array:
    """Physical time at index time ``s``: linear in ``grid``, exact at integers."""
    i = jnp.clip(jnp.floor(s), 0, grid.shape[0] - 1).astype(int)
    j = jnp.minimum(i + 1, grid.shape[0] - 1)
    return grid[i] + (s - i) * (grid[j] - grid[i])


class _IndexTime(dfx.AbstractTerm):  # type: ignore[misc]
    """``term`` in index time: the step ``s -> s + 1`` is ``grid[s] -> grid[s + 1]``.

    A step between equal grid points has zero control, so it leaves the state as
    it is; diffrax's `StepTo` rejects such a step in physical time.
    """

    term: dfx.AbstractTerm
    grid: Array

    def vf(self, t: Array, y: Any, args: Any) -> Any:
        return self.term.vf(grid_time(self.grid, t), y, args)

    def contr(self, t0: Array, t1: Array, **kwargs: Any) -> Any:  # noqa: ARG002
        return self.term.contr(grid_time(self.grid, t0), grid_time(self.grid, t1))

    def prod(self, vf: Any, control: Any) -> Any:
        return self.term.prod(vf, control)


def map_terms(f: Callable[[dfx.AbstractTerm], Any], terms: Any) -> Any:
    """Apply ``f`` to each term in ``terms`` (one term, or a tuple of them)."""
    return jax.tree.map(f, terms, is_leaf=lambda x: isinstance(x, dfx.AbstractTerm))


def _solve(
    solver: dfxtra.DiffEqSolver,
    terms: Any,
    y0: Any,
    step_ts: Array,
    saveat: dfx.SaveAt,
    args: Any = None,
) -> Any:
    """Solve and return ``ys``; NaN for a failed solve instead of raising.

    A fixed-step solve steps from ``step_ts[k]`` to ``step_ts[k + 1]`` in index
    time (`_IndexTime`), so equal or NaN times never make diffrax raise. The
    terms still see physical times.
    """
    if is_lattice(solver):
        terms = map_terms(lambda x: _IndexTime(x, step_ts), terms)
        step_ts = jnp.arange(step_ts.shape[0], dtype=float)
        kw: dict[str, Any] = {
            "dt0": None,
            "stepsize_controller": dfx.StepTo(ts=step_ts),
            "max_steps": step_ts.shape[0] - 1,
        }
    else:
        max_steps = solver.max_steps
        if max_steps is None:
            # Unbounded, a NaN state would loop forever instead of giving NaN.
            msg = "an adaptive solve needs an integer `max_steps`, got None"
            raise ValueError(msg)
        kw = {
            "dt0": None,
            "stepsize_controller": solver.stepsize_controller,
            "max_steps": max_steps,
        }

    sol = dfx.diffeqsolve(
        terms,
        solver.solver,
        step_ts[0],
        step_ts[-1],
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=solver.adjoint,
        throw=False,
        **kw,
    )
    ok = sol.result == dfx.RESULTS.successful
    return jax.tree.map(lambda y: jnp.where(ok, y, jnp.nan), sol.ys)


def solve_on(
    solver: dfxtra.DiffEqSolver,
    terms: Any,
    y0: Any,
    step_ts: Array,
    *,
    args: Any = None,
) -> Any:
    """Return the final state after solving along ``step_ts``.

    A fixed-step solver takes exactly the steps ``step_ts``. An adaptive one
    runs from ``step_ts[0]`` to ``step_ts[-1]``. ``args`` go to the terms. A
    failed solve returns NaN, it never raises.
    """
    ys = _solve(solver, terms, y0, step_ts, dfx.SaveAt(t1=True), args)
    return jax.tree.map(lambda y: y[0], ys)


def progenitor_states(
    pot: gp.AbstractPotential,
    w_today: Array,
    schedule: Uniform | Times,
    solver: dfxtra.DiffEqSolver,
) -> Array:
    """Return the progenitor's states ``(M, 6)`` on ``schedule.grid()``.

    It integrates ``w_today`` back from ``t1`` to the first grid point, then
    forward saving every grid point. A fixed-step solver steps back and forth
    on the grid, so the states at the release times are the integrator's own;
    a grid over zero time gives ``w_today`` at every point. An adaptive one
    saves at the grid points.
    """
    grid = schedule.grid()
    terms = host_terms(pot, solver)
    split = isinstance(solver.solver, gf.Leapfrog)
    y0 = (w_today[:3], w_today[3:]) if split else w_today
    # A NaN time, or a release after t1, gives NaN. In an adaptive solve a
    # NaN time would make diffrax raise (`SaveAt(ts=...)`) or grind to
    # `max_steps`: it solves over zero time instead.
    ts = schedule.release_times()
    ok = jnp.all(jnp.isfinite(grid)) & jnp.all(ts <= schedule.t1)
    if is_lattice(solver):
        # Back and forth on the same grid: Leapfrog retraces its steps, so the
        # orbit passes through `w_today` at t1.
        saveat = dfx.SaveAt(t0=True, steps=True)
    else:
        grid = jnp.where(ok, grid, 0.0)
        saveat = dfx.SaveAt(ts=grid)
    y_start = solve_on(solver, terms, y0, grid[::-1])
    ys = _solve(solver, terms, y_start, grid, saveat)
    w = jnp.concat(ys, axis=-1) if split else ys
    return jnp.where(ok, w, jnp.nan)
