"""Integration settings and the progenitor's orbit on the release lattice."""

__all__ = [
    "default_solver",
    "host_terms",
    "is_lattice",
    "progenitor_states",
    "solve_on",
]

from jaxtyping import Array
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp

import diffraxtra as dfxtra

import galax.fields as gf
import galax.potential as gp
from .schedule import Uniform


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


def _solve(
    solver: dfxtra.DiffEqSolver,
    terms: Any,
    y0: Any,
    step_ts: Array,
    saveat: dfx.SaveAt,
    args: Any = None,
) -> Any:
    """Solve and return ``ys``; NaN for a failed solve instead of raising."""
    if is_lattice(solver):
        kw: dict[str, Any] = {
            "dt0": None,
            "stepsize_controller": dfx.StepTo(ts=step_ts),
            "max_steps": step_ts.shape[0] - 1,
        }
    else:
        max_steps = solver.max_steps
        if max_steps is None and isinstance(
            solver.adjoint, dfx.RecursiveCheckpointAdjoint
        ):
            msg = (
                "an adaptive solve with RecursiveCheckpointAdjoint needs an "
                "integer `max_steps`, got None"
            )
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
    schedule: Uniform,
    solver: dfxtra.DiffEqSolver,
) -> Array:
    """Return the progenitor's states ``(n_steps + 1, 6)`` on the lattice.

    It integrates ``w_today`` back from ``t1`` to the first lattice point, then
    forward saving every lattice point.
    """
    lattice = schedule.lattice()
    terms = host_terms(pot, solver)
    split = isinstance(solver.solver, gf.Leapfrog)
    y0 = (w_today[:3], w_today[3:]) if split else w_today

    y_start = solve_on(solver, terms, y0, lattice[::-1])
    if is_lattice(solver):
        saveat = dfx.SaveAt(t0=True, steps=True)
    else:
        saveat = dfx.SaveAt(ts=lattice)
    ys = _solve(solver, terms, y_start, lattice, saveat)
    return jnp.concat(ys, axis=-1) if split else ys
