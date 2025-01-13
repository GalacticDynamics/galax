"""General wrapper around `diffrax.diffeqsolve`.

This is private API.

"""

__all__ = [
    "DiffEqSolver",  # exported to Public API
    # ---
    "default_stepsize_controller",
    "default_adjoint",
]

import inspect
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, TypeAlias

import diffrax
import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, PyTree, Real

RealScalarLike: TypeAlias = Real[int | float | Array | np.ndarray, ""]
BoolScalarLike: TypeAlias = Bool[ArrayLike, ""]


# Get the signature of `diffrax.diffeqsolve`, first unwrapping the
# `equinox.filter_jit`
params = inspect.signature(diffrax.diffeqsolve.__wrapped__).parameters
default_stepsize_controller = params["stepsize_controller"].default
default_saveat = params["saveat"].default
default_progress_meter = params["progress_meter"].default
default_event = params["event"].default
default_max_steps = params["max_steps"].default
default_throw = params["throw"].default
default_adjoint = params["adjoint"].default


class DiffEqSolver(eqx.Module, strict=True):  # type: ignore[call-arg,misc]
    """Class-based interface for solving differential equations.

    Examples
    --------
    >>> from diffrax import Dopri5, PIDController, SaveAt, ODETerm
    >>> from galax.dynamics._src.solve import DiffEqSolver

    >>> solver = DiffEqSolver(solver=Dopri5(),
    ...                       stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))
    >>> saveat = SaveAt(ts=[0., 1., 2., 3.])
    >>> term = ODETerm(lambda t, y, args: -y)
    >>> sol = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)

    >>> print(sol.ts)
    [0. 1. 2. 3.]

    >>> print(sol.ys)
    [1. 0.36788338 0.13533922 0.04978961]

    """

    _: KW_ONLY

    #: The solver for the differential equation.
    #: See the diffrax guide on how to choose a solver.
    solver: diffrax.AbstractSolver

    #: How to change the step size as the integration progresses.
    #: See diffrax's list of stepsize controllers.
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        default=default_stepsize_controller
    )

    #: How to differentiate `diffeqsolve`.
    #: See `diffrax` for options.
    adjoint: diffrax.AbstractAdjoint = eqx.field(default=default_adjoint)

    # TODO: should `max_steps` be a field? Given that `max_steps=None` can be
    # incompatible with some `SaveAt` options, it would still need to be
    # overridable in `__call__`.

    @partial(eqx.filter_jit)
    # @partial(quax.quaxify)  # TODO: so don't need to strip units
    def __call__(
        self: "DiffEqSolver",
        terms: PyTree[diffrax.AbstractTerm],
        /,
        t0: RealScalarLike,
        t1: RealScalarLike,
        dt0: RealScalarLike | None,
        y0: PyTree[ArrayLike],
        args: PyTree[Any] = None,
        *,
        saveat: diffrax.SaveAt = default_saveat,
        event: diffrax.Event | None = default_event,
        max_steps: int | None = default_max_steps,
        throw: bool = default_throw,
        progress_meter: diffrax.AbstractProgressMeter = default_progress_meter,
        solver_state: PyTree[ArrayLike] | None = None,
        controller_state: PyTree[ArrayLike] | None = None,
        made_jump: BoolScalarLike | None = None,
    ) -> diffrax.Solution:
        """Solve a differential equation.

        For all arguments, see `diffrax.diffeqsolve`.

        Args:
            terms : the terms of the differential equation.
            t0: the start of the region of integration.
            t1: the end of the region of integration.
            dt0: the step size to use for the first step.
            y0: the initial value. This can be any PyTree of JAX arrays.
            args: any additional arguments to pass to the vector field.
            saveat: what times to save the solution of the differential equation.
            adjoint: how to differentiate diffeqsolve.
            event: an event at which to terminate the solve early.
            max_steps: the maximum number of steps to take before quitting.
            throw: whether to raise an exception if the integration fails.
            progress_meter: a progress meter.
            solver_state: some initial state for the solver.
            controller_state: some initial state for the step size controller.
            made_jump: whether a jump has just been made at t0.

        """
        soln: diffrax.Solution = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0,
            t1,
            dt0,
            y0,
            args=args,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            event=event,
            max_steps=max_steps,
            throw=throw,
            progress_meter=progress_meter,
            solver_state=solver_state,
            controller_state=controller_state,
            made_jump=made_jump,
        )
        return soln

    # TODO: a contextmanager for producing a temporary DiffEqSolver with
    # different field values.
