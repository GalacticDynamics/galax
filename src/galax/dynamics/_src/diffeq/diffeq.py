"""General wrapper around `diffrax.diffeqsolve`.

This is private API.

"""
# ruff:noqa: ERA001

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

import diffrax as dfx
import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, PyTree, Real

from .interp import VectorizedDenseInterpolation

RealSz0Like: TypeAlias = Real[int | float | Array | np.ndarray, ""]
BoolSz0Like: TypeAlias = Bool[ArrayLike, ""]


# Get the signature of `dfx.diffeqsolve`, first unwrapping the
# `equinox.filter_jit`
params = inspect.signature(dfx.diffeqsolve.__wrapped__).parameters
default_stepsize_controller = params["stepsize_controller"].default
default_saveat = params["saveat"].default
default_progress_meter = params["progress_meter"].default
default_event = params["event"].default
default_max_steps = params["max_steps"].default
default_throw = params["throw"].default
default_adjoint = params["adjoint"].default


class DiffEqSolver(eqx.Module, strict=True):  # type: ignore[call-arg,misc]
    """Class-based interface for solving differential equations.

    This is a convenience wrapper around `diffrax.diffeqsolve`, allowing for
    pre-configuration of a `diffrax.AbstractSolver`,
    `diffrax.AbstractStepSizeController`, and `diffrax.AbstractAdjoint`.
    Pre-configuring these objects can be useful when you want to:

    - repeatedly solve similar differential equations and can reuse the same
       solver, step size controller, and adjoint.
    - pass the differential equation solver as an argument to a function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.dynamics.integrate import DiffEqSolver

    Construct a solver object.

    >>> solver = DiffEqSolver(dfx.Dopri5(),
    ...                stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))

    And a differential equation to solve.

    >>> term = dfx.ODETerm(lambda t, y, args: -y)

    Then solve the differential equation.

    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=f64[1], ... )

    The solution can be saved at specific times.

    >>> saveat = dfx.SaveAt(ts=[0., 1., 2., 3.])
    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[4],
              ys=f64[4], ... )

    The solution can be densely interpolated.

    >>> saveat = dfx.SaveAt(t1=True, dense=True)
    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=f64[1], ... )
    >>> soln.evaluate(0.5)
    Array(0.60653213, dtype=float64)

    Using the `VectorizedDenseInterpolation` class, the interpolation can be
    vectorized, enabling evaluation of batched solutions over batches of times.

    >>> from galax.dynamics.integrate import VectorizedDenseInterpolation
    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
    >>> soln = VectorizedDenseInterpolation.apply_to_solution(soln)
    >>> soln.evaluate(jnp.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2))
    Array([[0.90483742, 0.81872516],
           [0.74080871, 0.67031456]], dtype=float64)

    """

    #: The solver for the differential equation.
    #: See the diffrax guide on how to choose a solver.
    solver: dfx.AbstractSolver

    _: KW_ONLY

    #: How to change the step size as the integration progresses.
    #: See diffrax's list of stepsize controllers.
    stepsize_controller: dfx.AbstractStepSizeController = eqx.field(
        default=default_stepsize_controller
    )

    #: How to differentiate `diffeqsolve`.
    #: See `diffrax` for options.
    adjoint: dfx.AbstractAdjoint = eqx.field(default=default_adjoint)

    # TODO: should `max_steps` be a field? Given that `max_steps=None` can be
    # incompatible with some `SaveAt` options, it would still need to be
    # overridable in `__call__`.

    @partial(eqx.filter_jit)
    # @partial(quax.quaxify)  # TODO: so don't need to strip units
    def __call__(
        self: "DiffEqSolver",
        terms: PyTree[dfx.AbstractTerm],
        /,
        t0: RealSz0Like,
        t1: RealSz0Like,
        dt0: RealSz0Like | None,
        y0: PyTree[ArrayLike],
        args: PyTree[Any] = None,
        *,
        # Diffrax options
        saveat: dfx.SaveAt = default_saveat,
        event: dfx.Event | None = default_event,
        max_steps: int | None = default_max_steps,
        throw: bool = default_throw,
        progress_meter: dfx.AbstractProgressMeter = default_progress_meter,
        solver_state: PyTree[ArrayLike] | None = None,
        controller_state: PyTree[ArrayLike] | None = None,
        made_jump: BoolSz0Like | None = None,
        # Extra options
        _vectorize_interpolation: bool = False,  # TODO: make public
    ) -> dfx.Solution:
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
        soln: dfx.Solution = dfx.diffeqsolve(
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
        if _vectorize_interpolation and soln.interpolation is not None:
            soln = VectorizedDenseInterpolation.apply_to_solution(soln)
        return soln

    # TODO: a contextmanager for producing a temporary DiffEqSolver with
    # different field values.
