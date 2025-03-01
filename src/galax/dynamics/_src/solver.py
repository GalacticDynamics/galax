"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractSolver", "SolveState"]


import abc
from dataclasses import fields
from functools import partial
from typing import Any, TypeAlias

import diffrax as dfx
import equinox as eqx
import numpy as np
from jaxtyping import Array, PyTree, Real

import diffraxtra as dfxtra
import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt

USys: TypeAlias = u.AbstractUnitSystem
DenseInfo: TypeAlias = dict[str, PyTree[Array]]
Terms: TypeAlias = PyTree
DfxRealScalarLike: TypeAlias = Real[int | float | Array | np.ndarray[Any, Any], ""]

# =========================================================
# SolveState


class SolveState(eqx.Module, strict=True):  # type: ignore[misc, call-arg]
    """State of the solver.

    This is used as the return value for `galax.dynamics.AbstractSolver.init`
    and `galax.dynamics.AbstractSolver.step`. It is used as the argument to
    `galax.dynamics.AbstractSolver.step`, `galax.dynamics.AbstractSolver.run`,
    and can also be passed to `galax.dynamics.AbstractSolver.solve`.

    """

    #: Current time.
    t: gt.Sz0

    # ---- diffrax step outputs ----
    #: Current solution at `t`.
    y: PyTree

    # TODO: figure out how to extract this from `diffrax.Solution`
    # #: A local error estimate made during the step
    # err: PyTree | None  # noqa: ERA001

    # TODO: figure out how to extract this from `diffrax.Solution`
    # #: Save information. This is a dictionary of information that is passed to
    # # the solver's interpolation routine to calculate dense output. (Used with
    # # SaveAt(ts=...) or SaveAt(dense=...).)
    # save_info: DenseInfo  # noqa: ERA001
    #: The value of the solver state at t1.
    solver_state: Any

    #: Step success. An integer (corresponding to diffrax.RESULTS) indicating
    # whether the step happened successfully, or if (unusually) it failed for
    # some reason.
    success: dfx.RESULTS

    # --- reconstruction info ---
    units: USys = eqx.field(static=True)

    @classmethod
    def from_step_output(
        cls,
        t: Any,
        obj: tuple[PyTree, PyTree | None, DenseInfo, Any, dfx.RESULTS],
        units: USys,
        /,
    ) -> "SolveState":
        return cls(
            t=t,
            y=obj[0],
            # err=obj[1],  # noqa: ERA001
            # save_info=obj[2],  # noqa: ERA001
            solver_state=obj[3],
            success=obj[4],
            units=units,
        )


# =========================================================
# Abstract Solver


class AbstractSolver(dfxtra.AbstractDiffEqSolver, strict=True):  # type: ignore[call-arg,misc]
    """ABC for solvers.

    Notes
    -----
    The ``init``, ``step``, and ``solve`` methods are abstract and should be
    implemented by subclasses.

    """

    @partial(jnp.vectorize, excluded=(0, 1, 3, 4, 5))
    @partial(eqx.filter_jit)
    def _init_impl(
        self, terms: Terms, t0: gt.SzAny, y0: PyTree, args: Any, units: USys, /
    ) -> SolveState:
        """`init` helper."""
        # Initializes the state from diffrax. Steps from t0 to t0!
        solver_state = self.solver.init(terms, t0, t0, y0, args)
        # Step from t0 to t0, which is a no-op but sets the state
        step_output = self.solver.step(
            terms, t0, t0, y0, args=args, solver_state=solver_state, made_jump=False
        )
        return SolveState.from_step_output(t0, step_output, units)

    @abc.abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> SolveState:
        """Initialize the solver."""
        raise NotImplementedError

    # -----------------------

    def _step_impl_scalar(
        self,
        terms: Terms,
        state: SolveState,
        t1: gt.Sz0,
        args: Any,
        step_kw: dict[str, Any],
    ) -> SolveState:
        t0 = state.t
        t0 = eqx.error_if(t0, t0.ndim != 0, "t0 must be a scalar")
        step_output = self.solver.step(
            terms,
            t0,
            t1,
            state.y,
            args=args,
            solver_state=state.solver_state,
            **step_kw,
        )
        return SolveState.from_step_output(t1, step_output, state.units)

    @abc.abstractmethod
    def step(
        self,
        terms: Any,
        state: SolveState,
        t1: Any,
        /,
        args: PyTree,
        **step_kwargs: Any,  # e.g. solver_state, made_jump
    ) -> SolveState:
        """Step the solver."""
        raise NotImplementedError

    # ----------------

    @abc.abstractmethod
    def run(
        self, terms: Any, state: SolveState, t1: Any, args: Any, **solver_kw: Any
    ) -> SolveState:
        """Run the solver."""
        raise NotImplementedError  # pragma: no cover

    # ----------------

    @abc.abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Solve, initializing and stepping to the solution."""
        raise NotImplementedError


# =========================================================
# Constructors


@AbstractSolver.from_.dispatch  # type: ignore[misc]
def from_(cls: type[AbstractSolver], solver: dfxtra.DiffEqSolver) -> AbstractSolver:
    """Create a new solver from a `diffraxtra.DiffeqSolver`.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import diffraxtra as dfxtra
    >>> import galax.dynamics as gd

    >>> solver = dfxtra.DiffEqSolver(dfx.Dopri5())

    >>> new_solver = gd.OrbitSolver.from_(solver)
    >>> new_solver
    OrbitSolver(
      solver=Dopri5(scan_kind=None),
      stepsize_controller=ConstantStepSize(),
      adjoint=RecursiveCheckpointAdjoint(checkpoints=None),
      event=None,
      max_steps=4096
    )

    """
    return cls(**{f.name: getattr(solver, f.name) for f in fields(cls)})
