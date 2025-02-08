"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractSolver", "SolveState"]


import abc
from collections.abc import Mapping
from typing import Any, TypeAlias

import diffrax as dfx
import equinox as eqx
import numpy as np
from jaxtyping import Array, PyTree, Real
from plum import dispatch

import unxt as u

import galax.typing as gt

DenseInfo: TypeAlias = dict[str, PyTree[Array]]
Terms: TypeAlias = PyTree
DfxRealScalarLike: TypeAlias = Real[int | float | Array | np.ndarray[Any, Any], ""]


class SolveState(eqx.Module, strict=True):  # type: ignore[misc, call-arg]
    """State of the solver.

    This is used as the return value for `galax.dynamics.AbstractSolver.init`
    and `galax.dynamics.AbstractSolver.step`. It is used as the argument to
    `galax.dynamics.AbstractSolver.step`, `galax.dynamics.AbstractSolver.run`,
    and can also be passed to `galax.dynamics.AbstractSolver.solve`.

    """

    #: Current time.
    t: Any
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
    units: u.AbstractUnitSystem

    @classmethod
    def from_step_output(
        cls,
        t: Any,
        obj: tuple[PyTree, PyTree | None, DenseInfo, Any, dfx.RESULTS],
        units: u.AbstractUnitSystem,
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


class AbstractSolver(eqx.Module, strict=True):  # type: ignore[call-arg,misc]
    """ABC for solvers.

    Notes
    -----
    The ``init``, ``step``, and ``solve`` methods are abstract and should be
    implemented by subclasses.

    """

    def _init_impl(
        self,
        terms: Terms,
        t0: DfxRealScalarLike,
        y0: PyTree,
        args: Any,
        units: u.AbstractUnitSystem,
    ) -> SolveState:
        """`init` helper."""
        # Initializes the state from diffrax. Steps from t0 to t0!
        solver_state = self.diffeqsolver.solver.init(terms, t0, t0, y0, args)
        # Step from t0 to t0, which is a no-op but sets the state
        step_output = self.diffeqsolver.solver.step(
            terms, t0, t0, y0, args=args, solver_state=solver_state, made_jump=False
        )
        return SolveState.from_step_output(t0, step_output, units)

    @abc.abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> SolveState:
        """Initialize the solver."""
        raise NotImplementedError

    # -----------------------

    def _step_impl(
        self,
        terms: Terms,
        state: SolveState,
        t1: gt.RealSz0,
        args: Any,
        step_kw: dict[str, Any],
    ) -> SolveState:
        step_output = self.diffeqsolver.solver.step(
            terms,
            state.t,
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

    def _run_impl(
        self,
        terms: PyTree,
        state: SolveState,
        t1: DfxRealScalarLike,
        /,
        args: PyTree,
        solver_kw: dict[str, Any],
    ) -> dfx.Solution:
        solver_kw.setdefault("dt0", None)
        return self.diffeqsolver(
            terms,
            t0=state.t,
            t1=t1,
            y0=state.y,
            args=args,
            **solver_kw,
        )

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

    # ==================================================================
    # Convenience methods

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractSolver]", *args: Any, **kwargs: Any
    ) -> "AbstractSolver":
        """Create a new solver from the argument."""
        raise NotImplementedError  # pragma: no cover


# =========================================================
# Constructors


@AbstractSolver.from_.dispatch
def from_(cls: type[AbstractSolver], solver: AbstractSolver) -> AbstractSolver:
    """Create a new solver from the argument.

    Examples
    --------
    >>> import galax.dynamics as gd
    >>> solver = gd.DynamicsSolver()
    >>> new_solver = gd.DynamicsSolver.from_(solver)
    >>> new_solver is solver
    True

    >>> class MySolver(AbstractSolver):
    ...     def init(self, *args, **kwargs): pass
    ...     def step(self, *args, **kwargs): pass
    ...     def solve(self, *args, **kwargs): pass
    >>> try: new_solver = MySolver.from_(solver)
    ... except TypeError as e: print(e)
    Cannot convert <class 'galax.dynamics...DynamicsSolver'> to <class '...MySolver'>

    """
    if not isinstance(solver, cls):
        msg = f"Cannot convert {type(solver)} to {cls}"
        raise TypeError(msg)

    return solver


@AbstractSolver.from_.dispatch(precedence=-1)
def from_(cls: type[AbstractSolver], obj: Any) -> AbstractSolver:
    """Pass argument to solver's init.

    Examples
    --------
    >>> import diffrax as dfx
    >>> from galax.dynamics.solve import DynamicsSolver, DiffEqSolver

    >>> DynamicsSolver.from_( DiffEqSolver(dfx.Dopri5()))
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None),
        max_steps=4096
      )
    )

    >>> DynamicsSolver.from_(dfx.Dopri5())
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None),
        max_steps=4096
      )
    )

    """
    return cls(obj)


@AbstractSolver.from_.dispatch
def from_(cls: type[AbstractSolver], obj: Mapping[str, Any]) -> AbstractSolver:
    """Create a new solver from the argument.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import galax.dynamics as gd

    >>> gd.DynamicsSolver.from_({})
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri8(scan_kind=None),
        stepsize_controller=PIDController( ...),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None),
        max_steps=65536
      )
    )

    >>> gd.DynamicsSolver.from_({"diffeqsolver": dfx.Dopri5()})
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None),
        max_steps=4096
      )
    )

    """
    return cls(**obj)
