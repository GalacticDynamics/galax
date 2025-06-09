"""``galax`` dynamics."""

__all__ = ["MassSolver"]

from dataclasses import KW_ONLY
from typing import Any

import diffrax as dfx
import equinox as eqx
import optimistix as optx
from jaxtyping import PyTree
from plum import dispatch

import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from .dmdt import (
    AbstractMassRateField,
    CustomMassRateField,
    FieldArgs,
    MassVectorField,
)
from .events import MassBelowThreshold
from galax.dynamics._src.solver import AbstractSolver, SolveState
from galax.dynamics._src.utils import parse_saveat


class MassSolver(AbstractSolver, strict=True):  # type: ignore[call-arg]
    """Solver for mass history.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.dynamics as gd

    >>> mass_solver = gd.cluster.MassSolver()

    >>> mass_field = lambda t, Mc, args: -2e5 / (t + 1)
    >>> Mc0 = u.Quantity(1e6, "Msun")
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> saveat = jnp.linspace(t0, t1, 10)
    >>> mass_soln = mass_solver.solve(mass_field, Mc0, t0, t1, saveat=saveat)
    >>> mass_soln.ys
    Array([1000000. , 56101.9173497, inf, inf, ...], dtype=float64)

    """

    #: The solver for the differential equation.
    #: See the diffrax guide on how to choose a solver.
    solver: dfx.AbstractSolver[Any] = dfx.Dopri8()

    _: KW_ONLY

    #: How to change the step size as the integration progresses.
    #: See diffrax's list of stepsize controllers.
    stepsize_controller: dfx.AbstractStepSizeController[Any, Any] = dfx.PIDController(
        rtol=1e-6, atol=1e-6
    )

    #: How to differentiate in `diffeqsolve`.
    #: See `diffrax` for options.
    adjoint: dfx.AbstractAdjoint = dfx.ForwardMode()

    #: Event. Can override the `event` argument when calling `DiffEqSolver`
    event: dfx.Event | None = dfx.Event(
        cond_fn=MassBelowThreshold(u.Quantity(0.0, "Msun")),
        root_finder=optx.Newton(1e-5, 1e-5, optx.rms_norm),
    )

    #: The maximum number of steps to take before quitting.
    #: Some `diffrax.SaveAt` options can be incompatible with `max_steps=None`,
    #: so you can override the `max_steps` argument when calling `DiffEqSolver`
    max_steps: int | None = eqx.field(default=2**12, static=True)

    units: u.AbstractUnitSystem = eqx.field(
        default=u.unitsystems.galactic, converter=u.unitsystem, static=True
    )

    # -----------------------

    @dispatch.abstract
    def init(self: "MassSolver", dm_dt: Any, y0: Any, t0: Any, args: Any, /) -> Any:
        # See dispatches below
        raise NotImplementedError  # pragma: no cover

    def step(
        self,
        dm_dt: AbstractMassRateField,
        state: SolveState,
        t1: Any,
        /,
        args: PyTree,
        **step_kwargs: Any,  # e.g. solver_state, made_jump
    ) -> SolveState:
        """Step the state."""
        t1_ = u.ustrip(AllowValue, self.units["time"], t1)
        step_kwargs.setdefault("made_jump", False)
        return self._step_impl_scalar(dm_dt, state, t1_, args, step_kwargs)

    @dispatch.abstract
    def run(
        self, dm_dt: Any, state: SolveState, t1: Any, args: PyTree, /, **solver_kw: Any
    ) -> SolveState:
        """Run from the state."""
        raise NotImplementedError  # pragma: no cover

    # TODO: dispatch where the state from `init` is accepted
    @dispatch.abstract
    def solve(
        self: "MassSolver",
        field: Any,
        state: Any,
        t0: Any,
        t1: Any,
        /,
        args: FieldArgs = {},  # noqa: B006
        **solver_kw: Any,  # TODO: TypedDict
    ) -> dfx.Solution:
        """Call `diffrax.diffeqsolve`."""
        raise NotImplementedError  # pragma: no cover


# ===================================================================
# Init Dispatches


@MassSolver.init.dispatch  # type: ignore[misc]
def init(
    self: MassSolver,
    field: AbstractMassRateField,
    Mc0: gt.QuSz0,
    t0: gt.QuSz0,
    args: PyTree,
    /,
) -> SolveState:
    Mc0_ = u.ustrip(AllowValue, self.units["mass"], Mc0)
    t0_ = u.ustrip(AllowValue, self.units["time"], t0)
    state: SolveState = self._init_impl(field, t0_, Mc0_, args, self.units)
    return state


# ===================================================================
# Run Dispatches


@MassSolver.run.dispatch  # type: ignore[misc]
def run(
    self: MassSolver,
    field: AbstractMassRateField,
    state: SolveState,
    t1: Any,
    args: PyTree,
    /,
    **solver_kw: Any,
) -> SolveState:
    t1_ = u.ustrip(AllowValue, self.units["time"], t1)
    solver_kw = eqx.error_if(
        solver_kw, "saveat" in solver_kw, "`saveat` is not allowed in run"
    )
    solver_kw.setdefault("dt0", None)
    soln = self(field, t0=state.t, t1=t1_, y0=state.y, args=args, **solver_kw)

    return SolveState(
        t=t1_,
        y=soln.ys,
        solver_state=soln.solver_state,
        success=soln.result,
        units=self.units,
    )


# ===================================================================
# Solve Dispatches


default_saveat = dfx.SaveAt(t1=True)


@MassSolver.solve.dispatch  # type: ignore[misc]
@eqx.filter_jit  # type: ignore[misc]
def solve(
    self: MassSolver,
    field: AbstractMassRateField | MassVectorField,
    M0: Any,
    t0: Any,
    t1: Any,
    /,
    saveat: Any = default_saveat,
    **solver_kw: Any,
) -> dfx.Solution:
    # Setup
    units = self.units
    field_ = (
        field
        if isinstance(field, AbstractMassRateField)
        else CustomMassRateField(field, units=units)
    )

    # Solve the differential equation
    solver_kw.setdefault("dt0", None)
    saveat = parse_saveat(units, saveat, dense=solver_kw.pop("dense", None))
    args = solver_kw.pop("args", {})
    args.setdefault("units", units)  # TODO: should this error if not right?
    soln = self(
        field_,
        t0=u.ustrip(AllowValue, units["time"], t0),
        t1=u.ustrip(AllowValue, units["time"], t1),
        y0=u.ustrip(AllowValue, units["mass"], M0),
        event=self.event,
        args=args,
        saveat=saveat,
        **solver_kw,
    )

    return soln
