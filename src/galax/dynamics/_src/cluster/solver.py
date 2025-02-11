"""``galax`` dynamics."""

__all__ = ["MassSolver"]

from dataclasses import KW_ONLY
from typing import Any

import diffrax as dfx
import equinox as eqx
import optimistix as optx
from jaxtyping import PyTree
from plum import dispatch

import diffraxtra as dfxtra
import unxt as u

import galax.typing as gt
from .events import MassBelowThreshold
from .fields import AbstractMassField, CustomMassField, FieldArgs, MassVectorField
from galax.dynamics._src.compat import AllowValue
from galax.dynamics._src.solver import AbstractSolver, SolveState
from galax.dynamics._src.utils import parse_saveat


class MassSolver(AbstractSolver, strict=True):  # type: ignore[call-arg]
    """Solver for mass history.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.dynamics as gd

    >>> event = dfx.Event(gd.cluster.MassBelowThreshold(u.Quantity(0.0, "Msun")))
    >>> mass_solver = gd.cluster.MassSolver(event=event)

    >>> mass_field = lambda t, Mc, args: -2e5 / (t + 1)
    >>> Mc0 = u.Quantity(1e6, "Msun")
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> saveat = jnp.linspace(t0, t1, 10)
    >>> mass_soln = mass_solver.solve(mass_field, Mc0, t0, t1, saveat=saveat)
    >>> mass_soln.ys
    Array([1000000. , 56101.91157639, inf, inf, ...], dtype=float64)

    """

    diffeqsolver: dfxtra.DiffEqSolver = eqx.field(
        default=dfxtra.DiffEqSolver(
            solver=dfx.Dopri8(),
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8),
            max_steps=2**16,
        ),
        converter=dfxtra.DiffEqSolver.from_,
    )
    # TODO: should events be incorporated into `DiffEqSolver`?
    event: dfx.Event | None = eqx.field(
        default=dfx.Event(
            cond_fn=MassBelowThreshold(u.Quantity(0.0, "Msun")),
            root_finder=optx.Newton(1e-5, 1e-5, optx.rms_norm),
        )
    )

    _: KW_ONLY

    units: u.AbstractUnitSystem = eqx.field(
        default=u.unitsystems.galactic, converter=u.unitsystem, static=True
    )

    # -----------------------

    @dispatch.abstract
    def init(self: "MassSolver", field: Any, y0: Any, t0: Any, args: Any) -> Any:
        # See dispatches below
        raise NotImplementedError  # pragma: no cover

    def step(
        self,
        field: AbstractMassField,
        state: SolveState,
        t1: Any,
        /,
        args: PyTree,
        **step_kwargs: Any,  # e.g. solver_state, made_jump
    ) -> SolveState:
        """Step the state."""
        terms = field.terms(self.diffeqsolver)
        t1_ = u.ustrip(AllowValue, self.units["time"], t1)
        step_kwargs.setdefault("made_jump", False)
        return self._step_impl(terms, state, t1_, args, step_kwargs)

    @dispatch.abstract
    def run(
        self, field: Any, state: SolveState, t1: Any, args: PyTree, **solver_kw: Any
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
    field: AbstractMassField,
    Mc0: gt.RealQuSz0,
    t0: gt.RealQuSz0,
    args: PyTree,
    /,
) -> SolveState:
    terms = field.terms(self.diffeqsolver)
    Mc0_ = u.ustrip(AllowValue, self.units["mass"], Mc0)
    t0_ = u.ustrip(AllowValue, self.units["time"], t0)
    return self._init_impl(terms, t0_, Mc0_, args, self.units)


# ===================================================================
# Run Dispatches


@MassSolver.run.dispatch  # type: ignore[misc]
def run(
    self: MassSolver,
    field: AbstractMassField,
    state: SolveState,
    t1: Any,
    args: PyTree,
    /,
    **solver_kw: Any,
) -> SolveState:
    terms = field.terms(self.diffeqsolver)
    t1_ = u.ustrip(AllowValue, self.units["time"], t1)
    solver_kw = eqx.error_if(
        solver_kw, "saveat" in solver_kw, "`saveat` is not allowed in run"
    )
    soln = self._run_impl(terms, state, t1_, args, solver_kw)
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
    field: AbstractMassField | MassVectorField,
    M0: Any,
    t0: Any,
    t1: Any,
    /,
    saveat: Any = default_saveat,
    **solver_kw: Any,
) -> dfx.Solution:
    # Setup
    units = self.units
    field_ = field if isinstance(field, AbstractMassField) else CustomMassField(field)

    # Solve the differential equation
    solver_kw.setdefault("dt0", None)
    saveat = parse_saveat(units, saveat, dense=solver_kw.pop("dense", None))
    args = solver_kw.pop("args", {})
    args.setdefault("units", units)  # TODO: should this error if not right?
    soln = self.diffeqsolver(
        field_.terms(self.diffeqsolver),
        t0=u.ustrip(AllowValue, units["time"], t0),
        t1=u.ustrip(AllowValue, units["time"], t1),
        y0=u.ustrip(AllowValue, units["mass"], M0),
        event=self.event,
        args=args,
        saveat=saveat,
        **solver_kw,
    )

    return soln  # noqa: RET504
