"""``galax`` dynamics."""

__all__ = ["MassSolver"]

from dataclasses import KW_ONLY

from collections.abc import Mapping
from jaxtyping import PyTree
from typing import Any, override

import diffrax as dfx
import equinox as eqx
import optimistix as optx
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.dynamics.custom_types as gt
from .dmdt import AbstractMassRateField, CustomMassRateField, FieldArgs, MassVectorField
from .events import MassBelowThreshold
from galax.fields import AbstractSolver, SolveState
from galax.fields.utils import parse_saveat


class MassSolver(AbstractSolver):
    """Solver for mass history.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.dynamics as gd

    >>> mass_solver = gd.cluster.MassSolver()

    >>> mass_field = lambda t, Mc, args: -2e5 / (t + 1)
    >>> Mc0 = u.Q(1e6, "Msun")
    >>> t0, t1 = u.Q(0, "Gyr"), u.Q(1, "Gyr")
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
    #: With this default, `atol` is scaled down for a starting mass below 1
    #: (in the field's mass unit), so a tiny cluster doesn't overshoot below 0.
    stepsize_controller: dfx.AbstractStepSizeController[Any, Any] = dfx.PIDController(
        rtol=1e-6, atol=1e-6
    )

    #: How to differentiate in `diffeqsolve`.
    #: See `diffrax` for options.
    adjoint: dfx.AbstractAdjoint = dfx.ForwardMode()

    #: Event. Can override the `event` argument when calling `DiffEqSolver`
    event: dfx.Event | None = dfx.Event(
        cond_fn=MassBelowThreshold(u.Q(0.0, "Msun")),
        root_finder=optx.Newton(1e-5, 1e-5, optx.rms_norm),
    )

    #: The maximum number of steps to take before quitting.
    #: Some `diffrax.SaveAt` options can be incompatible with `max_steps=None`,
    #: so you can override the `max_steps` argument when calling `DiffEqSolver`
    max_steps: int | None = eqx.field(default=2**12, static=True)

    #: The units a bare function passed to `solve` is wrapped in, as a
    #: `CustomMassRateField`. A mass field already carries its own units, and
    #: times, masses and a Quantity ``dt0`` are read in those.
    units: u.AbstractUnitSystem = eqx.field(
        default=u.unitsystems.galactic, converter=u.unitsystem, static=True
    )

    # -----------------------

    @override
    @dispatch.abstract
    def init(self: "MassSolver", dm_dt: Any, y0: Any, t0: Any, args: Any, /) -> Any:  # type: ignore[override]
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
        t1_ = u.ustrip(AllowValue, dm_dt.units["time"], t1)
        step_kwargs.setdefault("made_jump", False)
        return self._step_impl_scalar(dm_dt, state, t1_, args, step_kwargs)

    @override
    @dispatch.abstract
    def run(  # type: ignore[override]
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


@MassSolver.init.dispatch
def init(
    self: MassSolver,
    field: AbstractMassRateField,
    Mc0: gt.QuSz0,
    t0: gt.QuSz0,
    args: PyTree,
    /,
) -> SolveState:
    Mc0_ = u.ustrip(AllowValue, field.units["mass"], Mc0)
    t0_ = u.ustrip(AllowValue, field.units["time"], t0)
    state: SolveState = self._init_impl(field, t0_, Mc0_, args, field.units)
    return state


def _event_args(args: PyTree, units: u.AbstractUnitSystem, y0: Any) -> PyTree:
    """Give the default event the field's units and a mass scale in ``args``.

    The mass is solved in the field's units, so those win over any the caller
    passed. ``mass_scale`` is the starting mass in those units; the event
    divides by the larger of it and the threshold, so the root finder's
    absolute tolerance acts as a relative one. The caller's dict is not
    changed. ``None`` and an empty tuple or list count as no args. Other
    ``args`` (for a custom event) are passed through as they are.
    """
    if args is None or (isinstance(args, tuple | list) and len(args) == 0):
        args = {}
    if isinstance(args, Mapping):
        return {**args, "units": units, "mass_scale": jnp.max(jnp.abs(y0))}
    return args


_DEFAULT_CONTROLLER = dfx.PIDController(rtol=1e-6, atol=1e-6)


def _for_mass(solver: "MassSolver", y0: Any) -> "MassSolver":
    """Scale the default controller's ``atol`` down for a starting mass below 1.

    ``atol`` is in the field's mass unit, so 1e-6 lets a cluster of ~1e-6
    overshoot below 0. From a mass of 1 up nothing changes. A custom
    controller is used as given.
    """
    ctrl = solver.stepsize_controller
    if not isinstance(ctrl, dfx.PIDController) or not eqx.tree_equal(
        ctrl, _DEFAULT_CONTROLLER
    ):
        return solver
    m = jnp.max(jnp.abs(y0))
    scale = jnp.where(m > 0, jnp.minimum(m, 1), 1)
    out: MassSolver = eqx.tree_at(
        lambda s: s.stepsize_controller.atol, solver, ctrl.atol * scale
    )
    return out


# ===================================================================
# Run Dispatches


@MassSolver.run.dispatch
def run(
    self: MassSolver,
    field: AbstractMassRateField,
    state: SolveState,
    t1: Any,
    args: PyTree,
    /,
    **solver_kw: Any,
) -> SolveState:
    t1_ = u.ustrip(AllowValue, field.units["time"], t1)
    solver_kw = eqx.error_if(
        solver_kw, "saveat" in solver_kw, "`saveat` is not allowed in run"
    )
    if "dense" in solver_kw:
        msg = (
            "`run` returns only the final state. For the mass history use "
            "`solve(..., dense=True)`."
        )
        raise TypeError(msg)
    solver_kw.setdefault("dt0", None)
    args = _event_args(args, field.units, state.y)
    soln = _for_mass(self, state.y)(
        field, t0=state.t, t1=t1_, y0=state.y, args=args, **solver_kw
    )

    # The event time, if the event stopped the solve, and a scalar mass, like
    # `init`'s, so `run` can be called again. The solver state is rebuilt
    # there, so `step` can continue from it: diffrax's saved state is from
    # the end of the last step, past the event.
    out = self._init_one(field, soln.ts[-1], soln.ys[-1], args, field.units)
    final: SolveState = eqx.tree_at(lambda s: s.success, out, soln.result)
    return final


# ===================================================================
# Solve Dispatches


default_saveat = dfx.SaveAt(t1=True)


@MassSolver.solve.dispatch
@eqx.filter_jit
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
    # Setup. `t0`, `t1`, `M0` and a Quantity `dt0` are read in the field's units,
    # as the shared field dispatch does. `self.units` only sets the units a
    # bare function is wrapped in.
    field_ = (
        field
        if isinstance(field, AbstractMassRateField)
        else CustomMassRateField(field, units=self.units)
    )
    units = field_.units

    # Solve the differential equation
    solver_kw.setdefault("dt0", None)
    saveat = parse_saveat(units, saveat, dense=solver_kw.pop("dense", None))
    y0 = u.ustrip(AllowValue, units["mass"], M0)
    args = _event_args(solver_kw.pop("args", None), units, y0)
    soln = _for_mass(self, y0)(
        field_,
        t0=u.ustrip(AllowValue, units["time"], t0),
        t1=u.ustrip(AllowValue, units["time"], t1),
        y0=y0,
        event=self.event,
        args=args,
        saveat=saveat,
        **solver_kw,
    )

    return soln
