"""``galax`` dynamics."""

__all__ = ["MassSolver"]

from dataclasses import KW_ONLY
from typing import Any

import diffrax as dfx
import equinox as eqx
import optimistix as optx
from plum import dispatch

import diffraxtra as dfxtra
import unxt as u
from unxt.quantity import AbstractQuantity

from .events import MassBelowThreshold
from .fields import AbstractMassField, MassVectorField, UserMassField
from galax.dynamics._src.solver import AbstractSolver
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
        ),
        converter=dfxtra.DiffEqSolver.from_,
    )
    # TODO: should this be incorporated into in `DiffEqSolver`?
    event: dfx.Event = eqx.field(
        default=dfx.Event(
            cond_fn=MassBelowThreshold(u.Quantity(0.0, "Msun")),
            root_finder=optx.Newton(1e-5, 1e-5, optx.rms_norm),
        )
    )

    _: KW_ONLY

    units: u.AbstractUnitSystem = eqx.field(
        default=u.unitsystems.galactic, converter=u.unitsystem, static=True
    )

    @dispatch.abstract
    def init(
        self: "MassSolver", terms: Any, t0: Any, t1: Any, y0: Any, args: Any
    ) -> Any:
        # See dispatches below
        raise NotImplementedError  # pragma: no cover

    @dispatch.abstract
    def step(
        self: "MassSolver",
        terms: Any,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
        **step_kwargs: Any,  # e.g. solver_state, made_jump
    ) -> Any:
        """Step the state."""
        # See dispatches below
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
        args: Any = (),
        **solver_kw: Any,  # TODO: TypedDict
    ) -> dfx.Solution:
        """Call `diffrax.diffeqsolve`."""
        raise NotImplementedError  # pragma: no cover


# ===================================================================
# Solve Dispatches


default_saveat = dfx.SaveAt(t1=True)


@MassSolver.solve.dispatch  # type: ignore[misc]
@eqx.filter_jit  # type: ignore[misc]
def solve(
    self: MassSolver,
    field: AbstractMassField | MassVectorField,
    state: Any,
    t0: AbstractQuantity,
    t1: AbstractQuantity,
    /,
    saveat: Any = default_saveat,
    **solver_kw: Any,
) -> dfx.Solution:
    # Setup
    units = self.units
    field_ = field if isinstance(field, AbstractMassField) else UserMassField(field)

    # Initial conditions
    y0 = state.ustrip(units["mass"])  # Mc

    # Solve the differential equation
    solver_kw.setdefault("dt0", None)
    saveat = parse_saveat(units, saveat, dense=solver_kw.pop("dense", None))
    soln = self.diffeqsolver(
        field_.terms(self.diffeqsolver),
        t0=t0.ustrip(units["time"]),
        t1=t1.ustrip(units["time"]),
        y0=y0,
        event=self.event,
        args={"units": units},
        saveat=saveat,
        **solver_kw,
    )

    return soln  # noqa: RET504
