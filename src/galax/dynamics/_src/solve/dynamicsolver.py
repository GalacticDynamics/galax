"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractSolver", "DynamicsSolver"]


import abc
from typing import Any

import diffrax
import equinox as eqx
from plum import dispatch

import galax.coordinates as gc
import galax.typing as gt
from .diffeqsolver import DiffEqSolver
from .utils import converter_diffeqsolver
from galax.dynamics._src.fields import AbstractDynamicsField


class AbstractSolver(eqx.Module, strict=True):  # type: ignore[call-arg,misc]
    """ABC for solvers.

    Notes
    -----
    The ``init``, ``step``, and ``solve`` methods are abstract and should be
    implemented by subclasses.

    """

    @abc.abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize the solver."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Step the solver."""
        raise NotImplementedError

    @abc.abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Solve, initializing and stepping to the solution."""
        raise NotImplementedError


##############################################################################


class DynamicsSolver(AbstractSolver, strict=True):  # type: ignore[call-arg]
    """Dynamics solver.

    The most useful method is `.solve`, which handles initialization and
    stepping to the final solution.
    Manual solves can be done with `.init()` and repeat `.step()`.

    """

    #: Solver for the differential equation.
    diffeqsolver: DiffEqSolver = eqx.field(
        default=DiffEqSolver(
            solver=diffrax.Dopri8(),
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        ),
        converter=converter_diffeqsolver,
    )

    @dispatch.abstract
    def init(
        self: "DynamicsSolver", term: Any, t0: Any, t1: Any, y0: Any, args: Any
    ) -> Any:
        # See dispatches below
        raise NotImplementedError  # pragma: no cover

    @dispatch.abstract
    def step(
        self: "DynamicsSolver",
        term: Any,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
        **step_kwargs: Any,  # e.g. solver_state, made_jump
    ) -> Any:
        """Step the state."""
        # See dispatches below
        raise NotImplementedError  # pragma: no cover

    # TODO: decide on the output type
    # TODO: dispatch where the state from `init` is accepted
    @dispatch.abstract
    def solve(
        self: "DynamicsSolver",
        field: Any,
        t0: Any,
        t1: Any,
        w0: Any,
        /,
        args: Any = (),
        **solver_kw: Any,  # TODO: TypedDict
    ) -> diffrax.Solution:
        """Call `diffrax.diffeqsolve`."""
        raise NotImplementedError  # pragma: no cover


# --------------------------------


@DynamicsSolver.solve.dispatch  # type: ignore[misc]
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    t0: gt.TimeScalar,
    t1: gt.TimeScalar,
    w0: gc.PhaseSpacePosition,  # TODO: handle frames
    /,
    args: Any = (),
    **solver_kw: Any,  # TODO: TypedDict
) -> diffrax.Solution:
    """Solve for scalar time, position.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([8, 0, 9], "kpc"),
    ...                            p=u.Quantity([0, 220, 0], "km/s"))

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, t0, t1, w0)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )

    """
    units = field.units
    time = units["time"]

    w0 = eqx.error_if(
        w0,
        not isinstance(w0.frame, gc.frames.SimulationFrame),
        "Only SimulationFrame is currently supported.",
    )
    qp = w0._qp(units=units)  # noqa: SLF001
    y0 = (qp[0].ustrip(units["length"]), qp[1].ustrip(units["speed"]))

    soln = self.diffeqsolver(
        field.terms,
        t0=t0.ustrip(time),
        t1=t1.ustrip(time),
        dt0=None,
        y0=y0,
        args=args,
        max_steps=None,
        **solver_kw,
    )

    # TODO: if t1=True should this remove the scalar time dimension?

    return soln  # noqa: RET504
