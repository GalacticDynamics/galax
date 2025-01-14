"""Dynamics Solvers.

This is private API.

"""

__all__ = ["DynamicsSolver"]


from typing import Any

import diffrax
import equinox as eqx
import jax.numpy as jnp
from plum import convert, dispatch

import coordinax as cx
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc
import galax.typing as gt
from .base import AbstractSolver
from .diffeq import DiffEqSolver
from .utils import converter_diffeqsolver, parse_saveat
from galax.dynamics._src.fields import AbstractDynamicsField


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


# ===============================================
# Solve Dispatches

default_saveat = diffrax.SaveAt(t1=True)

# --------------------------------
# JAX & Unxt


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    qp: tuple[gt.BBtQ, gt.BBtP],
    t0: gt.RealQScalar,
    t1: gt.RealQScalar,
    /,
    args: Any = (),
    saveat: Any = default_saveat,
    **solver_kw: Any,
) -> diffrax.Solution:
    """Solve for position tuple, start, end time.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = (u.Quantity([8, 0, 0], "kpc"),
    ...       u.Quantity([0, 220, 0], "km/s"))

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t0, t1)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    """
    # Units
    units = field.units
    time = units["time"]

    # Initial conditions
    y0 = (qp[0].ustrip(units["length"]), qp[1].ustrip(units["speed"]))
    y0 = tuple(jnp.broadcast_arrays(*y0))

    # Solve the differential equation
    solver_kw.setdefault("dt0", None)
    soln = self.diffeqsolver(
        field.terms,
        t0=t0.ustrip(time),
        t1=t1.ustrip(time),
        y0=y0,
        args=args,
        saveat=parse_saveat(units, saveat, dense=solver_kw.pop("dense", None)),
        **solver_kw,
    )

    # TODO: if t1=True should this remove the scalar time dimension?

    return soln  # noqa: RET504


# --------------------------------
# Coordinax


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    q3p3: tuple[cx.vecs.AbstractPos3D, cx.vecs.AbstractVel3D],
    t0: gt.RealQScalar,
    t1: gt.RealQScalar,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> diffrax.Solution:
    """Solve for position vector tuple, start, end time.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = (cx.CartesianPos3D.from_([8, 0, 0], "kpc"),
    ...       cx.CartesianVel3D.from_([0, 220, 0], "km/s"))

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t0, t1)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    """
    y0 = (convert(q3p3[0], FastQ), convert(q3p3[1], FastQ))
    # Redispatch on y0
    return self.solve(field, y0, t0, t1, args=args, **solver_kw)


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    q4p3: tuple[cx.vecs.FourVector, cx.vecs.AbstractVel3D],
    t1: gt.RealQScalar,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> diffrax.Solution:
    """Solve for 4-vector position tuple, end time.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = (cx.vecs.FourVector.from_([0, 8, 0, 0], "kpc"),
    ...       cx.CartesianVel3D.from_([0, 220, 0], "km/s"))

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    """
    q4, p = q4p3
    y0 = (convert(q4.q, FastQ), convert(p, FastQ))
    # Redispatch on y0
    return self.solve(field, y0, q4.t, t1, args=args, **solver_kw)


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    space: cx.Space,
    t1: gt.RealQScalar,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> diffrax.Solution:
    """Solve for Space[4vec, 3vel], end time.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = cx.Space(length=cx.vecs.FourVector.from_([0, 8, 0, 0], "kpc"),
    ...               speed=cx.CartesianVel3D.from_([0, 220, 0], "km/s"))

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    """
    q4, p = space["length"], space["speed"]
    q4 = eqx.error_if(
        q4,
        not isinstance(q4, cx.vecs.FourVector),
        "space['length']must be a 4-vector if `t0` is not given.",
    )
    # Redispatch on y0
    return self.solve(field, (q4.q, p), q4.t, t1, args=args, **solver_kw)


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    space: cx.Space,
    t0: gt.RealQScalar,
    t1: gt.RealQScalar,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> diffrax.Solution:
    """Solve for Space[3vec, 3vel], start, end time.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = cx.Space(length=cx.CartesianPos3D.from_([8, 0, 0], "kpc"),
    ...               speed=cx.CartesianVel3D.from_([0, 220, 0], "km/s"))

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t0, t1)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    """
    # Redispatch on y0
    y0 = (space["length"], space["speed"])
    return self.solve(field, y0, t0, t1, args=args, **solver_kw)


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0: cx.frames.AbstractCoordinate,
    t0: gt.RealQScalar,
    t1: gt.RealQScalar,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> diffrax.Solution:
    """Solve for `coordinax.frames.AbstractCoordinate`, start, end time.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = cx.Coordinate(
    ...     {"length": cx.CartesianPos3D.from_([8, 0, 0], "kpc"),
    ...      "speed": cx.CartesianVel3D.from_([0, 220, 0], "km/s")},
    ...     gc.frames.SimulationFrame()
    ... )

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t0, t1)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    """
    # Redispatch on y0
    return self.solve(field, w0.data, t0, t1, args=args, **solver_kw)


# --------------------------------
# PSPs


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0: gc.AbstractPhaseSpacePosition,  # TODO: handle frames
    t1: gt.RealQScalar,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> diffrax.Solution:
    """Solve for PSP with time, end time.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([8, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 220, 0], "km/s"),
    ...                            t=u.Quantity(0, "Gyr"))

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    """
    # Check that the initial conditions are valid.
    w0 = eqx.error_if(
        w0,
        w0.t is None,
        "If `t0` is not specified, `w0.t` must supply it.",
    )

    w0 = eqx.error_if(  # TODO: remove when frames are handled
        w0,
        not isinstance(w0.frame, gc.frames.SimulationFrame),
        "Only SimulationFrame is currently supported.",
    )

    # Redispatch on y0
    y0 = w0._qp(units=field.units)  # noqa: SLF001
    return self.solve(field, y0, w0.t, t1, args=args, **solver_kw)


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0: gc.AbstractPhaseSpacePosition,  # TODO: handle frames
    t0: gt.RealQScalar,
    t1: gt.RealQScalar,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> diffrax.Solution:
    """Solve for PSP without time, start, end time.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([8, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 220, 0], "km/s"))

    Vector field.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Solve EoM.

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t0, t1)
    >>> soln
    Solution(
      t0=f64[], t1=f64[], ts=f64[1],
      ys=(f64[1,3], f64[1,3]),
      interpolation=None,
      stats={ ... },
      result=EnumerationItem( ... ),
      ...
    )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    """
    # Check that the initial conditions are valid.
    w0 = eqx.error_if(
        w0,
        w0.t is not None,
        "If `t0` is specified, `w0.t` must be `None`.",
    )

    w0 = eqx.error_if(  # TODO: remove when frames are handled
        w0,
        not isinstance(w0.frame, gc.frames.SimulationFrame),
        "Only SimulationFrame is currently supported.",
    )

    # Redispatch on y0
    y0 = w0._qp(units=field.units)  # noqa: SLF001
    return self.solve(field, y0, t0, t1, args=args, **solver_kw)
