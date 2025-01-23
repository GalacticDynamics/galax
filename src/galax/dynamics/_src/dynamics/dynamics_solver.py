"""Dynamics Solvers.

This is private API.

"""

__all__ = ["DynamicsSolver"]


from typing import Any, final

import diffrax as dfx
import equinox as eqx
from jaxtyping import PyTree
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc
import galax.dynamics._src.custom_types as gdt
import galax.typing as gt
from .field_base import AbstractDynamicsField
from .utils import parse_saveat
from galax.dynamics._src.diffeq import DiffEqSolver
from galax.dynamics._src.solver import AbstractSolver


@final
class DynamicsSolver(AbstractSolver, strict=True):  # type: ignore[call-arg]
    """Dynamics solver.

    The most useful method is `.solve`, which handles initialization and
    stepping to the final solution. Manual solves can be done with `.init()` and
    repeat `.step()`.

    Examples
    --------
    The ``.solve()`` method uses multiple dispatch to handle many different
    problem setups. Check out the method's docstring for examples. Here we show
    a simple example.

    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()  # defaults to Dopri8

    Define the vector field. In this example it's to solve Hamilton's EoM in a
    gravitational potential.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...    r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Define the initial conditions, here a phase-space position

    >>> w0 = gc.PhaseSpacePosition(
    ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))

    Solve, stepping from `w0.t` to `t1`.

    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=(f64[1,2,3], f64[1,2,3]),
              ... )

    >>> w = gc.PhaseSpacePosition.from_(soln, units=pot.units, frame=w0.frame)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[-5.151 -6.454 -5.795]]
             [[ 4.277  4.633  1.426]]]>,
        p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
            [[[ 0.225 -0.068  0.253]]
             [[-0.439 -0.002 -0.146]]]>,
        t=Quantity['time'](Array([1000.], dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    The solver can be customized. Here are a few examples:

    1. From a `galax.dynamics.integrate.DiffEqSolver` instance. This allows for
       setting the `diffrax.AbstractSolver`,
       `diffrax.AbstractStepSizeController`, etc.

    >>> diffeqsolver = gd.integrate.DiffEqSolver(dfx.Dopri8(),
    ...     stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))
    >>> solver = gd.integrate.DynamicsSolver(diffeqsolver)
    >>> solver
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri8(scan_kind=None),
        stepsize_controller=PIDController( rtol=1e-05, atol=1e-05, ... ),
        ...
      )
    )

    2. A `dict` of keyword arguments that are passed to
       `galax.dynamics.integrate.DiffEqSolver`.

    >>> solver = gd.integrate.DynamicsSolver({
    ...     "solver": dfx.Dopri8(), "stepsize_controller": dfx.ConstantStepSize()})
    >>> solver
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri8(scan_kind=None), stepsize_controller=ConstantStepSize(),
        ...
      )
    )

    """

    #: Solver for the differential equation.
    diffeqsolver: DiffEqSolver = eqx.field(
        default=DiffEqSolver(
            solver=dfx.Dopri8(),
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8),
        ),
        converter=DiffEqSolver.from_,
    )

    # -------------------------------------------

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
    ) -> dfx.Solution:
        """Call `diffrax.diffeqsolve`."""
        raise NotImplementedError  # pragma: no cover


# ===============================================
# Solve Dispatches

default_saveat = dfx.SaveAt(t1=True)

# --------------------------------
# JAX & Unxt


@DynamicsSolver.solve.dispatch
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    qp: tuple[gdt.BBtQ, gdt.BBtP],
    t0: gt.RealQuSz0,
    t1: gt.RealQuSz0,
    /,
    args: Any = (),
    saveat: Any = default_saveat,
    **solver_kw: Any,
) -> dfx.Solution:
    """Solve for batch position tuple, scalar start, end time.

    In ``solver_kw``, the following keys are recognized:

    - All keys recognized by `diffrax.diffeqsolve`. In particular if "dt0" is
      not specified it is assumed to be `None`.
    - "dense" (bool): If `True`, `saveat` is modified to have ``dense=True``.

    The output shape aligns with `diffrax.diffeqsolve`: (*batch, [time],
    *shape), where [time] is >= 1.

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

    Solve EoM from `t0` to `t1`, returning the solution at `t1`.

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t0, t1)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=(f64[1,3], f64[1,3]), ... )
    >>> soln.ys
    (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
     Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

    This can be solved for a specific time, not just `t1`.

    >>> soln = solver.solve(field, w0, t0, t1, saveat=u.Quantity(0.5, "Gyr"))
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=(f64[1,3], f64[1,3]), ... )

    >>> soln = solver.solve(field, w0, t0, t1, saveat=u.Quantity([0.25, 0.5], "Gyr"))
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[2],
              ys=(f64[2,3], f64[2,3]), ... )

    """
    # Units
    units = field.units
    time = units["time"]

    # Initial conditions
    y0 = (qp[0].ustrip(units["length"]), qp[1].ustrip(units["speed"]))
    y0 = tuple(jnp.broadcast_arrays(*y0))

    # Solve the differential equation
    solver_kw.setdefault("dt0", None)
    saveat = parse_saveat(units, saveat, dense=solver_kw.pop("dense", None))
    soln = self.diffeqsolver(
        field.terms(self.diffeqsolver),
        t0=t0.ustrip(time),
        t1=t1.ustrip(time),
        y0=y0,
        args=args,
        saveat=saveat,
        **solver_kw,
    )

    return soln  # noqa: RET504


@DynamicsSolver.solve.dispatch(precedence=-1)
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    qp: tuple[gdt.BBtQ, gdt.BBtP],
    t0: gt.BBtRealQuSz0,
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    saveat: Any = default_saveat,
    **solver_kw: Any,
) -> dfx.Solution:
    """Solve for batch position tuple, batched start, end time.

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

    Solve EoM from `t0` to `t1`, returning the solution at `t1`.

    >>> t0 = u.Quantity(0, "Gyr")
    >>> t1 = u.Quantity([1, 1.1, 1.2], "Gyr")
    >>> soln = solver.solve(field, w0, t0, t1)
    >>> soln
    Solution( t0=f64[3], t1=f64[3], ts=f64[3,1],
              ys=(f64[3,1,3], f64[3,1,3]), ... )

    """

    def call(q: gdt.Q, p: gdt.P, t0: gt.RealQuSz0, t1: gt.RealQuSz0) -> dfx.Solution:
        return self.solve(field, (q, p), t0, t1, args=args, saveat=saveat, **solver_kw)

    vec_call = jnp.vectorize(call, signature="(3),(3),(),()->()")

    return vec_call(qp[0], qp[1], t0, t1)


# --------------------------------
# Coordinax


@DynamicsSolver.solve.dispatch
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    q3p3: tuple[cx.vecs.AbstractPos3D, cx.vecs.AbstractVel3D],
    t0: gt.BBtRealQuSz0,
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> dfx.Solution:
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
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    q4p3: tuple[cx.vecs.FourVector, cx.vecs.AbstractVel3D],
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> dfx.Solution:
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
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    space: cx.Space,
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> dfx.Solution:
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
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    space: cx.Space,
    t0: gt.BBtRealQuSz0,
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> dfx.Solution:
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
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0: cx.frames.AbstractCoordinate,
    t0: gt.BBtRealQuSz0,
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> dfx.Solution:
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
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0: gc.AbstractPhaseSpacePosition,  # TODO: handle frames
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> dfx.Solution:
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
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0: gc.AbstractPhaseSpacePosition,  # TODO: handle frames
    t0: gt.BBtRealQuSz0,
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> dfx.Solution:
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


@DynamicsSolver.solve.dispatch
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0s: gc.AbstractCompositePhaseSpacePosition,
    t0: gt.BBtRealQuSz0,
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    **solver_kw: Any,
) -> dict[str, dfx.Solution]:
    """Solve for CompositePhaseSpacePosition, start, end time.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver()

    Initial conditions.

    >>> w01 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                             p=u.Quantity([0, 200, 0], "km/s"))
    >>> w02 = gc.PhaseSpacePosition(q=u.Quantity([0, 10, 0], "kpc"),
    ...                             p=u.Quantity([-200, 0, 0], "km/s"))
    >>> w0s = gc.CompositePhaseSpacePosition(w01=w01, w02=w02)

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")

    >>> soln = solver.solve(field, w0s, t0, t1)
    >>> soln
    {'w01': Solution( t0=f64[], t1=f64[], ts=f64[1],
                      ys=(f64[1,3], f64[1,3]), ... ),
     'w02': Solution( t0=f64[], t1=f64[], ts=f64[1],
                      ys=(f64[1,3], f64[1,3]), ... )}

    """
    return {
        k: self.solve(field, w0, t0, t1, args=args, **solver_kw)
        for k, w0 in w0s.items()
    }


# ===================================================================


@AbstractDynamicsField.terms.dispatch  # type: ignore[misc]
def terms(
    self: AbstractDynamicsField, wrapper: DynamicsSolver, /
) -> PyTree[dfx.AbstractTerm]:
    """Return diffeq terms, redispatching to the solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DynamicsSolver(dfx.Dopri8())

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    """
    return self.terms(wrapper.diffeqsolver)


# ===============================================
# TODO: MOVE TO A BETTER PLACE


@gc.AbstractOnePhaseSpacePosition.from_.dispatch  # type: ignore[misc,attr-defined]
def from_(
    cls: type[gc.AbstractOnePhaseSpacePosition],
    soln: dfx.Solution,
    *,
    frame: cx.frames.AbstractReferenceFrame,  # not dispatched on, but required
    units: u.AbstractUnitSystem,  # not dispatched on, but required
    unbatch_time: bool = False,
) -> gc.AbstractOnePhaseSpacePosition:
    """Convert a solution to a phase-space position.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...    r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)
    >>> solver = gd.integrate.DynamicsSolver()  # defaults to Dopri8
    >>> w0 = gc.PhaseSpacePosition(
    ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))
    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)

    >>> w = gc.PhaseSpacePosition.from_(soln, units=pot.units, frame=w0.frame)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[-5.151 -6.454 -5.795]]
             [[ 4.277  4.633  1.426]]]>,
        p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
            [[[ 0.225 -0.068  0.253]]
             [[-0.439 -0.002 -0.146]]]>,
        t=Quantity['time'](Array([1000.], dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    """
    # Reshape (T, *batch) to (*batch, T)
    t = soln.ts  # already in the correct shape
    q = jnp.moveaxis(soln.ys[0], 0, -2)
    p = jnp.moveaxis(soln.ys[1], 0, -2)

    # Reshape (*batch,T=1,6) to (*batch,6) if t is a scalar
    if unbatch_time and t.shape[-1] == 1:
        t = t[..., -1]
        q = q[..., -1, :]
        p = p[..., -1, :]

    # Convert the solution to a phase-space position
    return cls(
        q=cx.CartesianPos3D.from_(q, units["length"]),
        p=cx.CartesianVel3D.from_(p, units["speed"]),
        t=FastQ(t, units["time"]),
        frame=frame,
    )
