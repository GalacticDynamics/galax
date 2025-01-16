"""Core integrator module."""

__all__ = ["Integrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, Literal, TypeAlias, TypeVar, final

import diffrax as dfx
import equinox as eqx
from jaxtyping import ArrayLike, Shaped
from plum import dispatch

import quaxed.numpy as jnp
from unxt.quantity import AbstractQuantity, UncheckedQuantity as FastQ
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.dynamics._src.custom_types as gdt
import galax.typing as gt
from .interp import Interpolant
from galax.dynamics._src.diffeq import DiffEqSolver
from galax.dynamics._src.solve.dynamics import DynamicsSolver
from galax.dynamics.fields import AbstractDynamicsField

R = TypeVar("R")
Interp = TypeVar("Interp")
Time: TypeAlias = gt.TimeSz0 | gt.RealSz0Like
Times: TypeAlias = gt.QuSzTime | gt.SzTime


save_t1_only = dfx.SaveAt(t1=True)
default_solver = dfx.Dopri8(scan_kind="bounded")
default_stepsize_controller = dfx.PIDController(rtol=1e-7, atol=1e-7)


def converter_dynamicsolver(obj: Any, /) -> DynamicsSolver:
    """Convert to a `DynamicsSolver`.

    Examples
    --------
    >>> import diffrax as dfx
    >>> from galax.dynamics.integrate import DynamicsSolver, DiffEqSolver

    >>> diffeqsolve = DynamicsSolver(DiffEqSolver(dfx.Dopri5()))
    >>> converter_dynamicsolver(diffeqsolve)
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None)
      )
    )

    >>> diffeqsolve = DiffEqSolver(dfx.Dopri5())
    >>> converter_dynamicsolver(diffeqsolve)
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None)
      )
    )

    >>> converter_dynamicsolver(dfx.Dopri5())
    DynamicsSolver(
      diffeqsolver=DiffEqSolver(
        solver=Dopri5(scan_kind=None),
        stepsize_controller=ConstantStepSize(),
        adjoint=RecursiveCheckpointAdjoint(checkpoints=None)
      )
    )

    >>> try: converter_dynamicsolver(object())
    ... except TypeError as e: print(e)
    cannot convert <object object at ...> to a `DynamicsSolver`.

    """
    if isinstance(obj, DynamicsSolver):
        out = obj
    elif isinstance(obj, DiffEqSolver | dfx.AbstractSolver):
        out = DynamicsSolver(obj)
    else:
        msg = f"cannot convert {obj} to a `DynamicsSolver`."  # type: ignore[unreachable]
        raise TypeError(msg)
    return out


@final
class Integrator(eqx.Module, strict=True):  # type: ignore[call-arg,misc]
    """Integrator using :func:`diffrax.diffeqsolve`.

    This integrator uses the :func:`diffrax.diffeqsolve` function to integrate
    the equations of motion. :func:`diffrax.diffeqsolve` supports a wide range
    of solvers and options. See the documentation of :func:`diffrax.diffeqsolve`
    for more information.

    Parameters
    ----------
    solver : diffrax.AbstractSolver, optional
        The solver to use. Default is
        :class:`diffrax.Dopri8`(``scan_kind="bounded"``).
    stepsize_controller : diffrax.AbstractStepSizeController, optional
        The stepsize controller to use. Default is a PID controller with
        relative and absolute tolerances of 1e-7.
    diffeq_kw : Mapping[str, Any], optional
        Keyword arguments to pass to :func:`diffrax.diffeqsolve`. Default is
        ``{"max_steps": None, "event": None}``. The ``"max_steps"`` key is
        removed if ``dense=True`` in the :meth`Integrator.__call__` method.

    Examples
    --------
    First some imports:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    Then we define initial conditions:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s"))

    (Note that the ``t`` attribute is not used.)

    Now we can integrate the phase-space position for 1 Gyr, getting the final
    position.  The integrator accepts any function for the equations of motion.
    Here we will reproduce what happens with orbit integrations.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(field, w0, t0, t1)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [ 6.247 -5.121  0.   ]>,
        p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
            [0.359 0.033 0.   ]>,
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame())
    >>> w.shape
    ()

    Instead of just returning the final position, we can get the state of the
    system at any times ``saveat``:

    >>> ts = u.Quantity(jnp.linspace(0, 1, 10), "Gyr")  # 10 steps
    >>> ws = integrator(field, w0, t0, t1, saveat=ts)
    >>> ws
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )
    >>> ws.shape
    (10,)

    In all these examples the integrator was used to integrate a single
    position. The integrator can also be used to integrate a batch of initial
    conditions at once, returning a batch of final conditions (or a batch of
    conditions at the requested times ``saveat``):

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([[10, 0, 0], [11, 0, 0]], "kpc"),
    ...                            p=u.Quantity([[0, 200, 0], [0, 210, 0]], "km/s"))
    >>> ws = integrator(field, w0, t0, t1)
    >>> ws.shape
    (2,)

    A cool feature of the integrator is that it can return an interpolated
    solution.

    >>> w = integrator(field, w0, t0, t1, saveat=ts, dense=True)
    >>> type(w)
    <class 'galax.coordinates...InterpolatedPhaseSpacePosition'>

    The interpolated solution can be evaluated at any time in the domain to get
    the phase-space position at that time:

    >>> t = u.Quantity(jnp.e, "Gyr")
    >>> w(t)
    PhaseSpacePosition(
      q=CartesianPos3D(
        x=Quantity[...](value=...f64[2], unit=Unit("kpc")),
        ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(2.71828183, dtype=float64, ...), unit='Gyr'),
      frame=SimulationFrame()
    )

    The interpolant is vectorized:

    >>> t = u.Quantity(jnp.linspace(0, 1, 100), "Gyr")
    >>> w(t)
    PhaseSpacePosition(
      q=CartesianPos3D(
        x=Quantity[...](value=f64[2,100], unit=Unit("kpc")),
        ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(..., dtype=float64), unit='Gyr'),
      frame=SimulationFrame()
    )

    And it works on batches:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([[10, 0, 0], [11, 0, 0]], "kpc"),
    ...                            p=u.Quantity([[0, 200, 0], [0, 210, 0]], "km/s"))
    >>> ws = integrator(field, w0, t0, t1, dense=True)
    >>> ws.shape
    (2,)
    >>> w(t)
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(..., dtype=float64), unit='Gyr'),
        frame=SimulationFrame()
    )
    """

    dynamics_solver: DynamicsSolver = eqx.field(
        default=DynamicsSolver(
            DiffEqSolver(
                solver=dfx.Dopri8(scan_kind="bounded"),
                stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
            )
        ),
        converter=converter_dynamicsolver,
    )
    _: KW_ONLY
    diffeq_kw: Mapping[str, Any] = eqx.field(
        default=(("max_steps", 100_000),),
        static=True,
        converter=ImmutableMap,
    )

    # =====================================================
    # Call

    @partial(eqx.filter_jit)
    def _call_(
        self: "Integrator",
        field: AbstractDynamicsField,
        q0: gdt.BtQ,
        p0: gdt.BtP,
        t0: gt.TimeSz0,
        t1: gt.TimeSz0,
        /,
        *,
        saveat: gt.QuSzTime | None = None,  # not jitted here
        dense: Literal[False, True] = False,
        max_steps: int | None | Literal[True] = True,
    ) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
        """Run the integrator.

        This handles the shape cases that `diffrax.diffeqsolve` can handle
        without application of `jax.vmap` or `jax.numpy.vectorize`.

        I/O shapes:

        - q0=(3,), p0=(3,), t0=(), t1=(), saveat=() -> ()
        - q0=(3,), p0=(3,), t0=(), t1=(), saveat=(T,) -> (T,)
        - q0=(*B,3), p0=(*B,3), t0=(), t1=(), saveat=() -> (*B,)
        - q0=(*B,3), p0=(*B,3), t0=(), t1=(), saveat=(T) -> (*B,T)

        Parameters
        ----------
        field : `galax.dynamics.fields.AbstractDynamicsField`
            The field to integrate. Excluded from JIT.
        q0, p0 : Quantity[number, (*batch, 3), 'position' | 'speed']
            Initial conditions. Can have any (or no) batch dimensions. Included
            in JIT.
        t0, t1 : Quantity[number, (), 'time']
            Initial and final times. Included in JIT.

        saveat : Quantity[float, (T,), 'time'] | None, optional
            Times to return the computation.  If `None`, the computation is
            returned only at the final time. Excluded from JIT.
        dense : bool, optional
            Whether to return an interpolated solution. Excluded from JIT.

        """
        # Ensure `dense=True` won't raise an error.
        diffeq_kw = dict(self.diffeq_kw)
        if max_steps is not True:
            diffeq_kw["max_steps"] = max_steps

        # Perform the integration
        save_at = save_t1_only if saveat is None else saveat
        soln: dfx.Solution = self.dynamics_solver.solve(
            field, (q0, p0), t0, t1, saveat=save_at, dense=dense, **diffeq_kw
        )

        # Return
        out_kw = {
            "frame": gc.frames.SimulationFrame(),  # TODO: determine the frame
            "units": field.units,
        }
        if dense:
            out_cls = gc.InterpolatedPhaseSpacePosition
            out_kw["interpolant"] = Interpolant(soln.interpolation, units=field.units)
        else:
            out_cls = gc.PhaseSpacePosition

        return out_cls.from_(soln, **out_kw, unbatch_time=saveat is None)

    # -----------------------------------------------------
    # Call method

    @dispatch.abstract
    def __call__(
        self, field: AbstractDynamicsField, /, y0: Any, t0: Any, t1: Any, **kwargs: Any
    ) -> Any:
        """Integrate the equations of motion.

        Broadly, the integrator takes the field ``F`` and integrates the initial
        conditions ``y0`` from time ``t0`` to time ``t1``.

        This is the abstract method for the integrator. Actual methods are
        registered for dispatching below.

        """


# -------------------------------------------
# Scalar call


@Integrator.__call__.dispatch(precedence=2)
@eqx.filter_jit
def call(
    self: Integrator,
    field: AbstractDynamicsField,
    qp0: gdt.BtQP | gdt.BtQParr | gt.BtSz6,
    t0: Time,
    t1: Time,
    /,
    *,
    saveat: Times | None = None,
    **kwargs: Any,
) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
    """Run the integrator.

    This is the base dispatch for the integrator and handles the shape cases
    that `diffrax.diffeqsolve` can handle without application of `jax.vmap`
    or `jax.numpy.vectorize`.

    I/O shapes:

    - y0=((3,),(3,)), t0=(), t1=(), saveat=() -> ()
    - y0=((3,),(3,)), t0=(), t1=(), saveat=(T,) -> (T,)
    - y0=((*batch,3),(*batch,3)), t0=(), t1=(), saveat=() -> (*batch,)
    - y0=((*batch,3),(*batch,3)), t0=(), t1=(), saveat=(T) -> (*batch,T)

    - y0=(6,), t0=(), t1=(), saveat=() -> ()
    - y0=(6,), t0=(), t1=(), saveat=(T,) -> (T,)
    - y0=(*batch,6), t0=(), t1=(), saveat=() -> (*batch,)
    - y0=(*batch,6), t0=(), t1=(), saveat=(T) -> (*batch,T)

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s")
    ...                            ).w(units="galactic")
    >>> w0.shape
    (6,)

    (Note that the ``t`` attribute is not used.)

    Now we can integrate the phase-space position for 1 Gyr, getting the
    final position.  The integrator accepts any function for the equations
    of motion.  Here we will reproduce what happens with orbit integrations.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(field, w0, t0, t1)
    >>> w
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )
    >>> w.shape
    ()

    We can also request the orbit at specific times:

    >>> ts = u.Quantity(jnp.linspace(0, 1, 10), "Myr")  # 10 steps
    >>> ws = integrator(field, w0, t0, t1, saveat=ts)
    >>> ws
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )
    >>> ws.shape
    (10,)

    If the integration fails, the `max_steps` parameter can be changed:

    >>> w2 = integrator(field, w0, t0, t1, max_steps=10_000)
    >>> w2 == w
    Array(True, dtype=bool)

    """
    y0 = qp0 if isinstance(qp0, tuple) else (qp0[..., 0:3], qp0[..., 3:6])
    units = field.units
    return self._call_(
        field,
        FastQ.from_(y0[0], units["length"]),
        FastQ.from_(y0[1], units["speed"]),
        FastQ.from_(t0, units["time"]),
        FastQ.from_(t1, units["time"]),
        saveat=FastQ.from_(saveat, units["time"]) if saveat is not None else None,
        **kwargs,
    )


# -------------------------------------------
# Kwarg options


@Integrator.__call__.dispatch_multi(
    (Integrator, AbstractDynamicsField),  # (F,)
    (Integrator, AbstractDynamicsField, Any),  # (F, y0)
    (Integrator, AbstractDynamicsField, Any, Any),  # (F, y0, t0)
)
def call(
    self: Integrator, field: AbstractDynamicsField, *args: Any, **kwargs: Any
) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
    """Support keyword arguments by re-dispatching.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s"))

    (Note that the ``t`` attribute is not used.)

    Now we can integrate the phase-space position for 1 Gyr, getting the
    final position.  The integrator accepts any function for the equations
    of motion.  Here we will reproduce what happens with orbit integrations.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")

    Different kwargs:

    >>> w = integrator(field, w0, t0, t1=t1)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [ 6.247 -5.121  0.   ]>,
        p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
            [0.359 0.033 0.   ]>,
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    >>> w = integrator(field, w0, t0=t0, t1=t1)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [ 6.247 -5.121  0.   ]>,
        p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
            [0.359 0.033 0.   ]>,
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    >>> w = integrator(field, y0=w0, t0=t0, t1=t1)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [ 6.247 -5.121  0.   ]>,
        p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
            [0.359 0.033 0.   ]>,
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    """
    # y0: Any, t0: Any, t1: Any
    match args:
        case (y0, t0):
            t1 = kwargs.pop("t1")
        case (y0,):
            t0 = kwargs.pop("t0")
            t1 = kwargs.pop("t1")
        case ():
            y0 = kwargs.pop("y0")
            t0 = kwargs.pop("t0")
            t1 = kwargs.pop("t1")
        case _:  # pragma: no cover
            match = f"Invalid number of arguments: {args}"
            raise TypeError(match)

    return self(field, y0, t0, t1, **kwargs)


# -------------------------------------------
# Vectorized call


@Integrator.__call__.dispatch(precedence=1)
@eqx.filter_jit
def call(
    self: Integrator,
    field: AbstractDynamicsField,
    y0: gdt.BBtQP | gdt.BBtQParr | gt.BBtSz6,
    t0: Shaped[AbstractQuantity, "*#batch"] | Shaped[ArrayLike, "*#batch"] | Time,
    t1: Shaped[AbstractQuantity, "*#batch"] | Shaped[ArrayLike, "*#batch"] | Time,
    /,
    *,
    saveat: Times | None = None,
    **kwargs: Any,
) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
    """Run the integrator, vectorizing in the initial/final times.

    I/O shapes:

    - y0=((*#B,3),(*#B,3)), t0=(*#B,), t1=(), saveat=() -> (*B,)
    - y0=((*#B,3),(*#B,3)), t0=(), t1=(*#B,), saveat=() -> (*B,)
    - y0=((*#B,3),(*#B,3)), t0=(*#B), t1=(*#B,), saveat=() -> (*B,)
    - y0=((*#B,3),(*#B,3)), t0=(*#B,), t1=(), saveat=(T,) -> (*B,T)
    - y0=((*#B,3),(*#B,3)), t0=(), t1=(*#B,), saveat=(T,) -> (*B,T)
    - y0=((*#B,3),(*#B,3)), t0=(*#B), t1=(*#B,), saveat=(T,) -> (*B,T)

    - y0=(*#batch,6), t0=(*#batch,), t1=(), saveat=() -> (*batch,)
    - y0=(*#batch,6), t0=(), t1=(*#batch,), saveat=() -> (*batch,)
    - y0=(*#batch,6), t0=(*#batch), t1=(*#batch,), saveat=() -> (*batch,)
    - y0=(*#batch,6), t0=(*#batch,), t1=(), saveat=(T,) -> (*batch,T)
    - y0=(*#batch,6), t0=(), t1=(*#batch,), saveat=(T,) -> (*batch,T)
    - y0=(*#batch,6), t0=(*#batch), t1=(*#batch,), saveat=(T,) -> (*batch,T)

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    The integrator can be used to integrate a batch of initial conditions at
    once, returning a batch of final conditions (or a batch of conditions at
    the requested times):

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([[10, 0, 0], [11, 0, 0]], "kpc"),
    ...                            p=u.Quantity([[0, 200, 0], [0, 210, 0]], "km/s"))

    Now we can integrate the phase-space position for 1 Gyr, getting the
    final position.  The integrator accepts any function for the equations
    of motion.  Here we will reproduce what happens with orbit integrations.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> integrator = gd.integrate.Integrator()
    >>> ws = integrator(field, w0, t0, t1)
    >>> ws.shape
    (2,)

    """
    y0_ = y0 if isinstance(y0, tuple) else (y0[..., 0:3], y0[..., 3:6])

    # Vectorize the call
    # This depends on the shape of saveat
    units = field.units
    saveat = None if saveat is None else FastQ.from_(saveat, units["time"])
    vec_call = jnp.vectorize(
        lambda *args: self._call_(*args, saveat=saveat, **kwargs),
        signature="(3),(3),(),()->" + ("()" if saveat is None else "(T)"),
        excluded=(0,),
    )

    return vec_call(
        field,
        FastQ.from_(y0_[0], units["length"]),
        FastQ.from_(y0_[1], units["speed"]),
        FastQ.from_(t0, units["time"]),
        FastQ.from_(t1, units["time"]),
    )


# -------------------------------------------
# w0 is a PhaseSpacePosition


@Integrator.__call__.dispatch
def call(
    self: Integrator,
    field: AbstractDynamicsField,
    w0: gc.AbstractOnePhaseSpacePosition,
    t0: Any,
    t1: Any,
    /,
    **kwargs: Any,
) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
    """Run the integrator.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions and a potential:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s"))

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    We can integrate the phase-space position:

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(field, w0, t0, t1)
    >>> w
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )

    """
    return self(field, w0._qp(units=field.units), t0, t1, **kwargs)  # noqa: SLF001


@Integrator.__call__.dispatch
def call(
    self: Integrator,
    field: AbstractDynamicsField,
    w0: gc.AbstractCompositePhaseSpacePosition,
    t0: Any,
    t1: Any,
    /,
    **kwargs: Any,
) -> gc.CompositePhaseSpacePosition:
    """Run the integrator on a composite phase-space position.

    Examples
    --------
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions and a potential:

    >>> w01 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                             p=u.Quantity([0, 200, 0], "km/s"))
    >>> w02 = gc.PhaseSpacePosition(q=u.Quantity([0, 10, 0], "kpc"),
    ...                             p=u.Quantity([-200, 0, 0], "km/s"))
    >>> w0 = gc.CompositePhaseSpacePosition(w01=w01, w02=w02)

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    We can integrate the composite phase-space position:

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(field, w0, t0, t1)
    >>> print(w)
    CompositePhaseSpacePosition(
        w01=PhaseSpacePosition(
            q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
                [ 6.247 -5.121  0.   ]>,
            p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
                [0.359 0.033 0.   ]>,
            t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
            frame=SimulationFrame()),
        w02=PhaseSpacePosition(
            q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
                [5.121 6.247 0.   ]>,
            p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
                [-0.033  0.359  0.   ]>,
            t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
            frame=SimulationFrame()))

    """
    # TODO: Interpolated form
    return gc.CompositePhaseSpacePosition(
        **{k: self(field, psp0, t0, t1, **kwargs) for k, psp0 in w0.items()}
    )
