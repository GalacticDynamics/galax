"""Core integrator module."""

__all__ = ["Integrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, Literal, TypeAlias, TypeVar, cast, final

import diffrax
import equinox as eqx
from jaxtyping import ArrayLike, Shaped
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AbstractQuantity, UncheckedQuantity as FastQ
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.typing as gt
from .interp import Interpolant
from .utils import converter_dynamicsolver
from galax.dynamics._src.solve.diffeq import DiffEqSolver
from galax.dynamics._src.solve.dynamics import DynamicsSolver
from galax.dynamics.fields import AbstractDynamicsField

R = TypeVar("R")
Interp = TypeVar("Interp")
Time: TypeAlias = gt.TimeScalar | gt.RealScalarLike
Times: TypeAlias = gt.QVecTime | gt.VecTime


default_solver = diffrax.Dopri8(scan_kind="bounded")
default_stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-7)


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
        removed if ``interpolated=True`` in the :meth`Integrator.__call__`
        method.

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
    >>> w = integrator(field, w0, t0, t1, units=galactic)
    >>> w
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )
    >>> w.shape
    ()

    Instead of just returning the final position, we can get the state of the
    system at any times ``saveat``:

    >>> ts = u.Quantity(jnp.linspace(0, 1, 10), "Gyr")  # 10 steps
    >>> ws = integrator(field, w0, t0, t1,
    ...                 saveat=ts, units=galactic)
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
    >>> ws = integrator(field, w0, t0, t1, units=galactic)
    >>> ws.shape
    (2,)

    A cool feature of the integrator is that it can return an interpolated
    solution.

    >>> w = integrator(field, w0, t0, t1, saveat=ts, units=galactic,
    ...                interpolated=True)
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
    >>> ws = integrator(field, w0, t0, t1, units=galactic,
    ...                 interpolated=True)
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
                solver=diffrax.Dopri8(scan_kind="bounded"),
                stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-7),
            )
        ),
        converter=converter_dynamicsolver,
    )
    _: KW_ONLY
    diffeq_kw: Mapping[str, Any] = eqx.field(
        default=(("max_steps", None),),
        static=True,
        converter=ImmutableMap,
    )

    # =====================================================
    # Call

    # @partial(jax.jit, static_argnums=(0, 1), static_argnames=("units", "interpolated"))  # noqa: E501
    @partial(eqx.filter_jit)
    def _call_(
        self: "Integrator",
        field: AbstractDynamicsField,
        q0: gt.BtQ,
        p0: gt.BtP,
        t0: gt.TimeScalar,
        t1: gt.TimeScalar,
        /,
        *,
        saveat: gt.QVecTime | None = None,  # not jitted here
        units: u.AbstractUnitSystem,
        interpolated: Literal[False, True] = False,
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
        units : `unxt.AbstractUnitSystem`
            The unit system to use. Excluded from JIT.
        interpolated : bool, optional
            Whether to return an interpolated solution. Excluded from JIT.

        """
        # ---------------------------------------
        # Parse inputs

        # Either save at `saveat` or at the final time.
        time = units["time"]
        only_final = saveat is None or len(saveat) <= 1
        save_at = diffrax.SaveAt(
            t1=only_final,
            ts=None if only_final else cast(AbstractQuantity, saveat).ustrip(time),
            dense=interpolated,
        )

        diffeq_kw = dict(self.diffeq_kw)
        if interpolated and diffeq_kw.get("max_steps") is None:
            diffeq_kw.pop("max_steps")

        # ---------------------------------------
        # Perform the integration

        soln: diffrax.Solution = self.dynamics_solver.solve(
            field, (q0, p0), t0, t1, saveat=save_at, **diffeq_kw
        )

        # Reshape (T, *batch) to (*batch, T)
        solt = soln.ts  # soln.ts is already in the correct shape
        solq = jnp.moveaxis(soln.ys[0], 0, -2)
        solp = jnp.moveaxis(soln.ys[1], 0, -2)

        # Parse the solution, (unbatching time when saveat is None)
        if saveat is None:
            solt = solt[..., -1]
            solq = solq[..., -1, :]
            solp = solp[..., -1, :]

        # ---------------------------------------
        # Return

        # TODO: determine the frame
        out_kw = {"frame": gc.frames.SimulationFrame()}

        if interpolated:
            out_cls = gc.InterpolatedPhaseSpacePosition
            out_kw["interpolant"] = Interpolant(soln.interpolation, units=units)
        else:
            out_cls = gc.PhaseSpacePosition

        return out_cls(  # shape = (*batch, T)
            t=FastQ(solt, time),
            q=FastQ(solq, units["length"]),
            p=FastQ(solp, units["speed"]),
            **out_kw,
        )

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
@eqx.filter_jit  # @partial(jax.jit, static_argnums=(0, 1), static_argnames=("units", "interpolated"))  # noqa: E501
def call(
    self: Integrator,
    field: AbstractDynamicsField,
    qp0: gt.BtQP | gt.BtQParr,
    t0: Time,
    t1: Time,
    /,
    *,
    units: u.AbstractUnitSystem,
    saveat: Times | None = None,
    interpolated: bool = False,
) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
    """Run the integrator.

    This is the base dispatch for the integrator and handles the shape cases
    that `diffrax.diffeqsolve` can handle without application of `jax.vmap`
    or `jax.numpy.vectorize`.

    I/O shapes:

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
    >>> w = integrator(field, w0, t0, t1, units=galactic)
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
    >>> ws = integrator(field, w0, t0, t1,
    ...                 saveat=ts, units=galactic)
    >>> ws
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )
    >>> ws.shape
    (10,)

    """
    return self._call_(
        field,
        FastQ.from_(qp0[0], units["length"]),
        FastQ.from_(qp0[1], units["speed"]),
        FastQ.from_(t0, units["time"]),
        FastQ.from_(t1, units["time"]),
        saveat=FastQ.from_(saveat, units["time"]) if saveat is not None else None,
        units=units,
        interpolated=interpolated,
    )


@Integrator.__call__.dispatch(precedence=2)
@eqx.filter_jit  # @partial(jax.jit, static_argnums=(0, 1), static_argnames=("units", "interpolated"))  # noqa: E501
def call(
    self: Integrator,
    field: AbstractDynamicsField,
    y0: gt.BtVec6,
    t0: Time,
    t1: Time,
    /,
    *,
    units: u.AbstractUnitSystem,
    saveat: Times | None = None,
    interpolated: bool = False,
) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
    """Run the integrator.

    This is the base dispatch for the integrator and handles the shape cases
    that `diffrax.diffeqsolve` can handle without application of `jax.vmap`
    or `jax.numpy.vectorize`.

    I/O shapes:

    - y0=(6,), t0=(), t1=(), saveat=() -> ()
    - y0=(6,), t0=(), t1=(), saveat=(T,) -> (T,)
    - y0=(*batch,6), t0=(), t1=(), saveat=() -> (*batch,)
    - y0=(*batch,6), t0=(), t1=(), saveat=(T) -> (*batch,T)

    """
    return self(  # redispatch to the q,p form
        field,
        (y0[..., 0:3], y0[..., 3:6]),
        t0,
        t1,
        saveat=saveat,
        units=units,
        interpolated=interpolated,
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

    >>> w = integrator(field, w0, t0, t1=t1, units=galactic)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [ 6.247 -5.121  0.   ]>,
        p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
            [0.359 0.033 0.   ]>,
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    >>> w = integrator(field, w0, t0=t0, t1=t1, units=galactic)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [ 6.247 -5.121  0.   ]>,
        p=<CartesianVel3D (d_x[kpc / Myr], d_y[kpc / Myr], d_z[kpc / Myr])
            [0.359 0.033 0.   ]>,
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    >>> w = integrator(field, y0=w0, t0=t0, t1=t1, units=galactic)
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
    y0: gt.BBtQP | gt.BBtQParr,
    t0: Shaped[AbstractQuantity, "*#batch"] | Shaped[ArrayLike, "*#batch"] | Time,
    t1: Shaped[AbstractQuantity, "*#batch"] | Shaped[ArrayLike, "*#batch"] | Time,
    /,
    *,
    units: u.AbstractUnitSystem,
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

    """
    # Vectorize the call
    # This depends on the shape of saveat
    saveat = None if saveat is None else FastQ.from_(saveat, units["time"])
    vec_call = jnp.vectorize(
        lambda *args: self._call_(*args, units=units, saveat=saveat, **kwargs),
        signature="(3),(3),(),()->" + ("()" if saveat is None else "(T)"),
        excluded=(0,),
    )

    return vec_call(
        field,
        FastQ.from_(y0[0], units["length"]),
        FastQ.from_(y0[1], units["speed"]),
        FastQ.from_(t0, units["time"]),
        FastQ.from_(t1, units["time"]),
    )


@Integrator.__call__.dispatch(precedence=1)
@eqx.filter_jit
def call(
    self: Integrator,
    field: AbstractDynamicsField,
    y0: gt.BBtVec6,
    t0: Shaped[AbstractQuantity, "*#batch"] | Shaped[ArrayLike, "*#batch"] | Time,
    t1: Shaped[AbstractQuantity, "*#batch"] | Shaped[ArrayLike, "*#batch"] | Time,
    /,
    *,
    units: u.AbstractUnitSystem,
    saveat: Times | None = None,
    **kwargs: Any,
) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
    """Run the integrator, vectorizing in the initial/final times.

    I/O shapes:

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

    >>> integrator = gd.integrate.Integrator()
    >>> ws = integrator(field, w0, t0, t1, units=galactic)
    >>> ws.shape
    (2,)

    """
    return self(  # redispatch to the q,p form
        field,
        (y0[..., 0:3], y0[..., 3:6]),
        t0,
        t1,
        units=units,
        saveat=saveat,
        **kwargs,
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
    *,
    units: u.AbstractUnitSystem,
    saveat: Times | None = None,
    interpolated: Literal[False, True] = False,
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
    >>> w = integrator(field, w0, t0, t1, units=galactic)
    >>> w
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )

    """
    return self(
        field,
        w0._qp(units=units),  # noqa: SLF001
        t0,
        t1,
        saveat=saveat,
        units=units,
        interpolated=interpolated,
    )


@Integrator.__call__.dispatch
def call(
    self: Integrator,
    field: AbstractDynamicsField,
    w0: gc.AbstractCompositePhaseSpacePosition,
    t0: Any,
    t1: Any,
    /,
    *,
    units: u.AbstractUnitSystem,
    saveat: Times | None = None,
    interpolated: Literal[False, True] = False,
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
    >>> w = integrator(field, w0, t0, t1, units=galactic)
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
        **{
            k: self(
                field,
                psp0,
                t0,
                t1,
                saveat=saveat,
                units=units,
                interpolated=interpolated,
            )
            for k, psp0 in w0.items()
        }
    )
