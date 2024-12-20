__all__ = ["Integrator"]

from collections.abc import Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, Literal, ParamSpec, TypeAlias, TypeVar, final

import diffrax
import equinox as eqx
import jax
from jaxtyping import ArrayLike, Shaped
from plum import dispatch

import quaxed.numpy as xp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.typing as gt
from .type_hints import VectorField

P = ParamSpec("P")
R = TypeVar("R")
Interp = TypeVar("Interp")
Time: TypeAlias = gt.TimeScalar | gt.RealScalarLike
Times: TypeAlias = gt.QVecTime | gt.VecTime


# ============================================================================
# Integration


@final
class Integrator(eqx.Module, strict=True):  # type: ignore[call-arg,misc]
    """Integrator using :func:`diffrax.diffeqsolve`.

    This integrator uses the :func:`diffrax.diffeqsolve` function to integrate
    the equations of motion. :func:`diffrax.diffeqsolve` supports a wide range
    of solvers and options. See the documentation of :func:`diffrax.diffeqsolve`
    for more information.

    Parameters
    ----------
    Solver : type[diffrax.AbstractSolver], optional
        The solver to use. Default is :class:`diffrax.Dopri8`.
    stepsize_controller : diffrax.AbstractStepSizeController, optional
        The stepsize controller to use. Default is a PID controller with
        relative and absolute tolerances of 1e-7.
    diffeq_kw : Mapping[str, Any], optional
        Keyword arguments to pass to :func:`diffrax.diffeqsolve`. Default is
        ``{"max_steps": None, "event": None}``. The ``"max_steps"`` key is
        removed if ``interpolated=True`` in the :meth`Integrator.__call__`
        method.
    solver_kw : Mapping[str, Any], optional
        Keyword arguments to pass to the solver. Default is ``{"scan_kind":
        "bounded"}``.

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

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10., 0., 0.], "kpc"),
    ...                            p=u.Quantity([0., 200., 0.], "km/s"))

    (Note that the ``t`` attribute is not used.)

    Now we can integrate the phase-space position for 1 Gyr, getting the final
    position.  The integrator accepts any function for the equations of motion.
    Here we will reproduce what happens with orbit integrations.

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(pot._dynamics_deriv, w0, t0, t1, units=galactic)
    >>> w
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity[...](value=f64[], unit=Unit("Myr"))
    )
    >>> w.shape
    ()

    Instead of just returning the final position, we can get the state of the
    system at any times ``saveat``:

    >>> ts = u.Quantity(jnp.linspace(0, 1, 10), "Gyr")  # 10 steps
    >>> ws = integrator(pot._dynamics_deriv, w0, t0, t1,
    ...                 saveat=ts, units=galactic)
    >>> ws
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity[...](value=f64[10], unit=Unit("Myr"))
    )
    >>> ws.shape
    (10,)

    In all these examples the integrator was used to integrate a single
    position. The integrator can also be used to integrate a batch of initial
    conditions at once, returning a batch of final conditions (or a batch of
    conditions at the requested times ``saveat``):

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([[10., 0, 0], [11., 0, 0]], "kpc"),
    ...                            p=u.Quantity([[0, 200, 0], [0, 210, 0]], "km/s"))
    >>> ws = integrator(pot._dynamics_deriv, w0, t0, t1, units=galactic)
    >>> ws.shape
    (2,)

    A cool feature of the integrator is that it can return an interpolated
    solution.

    >>> w = integrator(pot._dynamics_deriv, w0, t0, t1, saveat=ts, units=galactic,
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
      t=Quantity[...](value=...f64[], unit=Unit("Gyr"))
    )

    The interpolant is vectorized:

    >>> t = u.Quantity(jnp.linspace(0, 1, 100), "Gyr")
    >>> w(t)
    PhaseSpacePosition(
      q=CartesianPos3D(
        x=Quantity[...](value=f64[2,100], unit=Unit("kpc")),
        ... ),
      p=CartesianVel3D( ... ),
      t=Quantity[...](value=f64[100], unit=Unit("Gyr"))
    )

    And it works on batches:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([[10., 0, 0], [11., 0, 0]], "kpc"),
    ...                            p=u.Quantity([[0, 200, 0], [0, 210, 0]], "km/s"))
    >>> ws = integrator(pot._dynamics_deriv, w0, t0, t1, units=galactic,
    ...                 interpolated=True)
    >>> ws.shape
    (2,)
    >>> w(t)
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity[PhysicalType('time')](value=f64[100], unit=Unit("Gyr"))
    )
    """

    _: KW_ONLY
    Solver: type[diffrax.AbstractSolver] = eqx.field(
        default=diffrax.Dopri8, static=True
    )
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        default=diffrax.PIDController(rtol=1e-7, atol=1e-7), static=True
    )
    diffeq_kw: Mapping[str, Any] = eqx.field(
        default=(("max_steps", None), ("event", None)),
        static=True,
        converter=ImmutableMap,
    )
    solver_kw: Mapping[str, Any] = eqx.field(
        default=(("scan_kind", "bounded"),), static=True, converter=ImmutableMap
    )

    # =====================================================
    # Call

    # @partial(jax.jit, static_argnums=(0, 1), static_argnames=("units", "interpolated"))  # noqa: E501
    @partial(eqx.filter_jit)
    def _call_(
        self,
        F: VectorField,
        y0: gt.BatchVec6,
        t0: Time,
        t1: Time,
        /,
        *,
        saveat: Times | None = None,  # not jitted here
        units: u.AbstractUnitSystem,
        interpolated: Literal[False, True] = False,
    ) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
        """Run the integrator.

        This handles the shape cases that `diffrax.diffeqsolve` can handle
        without application of `jax.vmap` or `jax.numpy.vectorize`.

        I/O shapes:

        - y0=(6,), t0=(), t1=(), saveat=() -> ()
        - y0=(6,), t0=(), t1=(), saveat=(T,) -> (T,)
        - y0=(*batch,6), t0=(), t1=(), saveat=() -> (*batch,)
        - y0=(*batch,6), t0=(), t1=(), saveat=(T) -> (*batch,T)

        Parameters
        ----------
        F : `galax.dynamics.integrate.VectorField`
            The field to integrate. Excluded from JIT.
        y0 : Array[float, (*batch, 6)]
            Initial conditions. Can have any (or no) batch dimensions. Included
            in JIT.
        t0, t1 : scalar
            Initial and final times. Included in JIT.

        saveat : (Quantity | Array)[float, (T,)] | None, optional
            Times to return the computation.  If `None`, the computation is
            returned only at the final time. Excluded from JIT.
        units : `unxt.AbstractUnitSystem`
            The unit system to use. Excluded from JIT.
        interpolated : bool, optional
            Whether to return an interpolated solution. Excluded from JIT.

        """
        # ---------------------------------------
        # Parse inputs

        time = units["time"]
        t0_: gt.RealScalar = u.Quantity.from_(t0, time).value
        t1_: gt.RealScalar = u.Quantity.from_(t1, time).value
        # Either save at `saveat` or at the final time.
        only_final = saveat is None or len(saveat) <= 1
        save_at = diffrax.SaveAt(
            t0=False,
            t1=only_final,
            ts=u.Quantity.from_(saveat, time).value if not only_final else None,
            dense=interpolated,
        )

        diffeq_kw = dict(self.diffeq_kw)
        if interpolated and diffeq_kw.get("max_steps") is None:
            diffeq_kw.pop("max_steps")

        # ---------------------------------------
        # Perform the integration

        soln = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(F),
            solver=self.Solver(**self.solver_kw),
            t0=t0_,
            t1=t1_,
            y0=y0,
            dt0=None,
            args=(),
            saveat=save_at,
            stepsize_controller=self.stepsize_controller,
            **diffeq_kw,
        )

        # Reshape (T, *batch) to (*batch, T)
        # soln.ts is already in the correct shape
        ys = xp.moveaxis(soln.ys, 0, -2)

        # Parse the solution, (unbatching time when saveat is None)
        t_dim_sel = -1 if saveat is None else slice(None)
        solt = soln.ts[..., t_dim_sel]
        solq = ys[..., t_dim_sel, 0:3]
        solp = ys[..., t_dim_sel, 3:6]

        # ---------------------------------------
        # Return

        if interpolated:
            out_cls = gc.InterpolatedPhaseSpacePosition
            out_kw = {"interpolant": Interpolant(soln.interpolation, units=units)}
        else:
            out_cls = gc.PhaseSpacePosition
            out_kw = {}

        return out_cls(  # shape = (*batch, T)
            t=u.Quantity(solt, time),
            q=u.Quantity(solq, units["length"]),
            p=u.Quantity(solp, units["speed"]),
            **out_kw,
        )

    # -----------------------------------------------------
    # Abstract call method

    @dispatch.abstract
    def __call__(self, F: VectorField, y0: Any, t0: Any, t1: Any, **kwargs: Any) -> Any:
        """Integrate the equations of motion.

        Broadly, the integrator takes the field ``F`` and integrates the initial
        conditions ``y0`` from time ``t0`` to time ``t1``.

        This is the abstract method for the integrator. Actual methods are
        registered for dispatching below.

        """

    # -------------------------------------------
    # Kwarg options

    @dispatch.multi(
        (Any, VectorField),  # (F,)
        (Any, VectorField, Any),  # (F, y0)
        (Any, VectorField, Any, Any),  # (F, y0, t0)
    )
    def __call__(
        self, F: VectorField, *args: Any, **kwargs: Any
    ) -> gc.PhaseSpacePosition | gc.InterpolatedPhaseSpacePosition:
        """Support keyword arguments by re-dispatching."""
        return self(F, *args, **kwargs)

    # -------------------------------------------
    # Scalar call

    @dispatch(precedence=2)
    @eqx.filter_jit  # @partial(jax.jit, static_argnums=(0, 1), static_argnames=("units", "interpolated"))  # noqa: E501
    def __call__(
        self,
        F: VectorField,
        y0: gt.BatchVec6,
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

        >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10., 0., 0.], "kpc"),
        ...                            p=u.Quantity([0., 200., 0.], "km/s")
        ...                            ).w(units="galactic")
        >>> w0.shape
        (6,)

        (Note that the ``t`` attribute is not used.)

        Now we can integrate the phase-space position for 1 Gyr, getting the
        final position.  The integrator accepts any function for the equations
        of motion.  Here we will reproduce what happens with orbit integrations.

        >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
        ...                             r_s=u.Quantity(5, "kpc"), units="galactic")

        >>> integrator = gd.integrate.Integrator()
        >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
        >>> w = integrator(pot._dynamics_deriv, w0, t0, t1, units=galactic)
        >>> w
        PhaseSpacePosition(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity[...](value=f64[], unit=Unit("Myr"))
        )
        >>> w.shape
        ()

        We can also request the orbit at specific times:

        >>> ts = u.Quantity(jnp.linspace(0, 1, 10), "Myr")  # 10 steps
        >>> ws = integrator(pot._dynamics_deriv, w0, t0, t1,
        ...                 saveat=ts, units=galactic)
        >>> ws
        PhaseSpacePosition(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity[...](value=f64[10], unit=Unit("Myr"))
        )
        >>> ws.shape
        (10,)

        """
        return self._call_(
            F, y0, t0, t1, saveat=saveat, units=units, interpolated=interpolated
        )

    # -------------------------------------------
    # Vectorized call

    @dispatch(precedence=1)
    @eqx.filter_jit
    def __call__(
        self,
        F: VectorField,
        y0: gt.BatchableVec6,
        t0: Shaped[u.Quantity["time"], "*#batch"] | Shaped[ArrayLike, "*#batch"] | Time,
        t1: Shaped[u.Quantity["time"], "*#batch"] | Shaped[ArrayLike, "*#batch"] | Time,
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

        >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([[10., 0, 0], [11., 0, 0]], "kpc"),
        ...                            p=u.Quantity([[0, 200, 0], [0, 210, 0]], "km/s"))

        Now we can integrate the phase-space position for 1 Gyr, getting the
        final position.  The integrator accepts any function for the equations
        of motion.  Here we will reproduce what happens with orbit integrations.

        >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
        ...                             r_s=u.Quantity(5, "kpc"), units="galactic")

        >>> integrator = gd.integrate.Integrator()
        >>> ws = integrator(pot._dynamics_deriv, w0, t0, t1, units=galactic)
        >>> ws.shape
        (2,)

        """
        # Vectorize the call
        # This depends on the shape of saveat
        vec_call = xp.vectorize(
            lambda *args: self._call_(*args, units=units, saveat=saveat, **kwargs),
            signature="(6),(),()->(T)" if saveat is not None else "(6),(),()->()",
            excluded=(0,),
        )

        # TODO: vectorize with units!
        time = units["time"]
        t0_: gt.VecTime = u.Quantity.from_(t0, time).value
        t1_: gt.VecTime = u.Quantity.from_(t1, time).value

        return vec_call(F, y0, t0_, t1_)


# -------------------------------------------
# w0 is a PhaseSpacePosition


@Integrator.__call__.dispatch
def call(
    self: Any,
    F: VectorField,
    w0: gc.AbstractPhaseSpacePosition,
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
    >>> import quaxed.numpy as xp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions and a potential:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10., 0., 0.], "kpc"),
    ...                            p=u.Quantity([0., 200., 0.], "km/s"))

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")

    We can integrate the phase-space position:

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(pot._dynamics_deriv, w0, t0, t1, units=galactic)
    >>> w
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity[...](value=f64[], unit=Unit("Myr"))
    )

    """
    return self(
        F,
        w0.w(units=units),
        t0,
        t1,
        saveat=saveat,
        units=units,
        interpolated=interpolated,
    )


@Integrator.__call__.dispatch
def call(
    self: Any,
    F: VectorField,
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
    >>> import quaxed.numpy as xp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions and a potential:

    >>> w01 = gc.PhaseSpacePosition(q=u.Quantity([10., 0., 0.], "kpc"),
    ...                             p=u.Quantity([0., 200., 0.], "km/s"))
    >>> w02 = gc.PhaseSpacePosition(q=u.Quantity([0., 10., 0.], "kpc"),
    ...                             p=u.Quantity([-200., 0., 0.], "km/s"))
    >>> w0 = gc.CompositePhaseSpacePosition(w01=w01, w02=w02)

    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")

    We can integrate the composite phase-space position:

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(pot._dynamics_deriv, w0, t0, t1, units=galactic)
    >>> w
    CompositePhaseSpacePosition({'w01': PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity...,
        'w02': PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity...
    )})

    """
    # TODO: Interpolated form
    return gc.CompositePhaseSpacePosition(
        **{
            k: self(F, v, t0, t1, saveat=saveat, units=units, interpolated=interpolated)
            for k, v in w0.items()
        }
    )


# ============================================================================
# Interpolant


class Interpolant(eqx.Module):  # type: ignore[misc]#
    """Wrapper for `diffrax.DenseInterpolation`.

    This satisfies the `galax.coordinates.PhaseSpacePositionInterpolant`
    Protocol.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions and a potential:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10., 0., 0.], "kpc"),
    ...                            p=u.Quantity([0., 200., 0.], "km/s"))
    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")

    We can integrate the phase-space position for 1 Gyr, getting the final
    position.  The integrator accepts any function for the equations of motion.
    Here we will reproduce what happens with orbit integrations.

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(pot._dynamics_deriv, w0, t0, t1, units=galactic,
    ...                interpolated=True)
    >>> type(w)
    <class 'galax.coordinates...InterpolatedPhaseSpacePosition'>

    >>> isinstance(w.interpolant, gc.PhaseSpacePositionInterpolant)
    True

    """

    interpolant: diffrax.DenseInterpolation
    """:class:`diffrax.DenseInterpolation` object.

    This object is the result of the integration and can be used to evaluate the
    interpolated solution at any time. However it does not understand units, so
    the input is the time in ``units["time"]``. The output is a 6-vector of
    (q, p) values in the units of the integrator.
    """

    units: u.AbstractUnitSystem = eqx.field(static=True, converter=u.unitsystem)
    """The :class:`unxt.AbstractUnitSystem`.

    This is used to convert the time input to the interpolant and the phase-space
    position output.
    """

    def __call__(self, t: u.Quantity["time"], **_: Any) -> Any:
        """Evaluate the interpolation."""
        # Parse t
        t_ = xp.atleast_1d(u.ustrip(self.units["time"], t))

        # Evaluate the interpolation
        ys = jax.vmap(self.interpolant.evaluate)(t_)

        # Reshape (T, *batch) to (*batch, T)
        # ts is already in the correct shape
        ys = xp.moveaxis(ys, 0, -2)
        # Reshape (*batch,T=1,6) to (*batch,6) if t is a scalar
        t_dim_sel = -1 if t.shape == () else slice(None)
        ys = ys[..., t_dim_sel, :]

        # Construct and return the result
        return gc.PhaseSpacePosition(
            q=u.Quantity(ys[..., 0:3], self.units["length"]),
            p=u.Quantity(ys[..., 3:6], self.units["speed"]),
            t=t,
        )
