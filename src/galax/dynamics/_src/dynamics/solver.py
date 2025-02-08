"""Dynamics Solvers.

This is private API.

"""

__all__ = ["DynamicsSolver"]


from functools import partial
from typing import Any, TypeAlias, final

import diffrax as dfx
import equinox as eqx
import jax.tree as jtu
from jaxtyping import PyTree
from plum import convert, dispatch

import coordinax as cx
import diffraxtra as dfxtra
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity as FastQ

import galax.coordinates as gc
import galax.dynamics._src.custom_types as gdt
import galax.typing as gt
from .compat import AllowValue
from .field_base import AbstractDynamicsField
from galax.dynamics._src.solver import (
    AbstractSolver,
    SolveState,
    Terms,
)
from galax.dynamics._src.utils import parse_saveat

BBtQParr: TypeAlias = tuple[gdt.BBtQarr, gdt.BBtParr]

default_saveat = dfx.SaveAt(t1=True)


def _parse_field(field: AbstractDynamicsField | Terms, solver: AbstractSolver) -> Terms:
    return field.terms(solver) if isinstance(field, AbstractDynamicsField) else field


@final
class DynamicsSolver(AbstractSolver, strict=True):  # type: ignore[call-arg]
    """Dynamics solver.

    The most useful method is `.solve()`, which handles initialization and
    stepping to the final solution. Manual solves can be done with `.init()` and
    repeat `.step()`, or `.init()` and `.run()`.

    Examples
    --------
    The ``.solve()`` method uses multiple dispatch to handle many different
    problem setups. Check out the method's docstring for examples. Here we show
    a simple example.

    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.DynamicsSolver()  # defaults to Dopri8

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
              ys=(f64[1,2,3], f64[1,2,3]), ... )

    >>> w = gc.PhaseSpacePosition.from_(soln, units=pot.units, frame=w0.frame)
    >>> print(w)
    PhaseSpacePosition(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[-5.151 -6.454 -5.795]]
             [[ 4.277  4.633  1.426]]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[[ 0.225 -0.068  0.253]]
             [[-0.439 -0.002 -0.146]]]>,
        t=Quantity['time'](Array([1000.], dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    The solver can be customized. Here are a few examples:

    1. From a `galax.dynamics.integrate.DiffEqSolver` instance. This allows for
       setting the `diffrax.AbstractSolver`,
       `diffrax.AbstractStepSizeController`, etc.

    >>> diffeqsolver = gd.solve.DiffEqSolver(dfx.Dopri8(),
    ...     stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))
    >>> solver = gd.DynamicsSolver(diffeqsolver)
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

    >>> solver = gd.DynamicsSolver({
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
    diffeqsolver: dfxtra.DiffEqSolver = eqx.field(
        default=dfxtra.DiffEqSolver(
            solver=dfx.Dopri8(),
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8),
            max_steps=2**16,
        ),
        converter=dfxtra.DiffEqSolver.from_,
    )

    # -------------------------------------------

    # def _init_impl(
    #     self,
    #     terms: Terms,
    #     t0: gt.RealSz0,
    #     y0: BBtQParr,
    #     args: PyTree,
    #     units: u.AbstractUnitSystem,
    # ) -> SolveState:
    #     if t0.ndim == 0:  # Scalar
    #         return super()._init_impl(terms, t0, y0, args, units)  # noqa: ERA001
    #     raise NotImplementedError("TODO")  # noqa: ERA001

    @dispatch.abstract
    def init(
        self: "DynamicsSolver", field: Any, t0: Any, t1: Any, y0: Any, args: Any
    ) -> Any:
        """Initialize the `galax.dynamics.solve.SolveState`.

        Examples
        --------
        >>> from dataclassish import replace
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> solver = gd.DynamicsSolver()

        Define the vector field. In this example it's to solve Hamilton's EoM in
        a gravitational potential.

        >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
        ...    r_s=u.Quantity(5, "kpc"), units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        Define the initial conditions, here a phase-space position

        >>> w0 = gc.PhaseSpacePosition(
        ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
        ...     p=u.Quantity([0, 220, 0], "km/s"),
        ...     t=u.Quantity(0, "Gyr"))

        Then the `galax.dynamics.solve.SolveState` can be initialized.

        >>> state = solver.init(field, w0, None)
        >>> state
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]),
            solver_state=(False, (f64[2,3], f64[2,3])),
            success=..., units=... )

        The state can be initialed many different ways:

        - rrom a `galax.coordinates.PhaseSpacePosition` without a time:

        >>> w0 = replace(w0, t=None)  # without a time
        >>> t0 = u.Quantity(0, "Gyr")
        >>> solver.init(field, w0, t0, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - from a `coordinax.frames.AbstractCoordinate`:

        >>> coord = cx.Coordinate({"length": w0.q, "speed": w0.p},
        ...                       frame=gc.frames.SimulationFrame())
        >>> solver.init(field, coord, t0, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - From a `coordinax.vecs.Space`:

        >>> space = coord.data
        >>> solver.init(field, space, t0, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        >>> space = cx.Space({"length": cx.vecs.FourVector(t0, w0.q), "speed": w0.p})
        >>> solver.init(field, space, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - From a tuple of `coordinax.vecs.AbstractVector`:

        >>> solver.init(field, (w0.q, w0.p), t0, None, units=pot.units)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - From a tuple of `unxt.Quantity`:

        >>> y0 = (u.Quantity([8, 0, 0], "kpc"), u.Quantity([0, 220, 0], "km/s"))
        >>> solver.init(field, y0, t0, None, units=pot.units)
        SolveState( t=weak_f64[], y=(f64[3], f64[3]), ... )

        - From a tuple of jax arrays.

        >>> y0 = (y0[0].ustrip("kpc"), y0[1].ustrip("kpc/Myr"))
        >>> solver.init(field, y0, t0, None, units=pot.units)
        SolveState( t=weak_f64[], y=(f64[3], f64[3]), ... )

        - From an (N, 6) array:

        >>> y0 = jnp.concatenate(y0)
        >>> solver.init(field, y0, t0, None, units=pot.units)
        SolveState( t=weak_f64[], y=(f64[3], f64[3]), ... )

        - From a `galax.coordinates.CompositePhaseSpacePosition`:

        >>> w01 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
        ...                             p=u.Quantity([0, 200, 0], "km/s"),
        ...                             t=u.Quantity(0, "Gyr"))
        >>> w02 = gc.PhaseSpacePosition(q=u.Quantity([0, 10, 0], "kpc"),
        ...                             p=u.Quantity([-200, 0, 0], "km/s"),
        ...                             t=u.Quantity(0, "Gyr"))
        >>> w0s = gc.CompositePhaseSpacePosition(w01=w01, w02=w02)

        >>> solver.init(field, w0s, None)
        {'w01': SolveState( t=weak_f64[], y=(f64[3], f64[3]), ... ),
         'w02': SolveState( t=weak_f64[], y=(f64[3], f64[3]), ... )}

        """
        # See dispatches below
        raise NotImplementedError  # pragma: no cover

    # -------------------------------------------

    def _step_impl(
        self,
        terms: Terms,
        state: SolveState,
        t1: gt.RealSz0,
        args: Any,
        step_kw: dict[str, Any],
    ) -> SolveState:
        step_output = self.diffeqsolver.solver.step(
            terms,
            state.t,
            t1,
            state.y,
            args=args,
            solver_state=state.solver_state,
            **step_kw,
        )
        return SolveState.from_step_output(t1, step_output, state.units)

    def step(
        self: "DynamicsSolver",
        field: AbstractDynamicsField | Terms,
        state: SolveState,
        t1: Any,
        /,
        args: PyTree,
        **step_kwargs: Any,  # e.g. solver_state, made_jump
    ) -> SolveState:
        """Step the state.

        Note that this will NOT create a `diffrax.Solution` object and nor any
        dense output. Use ``.solve`` to get a (dense) `diffrax.Solution` object.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> solver = gd.DynamicsSolver()

        Define the vector field. In this example it's to solve Hamilton's EoM in
        a gravitational potential.

        >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
        ...    r_s=u.Quantity(5, "kpc"), units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        Define the initial conditions:

        >>> w0 = gc.PhaseSpacePosition(
        ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
        ...     p=u.Quantity([0, 220, 0], "km/s"),
        ...     t=u.Quantity(0, "Gyr"))

        Initialize the state.

        >>> state = solver.init(field, w0, None)
        >>> state.y
        (Array([[8., 0., 9.], [9., 0., 3.]], dtype=float64),
         Array([[0. , 0.22499668, 0. ],
                [0. , 0.22499668, 0. ]], dtype=float64))

        Evolve the state to `t1`.

        >>> t1 = u.Quantity(10, "Myr")
        >>> state = solver.step(field, state, t1, None)
        >>> state.y
        (Array([[7.48122073, 2.20035764, 8.41637332],
                [7.96890464, 2.16097321, 2.65630155]], dtype=float64),
         Array([[-0.104613  ,  0.20983038, -0.11768962],
                [-0.20914465,  0.19739402, -0.06971488]], dtype=float64))

        """
        terms = _parse_field(field, self)
        t1_ = u.ustrip(AllowValue, state.units["time"], t1)
        step_kwargs.setdefault("made_jump", False)
        return self._step_impl(terms, state, t1_, args, step_kwargs)

    # -------------------------------------------

    @dispatch.abstract
    def run(
        self, field: Any, state: SolveState, t1: Any, args: PyTree, **solver_kw: Any
    ) -> SolveState:
        """Run the state to `t1`.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> solver = gd.DynamicsSolver()

        Define the vector field. In this example it's to solve Hamilton's EoM in
        a gravitational potential.

        >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
        ...    r_s=u.Quantity(5, "kpc"), units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        Define the initial conditions:

        >>> w0 = gc.PhaseSpacePosition(
        ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
        ...     p=u.Quantity([0, 220, 0], "km/s"),
        ...     t=u.Quantity(0, "Gyr"))

        Initialize the state.

        >>> state = solver.init(field, w0, None)
        >>> state.y
        (Array([[8., 0., 9.], [9., 0., 3.]], dtype=float64),
         Array([[0. , 0.22499668, 0. ],
                [0. , 0.22499668, 0. ]], dtype=float64))

        Evolve the state to `t1`.

        >>> t1 = u.Quantity(1, "Gyr")
        >>> state = solver.run(field, state, t1, None)
        >>> state.y
        (Array([[-5.15111583, -6.45413687, -5.79500531],
                [ 4.2771096 ,  4.63284576,  1.4257032 ]], dtype=float64),
         Array([[ 0.22466724, -0.06793485,  0.25275065],
                [-0.43921376, -0.0023005 , -0.14640459]], dtype=float64))

        """
        raise NotImplementedError  # pragma: no cover

    # -------------------------------------------

    # TODO: dispatch where the state from `init` is accepted
    @dispatch.abstract
    def solve(
        self: "DynamicsSolver",
        field: Any,
        w0: Any,
        t0: Any,
        t1: Any,
        /,
        args: Any = (),
        *,
        unbatch_time: bool = False,
        **solver_kw: Any,  # TODO: TypedDict
    ) -> dfx.Solution:
        """Solve the dynamics with `diffrax.diffeqsolve`.

        In ``solver_kw``, the following keys are recognized:

        - All keys recognized by `diffrax.diffeqsolve`. In particular if "dt0"
          is not specified it is assumed to be `None`.
        - "dense" (bool): If `True`, `saveat` is modified to have
          ``dense=True``.
        - "vectorize_interpolation" (bool): If `True`, the interpolation is
          vectorized using
          `galax.dynamics.integrate.VectorizedDenseInterpolation`.

        The output shape aligns with `diffrax.diffeqsolve`: (*batch, [time],
        *shape), where [time] is >= 1. The `unbatch_time` keyword argument can
        be used to squeeze scalar times.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> solver = gd.DynamicsSolver()

        Specify the vector field.

        >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
        ...                             r_s=u.Quantity(5, "kpc"), units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        The solver is very flexible. Here we show a few examples of variety of
        initial conditions.

        - tuple of `unxt.Quantity`:

        >>> w0 = (u.Quantity([8, 0, 0], "kpc"),
        ...       u.Quantity([0, 220, 0], "km/s"))

        Solve EoM from `t0` to `t1`, returning the solution at `t1`.

        >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
        >>> soln = solver.solve(field, w0, t0, t1)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[1],
                  ys=(f64[1,3], f64[1,3]), ... )
        >>> soln.ys
        (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
         Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

        This can be solved for a specific set of times, not just `t1`.

        >>> soln = solver.solve(field, w0, t0, t1, saveat=u.Quantity(0.5, "Gyr"))
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[1],
                  ys=(f64[1,3], f64[1,3]), ... )
        >>> soln.ys
        (Array([[-0.83933788, -7.73317472,  0. ]], dtype=float64),
         Array([[ 0.21977442, -0.1196412 ,  0. ]], dtype=float64))

        >>> soln = solver.solve(field, w0, t0, t1,
        ...     saveat=u.Quantity([0.25, 0.5], "Gyr"))
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[2],
                  ys=(f64[2,3], f64[2,3]), ... )

        If ``unbatch_time=True``, the time dimension is squeezed if it is
        scalar.

        >>> soln = solver.solve(field, w0, t0, t1,
        ...      saveat=u.Quantity(0.5, "Gyr"), unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ... )

        A set of initial conditions can be solved at once. The resulting
        `diffrax.Solution` has a `ys` shape of ([time], *shape, 3),

        >>> w0s = (u.Quantity([[8, 0, 0], [9, 0, 0]], "kpc"),
        ...       u.Quantity([[0, 220, 0], [0, 230, 0]], "km/s"))

        >>> soln = solver.solve(field, w0s, t0, t1)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[1],
                  ys=(f64[1,2,3], f64[1,2,3]), ... )

        This can be batched with a set of times. The resulting
        `diffrax.Solution` has a `ys` shape of (*batch, [time], *shape, 3).



        >>> t1 = u.Quantity([1, 1.1, 1.2], "Gyr")
        >>> soln = solver.solve(field, w0, t0, t1)
        >>> soln
        Solution( t0=f64[3], t1=f64[3], ts=f64[3,1],
                  ys=(f64[3,1,3], f64[3,1,3]), ... )

        All these examples can be interpolated as dense solutions. If the time
        or position are batched, then the `diffrax.DenseInterpolation` needs to
        be vectorized into a
        `galax.dynamics.integrate.VectorizedDenseInterpolation`. This can be
        done by passing ``vectorize_interpolation=True``. To emphasize the
        differences, let's batch across different start times.

        >>> t0 = u.Quantity([-1, -0.5, 0], "Gyr")
        >>> t1 = u.Quantity(1, "Gyr")
        >>> soln = solver.solve(field, w0, t0, t1, dense=True,
        ...                     vectorize_interpolation=True)
        >>> newq, newp = soln.evaluate(0.5)  # Myr
        >>> print((newq.round(3), newp.round(3)))
        (Array([[-7.034,  1.538,  0.   ],
                [-0.729, -7.79 ,  0.   ],
                [ 7.997,  0.112,  0.   ]], dtype=float64),
         Array([[-0.232, -0.205,  0.   ],
                [ 0.221, -0.106,  0.   ],
                [-0.013,  0.225,  0.   ]], dtype=float64))

        Now let's explore some more options for the initial conditions.

        - `coordinax.vecs.AbstractPos3D`, `coordinax.vecs.AbstractVel3D`:

        >>> w0 = (cx.CartesianPos3D.from_([8, 0, 0], "kpc"),
        ...       cx.CartesianVel3D.from_([0, 220, 0], "km/s"))

        >>> t0, t1 = u.Quantity([0, 1], "Gyr")
        >>> soln = solver.solve(field, w0, t0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `coordinax.vecs.FourVector`, `coordinax.vecs.AbstractVel3D`:

        >>> w0 = (cx.vecs.FourVector.from_([0, 8, 0, 0], "kpc"),
        ...       cx.CartesianVel3D.from_([0, 220, 0], "km/s"))

        >>> soln = solver.solve(field, w0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `coordinax.Space`:

        >>> w0 = cx.Space(length=cx.CartesianPos3D.from_([8, 0, 0], "kpc"),
        ...               speed=cx.CartesianVel3D.from_([0, 220, 0], "km/s"))

        >>> soln = solver.solve(field, w0, t0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        >>> w0 = cx.Space(length=cx.vecs.FourVector.from_([0, 8, 0, 0], "kpc"),
        ...               speed=cx.CartesianVel3D.from_([0, 220, 0], "km/s"))

        >>> soln = solver.solve(field, w0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `coordinax.frames.AbstractCoordinate`:

        >>> w0 = cx.Coordinate(
        ...     {"length": cx.CartesianPos3D.from_([8, 0, 0], "kpc"),
        ...      "speed": cx.CartesianVel3D.from_([0, 220, 0], "km/s")},
        ...     gc.frames.SimulationFrame()
        ... )

        >>> soln = solver.solve(field, w0, t0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `galax.coordinates.PhaseSpacePosition` with time:

        >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([8, 0, 0], "kpc"),
        ...                            p=u.Quantity([0, 220, 0], "km/s"),
        ...                            t=u.Quantity(0, "Gyr"))

        >>> soln = solver.solve(field, w0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `galax.coordinates.PhaseSpacePosition` without time:

        >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([8, 0, 0], "kpc"),
        ...                            p=u.Quantity([0, 220, 0], "km/s"))

        >>> soln = solver.solve(field, w0, t0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `galax.coordinates.AbstractCompositePhaseSpacePosition`:

        >>> w01 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
        ...                             p=u.Quantity([0, 200, 0], "km/s"))
        >>> w02 = gc.PhaseSpacePosition(q=u.Quantity([0, 10, 0], "kpc"),
        ...                             p=u.Quantity([-200, 0, 0], "km/s"))
        >>> w0s = gc.CompositePhaseSpacePosition(w01=w01, w02=w02)

        >>> soln = solver.solve(field, w0s, t0, t1, unbatch_time=True)
        >>> soln
        {'w01': Solution( t0=f64[], t1=f64[], ts=f64[],
                        ys=(f64[3], f64[3]), ... ),
        'w02': Solution( t0=f64[], t1=f64[], ts=f64[],
                        ys=(f64[3], f64[3]), ... )}

        """
        raise NotImplementedError  # pragma: no cover


# ===============================================
# Helper


@dispatch
def parse_to_y0(
    qp: tuple[gdt.BBtQ, gdt.BBtP] | BBtQParr, units: u.AbstractUnitSystem, /
) -> BBtQParr:
    return tuple(
        jnp.broadcast_arrays(
            u.ustrip(AllowValue, units["length"], qp[0]).astype(float),
            u.ustrip(AllowValue, units["speed"], qp[1]).astype(float),
        )
    )


@dispatch
def parse_to_y0(qp: gt.BBtSz6, _: u.AbstractUnitSystem, /) -> BBtQParr:
    return jnp.broadcast_arrays(qp[..., :3].astype(float), qp[..., 3:].astype(float))


@dispatch
def parse_to_y0(
    q3p3: tuple[cx.vecs.AbstractPos3D, cx.vecs.AbstractVel3D],
    units: u.AbstractUnitSystem,
    /,
) -> BBtQParr:
    return parse_to_y0((convert(q3p3[0], FastQ), convert(q3p3[1], FastQ)), units)


# ===============================================
# Init Dispatches


@DynamicsSolver.init.dispatch
def init(
    self: DynamicsSolver,
    field: AbstractDynamicsField | Terms,
    qp: tuple[gdt.BBtQ, gdt.BBtP]
    | BBtQParr
    | gt.BBtSz6
    | tuple[cx.vecs.AbstractPos3D, cx.vecs.AbstractVel3D],
    t0: gt.RealQuSz0 | gt.RealSz0,
    args: PyTree,
    /,
    *,
    units: u.AbstractUnitSystem | None = None,
) -> SolveState:
    """Initialize from terms, unit/array tuple, and time."""
    if isinstance(field, AbstractDynamicsField):
        terms = field.terms(self.diffeqsolver)
        units = units if units is not None else field.units
        units = eqx.error_if(units, units != field.units, "units must match field")
    else:
        terms = field
        units = eqx.error_if(units, units is None, "units must be specified")

    y0 = parse_to_y0(qp, units)
    t0_ = u.ustrip(AllowValue, units["time"], t0)
    return self._init_impl(terms, t0_, y0, args, units)


# --------------------------------
# Tuple of Vector


@DynamicsSolver.init.dispatch
def init(
    self: DynamicsSolver,
    field: AbstractDynamicsField | Terms,
    q4p3: tuple[cx.vecs.FourVector, cx.vecs.AbstractVel3D],
    args: PyTree,
    /,
    *,
    units: u.AbstractUnitSystem | None = None,
) -> SolveState:
    return self.init(field, (q4p3[0].q, q4p3[1]), q4p3[0].t, args, units=units)


# --------------------------------
# Vector Space


# TODO: consolidate into main method
@DynamicsSolver.init.dispatch
def init(
    self: DynamicsSolver,
    field: AbstractDynamicsField | Terms,
    space: cx.Space,
    t0: gt.BBtRealQuSz0,
    args: PyTree,
    /,
    *,
    units: u.AbstractUnitSystem | None = None,
) -> SolveState:
    q, p = space["length"], space["speed"]
    q = eqx.error_if(
        q,
        isinstance(q, cx.vecs.FourVector),
        "space['length'] must not be a 4-vector if `t0` is given.",
    )
    return self.init(field, (q, p), t0, args, units=units)


# TODO: consolidate into main method
@DynamicsSolver.init.dispatch
def init(
    self: DynamicsSolver,
    field: AbstractDynamicsField | Terms,
    space: cx.Space,
    args: PyTree,
    /,
    *,
    units: u.AbstractUnitSystem | None = None,
) -> SolveState:
    q, p = space["length"], space["speed"]
    q = eqx.error_if(
        q,
        not isinstance(q, cx.vecs.FourVector),
        "space['length'] must be a 4-vector if `t0` is not given.",
    )
    return self.init(field, (q, p), args, units=units)


# --------------------------------
# Coordinate


# TODO: consolidate into main method
@DynamicsSolver.init.dispatch
def init(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    coord: cx.frames.AbstractCoordinate,
    t0: gt.BBtRealQuSz0,
    args: PyTree,
    /,
) -> SolveState:
    return self.init(field, coord.data, t0, args)


# --------------------------------
# PSPs


# TODO: consolidate into main method
@DynamicsSolver.init.dispatch
def init(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0: gc.AbstractOnePhaseSpacePosition,
    args: PyTree,
    /,
) -> SolveState:
    # Check that the initial conditions are valid.
    w0 = eqx.error_if(
        w0, w0.t is None, "If `t0` is not specified, `w0.t` must supply it."
    )
    w0 = eqx.error_if(  # TODO: remove when frames are handled
        w0,
        not isinstance(w0.frame, gc.frames.SimulationFrame),
        "Only SimulationFrame is currently supported.",
    )

    return self.init(field, (w0.q, w0.p), w0.t, args)


# TODO: consolidate into main method
@DynamicsSolver.init.dispatch
def init(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0: gc.AbstractOnePhaseSpacePosition,
    t0: gt.BBtRealQuSz0,
    /,
    args: PyTree,
) -> SolveState:
    w0 = eqx.error_if(
        w0,
        False if w0.t is None else jnp.logical_not(jnp.array_equal(w0.t, t0)),
        "If `t0` is specified, `w0.t` must be `None` or `t0`.",
    )
    w0 = eqx.error_if(  # TODO: remove when frames are handled
        w0,
        not isinstance(w0.frame, gc.frames.SimulationFrame),
        "Only SimulationFrame is currently supported.",
    )
    return self.init(field, (w0.q, w0.p), t0, args)


# --------------------------------
# Composite PSPs


@DynamicsSolver.init.dispatch
def init(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    w0s: gc.AbstractCompositePhaseSpacePosition,
    args: PyTree,
    /,
) -> dict[str, SolveState]:
    return {k: self.init(field, w0, args) for k, w0 in w0s.items()}


# ===============================================
# Run Dispatches


@DynamicsSolver.run.dispatch
def run(
    self: DynamicsSolver,
    field: AbstractDynamicsField | Terms,
    state: SolveState,
    t1: Any,
    args: PyTree,
    /,
    **solver_kw: Any,
) -> SolveState:
    terms = _parse_field(field, self)  # Parse the terms
    t1_ = u.ustrip(AllowValue, state.units["time"], t1)  # Parse the time
    # Validate the solver keyword arguments
    solver_kw = eqx.error_if(
        solver_kw, "saveat" in solver_kw, "cannot specify `saveat`"
    )

    # Run the solver
    soln = self._run_impl(terms, state, t1_, args, solver_kw)

    # Unbatch the time and state if possible
    if soln.ts.shape[soln.t0.ndim] == 1:
        soln = eqx.tree_at(lambda tree: tree.ts, soln, soln.ts[0])
        soln = eqx.tree_at(lambda tree: tree.ys, soln, jtu.map(lambda y: y[0], soln.ys))

    # Unpack the Solution
    return SolveState(
        t=t1_,
        y=soln.ys,
        solver_state=soln.solver_state,
        success=soln.result,
        units=state.units,
    )


@DynamicsSolver.run.dispatch
def run(
    self: DynamicsSolver,
    field: AbstractDynamicsField | Terms,
    state: dict[str, SolveState],
    t1: Any,
    args: PyTree,
    /,
) -> dict[str, SolveState]:
    return {k: self.run(field, v, t1, args) for k, v in state.items()}


# ===============================================
# Solve Dispatches
# TODO: refactor these so there's less code duplication
# by leveraging more the `.init` and `._run_impl` dispatches.

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
    *,
    unbatch_time: bool = False,
    **solver_kw: Any,
) -> dfx.Solution:
    """Solve for batch position tuple, scalar start, end time."""
    # Parse inputs
    terms = field.terms(self.diffeqsolver)
    usys = field.units

    # Initialize the state
    state = self.init(terms, qp, t0, args, units=usys)
    # Run the solver
    saveat = parse_saveat(usys, saveat, dense=solver_kw.pop("dense", None))
    solver_kw["saveat"] = saveat
    soln = self._run_impl(
        terms, state, t1.ustrip(usys["time"]), args=args, solver_kw=solver_kw
    )

    # Check to see if we should try to unbatch in the time dimension.
    if unbatch_time and soln.ts.shape[soln.t0.ndim] == 1:
        soln = eqx.tree_at(lambda tree: tree.ts, soln, soln.ts[0])
        soln = eqx.tree_at(lambda tree: tree.ys, soln, jtu.map(lambda y: y[0], soln.ys))

    return soln


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
    """Solve for batch position tuple, batched start, end time."""
    # Because of how JAX does its tree building over vmaps, vectorizing the
    # interpolation within the loop does not work, it needs to be done after.
    vectorize_interpolation = solver_kw.pop("vectorize_interpolation", False)

    # Build the vectorized solver. To get the right broadcasting the Q, P need
    # to be split up, then recombined.
    @partial(jnp.vectorize, signature="(3),(3),(),()->()")
    def call(q: gdt.Q, p: gdt.P, t0: gt.RealQuSz0, t1: gt.RealQuSz0) -> dfx.Solution:
        return self.solve(field, (q, p), t0, t1, args=args, saveat=saveat, **solver_kw)

    # Solve the batched problem
    soln = call(qp[0], qp[1], t0, t1)

    # Now possibly vectorize the interpolation
    if vectorize_interpolation:
        soln = dfxtra.VectorizedDenseInterpolation.apply_to_solution(soln)

    return soln


@DynamicsSolver.solve.dispatch
@eqx.filter_jit
def solve(
    self: DynamicsSolver,
    field: AbstractDynamicsField,
    qp: gt.BBtSz6,
    t0: gt.BBtRealQuSz0,
    t1: gt.BBtRealQuSz0,
    /,
    args: Any = (),
    saveat: Any = default_saveat,
    **solver_kw: Any,
) -> dfx.Solution:
    """Solve for q,p array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.DynamicsSolver()

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> qp = jnp.asarray([[8, 0, 0, 0, 0.22499668, 0]])
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")

    >>> soln = solver.solve(field, qp, t0, t1)
    >>> soln.ys
    (Array([[[4.22038796, 1.72855685, 0. ]]], dtype=float64),
     Array([[[-0.94723615,  0.03853245,  0. ]]], dtype=float64))

    """
    units = field.units
    y0 = (FastQ(qp[..., :3], units["length"]), FastQ(qp[..., 3:], units["speed"]))
    return self.solve(field, y0, t0, t1, args=args, saveat=saveat, **solver_kw)


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
    """Solve for position vector tuple, start, end time."""
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
    """Solve for 4-vector position tuple, end time."""
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
    """Solve for Space[4vec, 3vel], end time."""
    q4, p = space["length"], space["speed"]
    q4 = eqx.error_if(
        q4,
        not isinstance(q4, cx.vecs.FourVector),
        "space['length'] must be a 4-vector if `t0` is not given.",
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
    """Solve for Space[3vec, 3vel], start, end time."""
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
    """Solve for `coordinax.frames.AbstractCoordinate`, start, end time."""
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
    """Solve for PSP with time, end time."""
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
    """Solve for PSP without time, start, end time."""
    # Check that the initial conditions are valid.
    w0 = eqx.error_if(
        w0,
        False if w0.t is None else jnp.logical_not(jnp.array_equal(w0.t, t0)),
        "If `t0` is specified, `w0.t` must be `None` or `t0`.",
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
    """Solve for CompositePhaseSpacePosition, start, end time."""
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

    >>> solver = gd.DynamicsSolver(dfx.Dopri8())

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    """
    return self.terms(wrapper.diffeqsolver)
