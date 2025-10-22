"""Dynamics Solvers.

This is private API.

"""

__all__ = ["OrbitSolver"]


import functools as ft
from dataclasses import KW_ONLY
from typing import Any, TypeAlias, final

import diffrax as dfx
import equinox as eqx
import jax
import jax.extend as jex
import jax.tree as jtu
from jaxtyping import PyTree
from plum import dispatch

import diffraxtra as dfxtra
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
import galax.coordinates as gc
import galax.dynamics._src.custom_types as gdt
from .field_base import AbstractOrbitField
from galax.dynamics._src.solver import AbstractSolver, SolveState, Terms
from galax.dynamics._src.utils import parse_saveat, parse_to_t_y
from galax.dynamics.fields import AbstractField
from galax.utils import loop_strategies as lstrat

BBtQParr: TypeAlias = tuple[gdt.BBtQarr, gdt.BBtParr]

default_saveat = dfx.SaveAt(t1=True)


@final
class OrbitSolver(AbstractSolver):
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

    >>> solver = gd.OrbitSolver()  # defaults to Dopri8

    Define the vector field. In this example it's to solve Hamilton's EoM in a
    gravitational potential.

    >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Define the initial conditions, here a phase-space position

    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))

    Solve, stepping from `w0.t` to `t1`.

    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=(f64[1,2,3], f64[1,2,3]), ... )

    >>> w = gc.PhaseSpaceCoordinate.from_(soln, units=pot.units, frame=w0.frame)
    >>> print(w)
    PhaseSpaceCoordinate(
        q=<CartesianPos3D: (x, y, z) [kpc]
            [[-5.151 -6.454 -5.795]
             [ 4.277  4.633  1.426]]>,
        p=<CartesianVel3D: (x, y, z) [kpc / Myr]
            [[ 0.225 -0.068  0.253]
             [-0.439 -0.002 -0.146]]>,
        t=Quantity['time'](1000., unit='Myr'),
        frame=SimulationFrame())

    The solver can be customized. Here are a few examples:

    1. From a `galax.dynamics.integrate.DiffEqSolver` instance.

    >>> diffeqsolver = gd.DiffEqSolver(dfx.Dopri8(),
    ...     stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))
    >>> solver = gd.OrbitSolver.from_(diffeqsolver)
    >>> solver
    OrbitSolver(
        solver=Dopri8(),
        stepsize_controller=PIDController(rtol=1e-05, atol=1e-05),
        adjoint=RecursiveCheckpointAdjoint(),
        max_steps=4096
    )

    2. A `dict` of keyword arguments:

    >>> solver = gd.OrbitSolver.from_({
    ...     "solver": dfx.Dopri8(), "stepsize_controller": dfx.ConstantStepSize()})
    >>> solver
    OrbitSolver(solver=Dopri8(), stepsize_controller=ConstantStepSize())

    """

    #: The solver for the differential equation.
    #: See the diffrax guide on how to choose a solver.
    solver: dfx.AbstractSolver[Any] = dfx.Dopri8()

    _: KW_ONLY

    #: How to change the step size as the integration progresses.
    #: See diffrax's list of stepsize controllers.
    stepsize_controller: dfx.AbstractStepSizeController[Any, Any] = dfx.PIDController(
        rtol=1e-8, atol=1e-8
    )

    #: How to differentiate in `diffeqsolve`.
    #: See `diffrax` for options.
    adjoint: dfx.AbstractAdjoint = dfx.ForwardMode()

    #: Event. Can override the `event` argument when calling `DiffEqSolver`
    event: dfx.Event | None = None

    #: The maximum number of steps to take before quitting.
    #: Some `diffrax.SaveAt` options can be incompatible with `max_steps=None`,
    #: so you can override the `max_steps` argument when calling `DiffEqSolver`
    max_steps: int | None = eqx.field(default=2**16, static=True)

    # -------------------------------------------

    @dispatch.abstract
    def init(
        self: "OrbitSolver", field: Any, t0: Any, t1: Any, y0: Any, args: Any
    ) -> Any:
        """Initialize the `galax.dynamics.solve.SolveState`.

        Examples
        --------
        >>> from dataclassish import replace
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> solver = gd.OrbitSolver()

        Define the vector field. In this example it's to solve Hamilton's EoM in
        a gravitational potential.

        >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        Define the initial conditions, here a phase-space position

        >>> w0 = gc.PhaseSpaceCoordinate(
        ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
        ...     p=u.Quantity([0, 220, 0], "km/s"),
        ...     t=u.Quantity(0, "Gyr"))

        Then the `galax.dynamics.solve.SolveState` can be initialized.

        >>> state = solver.init(field, w0, None)
        >>> state
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]),
            solver_state=(False, (f64[2,3], f64[2,3])),
            success=..., units=... )

        The state can be initialed many different ways. Let's work up the type
        ladder:

        - From a tuple of `jax.Array`: The (q,p) tuple is the natural PyTree
          structure of the dynamics solver.

        >>> y0 = (jnp.array([8,0,0]), jnp.array([0,0.22499669,0]))  # ([kpc], [kpc/Myr])
        >>> t0 = 0
        >>> solver.init(field, y0, t0, None)
        SolveState( t=weak_i64[], y=(f64[3], f64[3]), ... )

        - From an (N, 6) `jax.Array`:

        >>> y0 = jnp.concatenate(y0)
        >>> solver.init(field, y0, t0, None)
        SolveState( t=weak_i64[], y=(f64[3], f64[3]), ... )

        - From a tuple of `unxt.Quantity`:

        >>> y0 = (u.Quantity([8, 0, 0], "kpc"), u.Quantity([0, 220, 0], "km/s"))
        >>> t0 = u.Quantity(0, "Gyr")
        >>> solver.init(field, y0, t0, None)
        SolveState( t=weak_f64[], y=(f64[3], f64[3]), ... )

        - From a tuple of `coordinax.vecs.AbstractVector`:

        >>> q0 = cx.vecs.CartesianPos3D.from_([[8, 0, 0], [9, 0, 0]], "kpc")
        >>> p0 = cx.vecs.CartesianVel3D.from_([0, 220, 0], "km/s")
        >>> solver.init(field, (q0, p0), t0, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - From a `coordinax.vecs.KinematicSpace`:

        >>> space = cx.KinematicSpace(length=q0, speed=p0)
        >>> solver.init(field, space, t0, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        >>> space = cx.KinematicSpace(length=cx.vecs.FourVector(t0, q0), speed=p0)
        >>> solver.init(field, space, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - from a `coordinax.frames.AbstractCoordinate`:

        >>> coord = cx.Coordinate({"length": w0.q, "speed": w0.p},
        ...                       frame=gc.frames.simulation_frame)
        >>> solver.init(field, coord, t0, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - from a `galax.coordinates.PhaseSpacePosition` (no time):

        >>> w0 = gc.PhaseSpacePosition(q=w0.q, p=w0.p)  # no time
        >>> t0 = u.Quantity(0, "Gyr")
        >>> solver.init(field, w0, t0, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - From a `galax.coordinates.PhaseSpaceCoordinate`:

        >>> w0 = gc.PhaseSpaceCoordinate(
        ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
        ...     p=u.Quantity([0, 220, 0], "km/s"),
        ...     t=u.Quantity(0, "Gyr"))
        >>> solver.init(field, w0, None)
        SolveState( t=weak_f64[], y=(f64[2,3], f64[2,3]), ... )

        - From a `galax.coordinates.CompositePhaseSpaceCoordinate`:

        >>> w01 = gc.PhaseSpaceCoordinate(q=u.Quantity([10, 0, 0], "kpc"),
        ...                               p=u.Quantity([0, 200, 0], "km/s"),
        ...                               t=u.Quantity(0, "Gyr"))
        >>> w02 = gc.PhaseSpaceCoordinate(q=u.Quantity([0, 10, 0], "kpc"),
        ...                               p=u.Quantity([-200, 0, 0], "km/s"),
        ...                               t=u.Quantity(0, "Gyr"))
        >>> w0s = gc.CompositePhaseSpaceCoordinate(w01=w01, w02=w02)

        >>> solver.init(field, w0s, None)
        {'w01': SolveState( t=weak_f64[], y=(f64[3], f64[3]), ... ),
         'w02': SolveState( t=weak_f64[], y=(f64[3], f64[3]), ... )}

        """
        # See dispatches below
        raise NotImplementedError  # pragma: no cover

    # -------------------------------------------

    def step(
        self: "OrbitSolver",
        field: AbstractOrbitField | Terms,
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

        >>> solver = gd.OrbitSolver()

        Define the vector field. In this example it's to solve Hamilton's EoM in
        a gravitational potential.

        >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        Define the initial conditions:

        >>> w0 = gc.PhaseSpaceCoordinate(
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
        t1_ = u.ustrip(AllowValue, state.units["time"], t1)
        step_kwargs.setdefault("made_jump", False)
        # TODO: figure out stepping over batched `state`
        outstate: SolveState = self._step_impl_scalar(
            field, state, t1_, args, step_kwargs
        )
        return outstate

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

        >>> solver = gd.OrbitSolver()

        Define the vector field. In this example it's to solve Hamilton's EoM in
        a gravitational potential.

        >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        Define the initial conditions:

        >>> w0 = gc.PhaseSpaceCoordinate(
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
         Array([[ 0.22466725, -0.06793485,  0.25275065],
                [-0.43921376, -0.0023005 , -0.14640459]], dtype=float64))

        """
        raise NotImplementedError  # pragma: no cover

    # -------------------------------------------

    # TODO: dispatch where the state from `init` is accepted
    @dispatch.abstract
    def solve(
        self: "OrbitSolver",
        field: AbstractField,
        w0: Any,
        t0: Any,
        t1: Any,
        /,
        args: Any = None,
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

        >>> solver = gd.OrbitSolver()

        Specify the vector field.

        >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        Solve for a single set of initial conditions.

        >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([8, 0, 0], "kpc"),
        ...                            p=u.Quantity([0, 220, 0], "km/s"),
        ...                            t=u.Quantity(0, "Myr"))
        >>> t1 = u.Quantity(1, "Gyr")

        >>> soln = solver.solve(field, w0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)
        >>> soln.ys
        (Array([-6.91453518,  1.64014782,  0.        ], dtype=float64),
         Array([-0.24701038, -0.20172576,  0.        ], dtype=float64))

        ## Input Types

        The solver is very flexible. Let's work up the type ladder:

        - From a tuple of `jax.Array`: The (q,p) tuple is the natural PyTree
          structure of the dynamics solver.

        >>> xyz0 = jnp.array([8,0,0])  # [kpc]
        >>> v_xyz = jnp.array([0,0.22499669,0])  # [kpc/Myr]
        >>> t0, t1 = 0, 1000  # [Myr]
        >>> soln = solver.solve(field, (xyz0, v_xyz), t0, t1)
        >>> soln.ys
        (Array([[-6.91453209,  1.64014633,  0.        ]], dtype=float64),
         Array([[-0.24701075, -0.20172583,  0.        ]], dtype=float64))

        - From an (N, 6) `jax.Array`:

        >>> w0 = jnp.concatenate((xyz0, v_xyz))
        >>> soln = solver.solve(field, w0, t0, t1)
        >>> soln.ys
        (Array([[-6.91453209,  1.64014633,  0.        ]], dtype=float64),
         Array([[-0.24701075, -0.20172583,  0.        ]], dtype=float64))

        - tuple of `unxt.Quantity`:

        >>> xyz0 = u.Quantity([8, 0, 0], "kpc")
        >>> v_xyz = u.Quantity([0, 220, 0], "km/s")
        >>> w0 = (xyz0, v_xyz)

        >>> t0, t1 = u.Quantity([0, 1], "Gyr")
        >>> soln = solver.solve(field, w0, t0, t1)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[1],
                  ys=(f64[1,3], f64[1,3]), ... )
        >>> soln.ys
        (Array([[-6.91453518,  1.64014782,  0. ]], dtype=float64),
         Array([[-0.24701038, -0.20172576,  0. ]], dtype=float64))

        - `coordinax.vecs.AbstractPos3D`, `coordinax.vecs.AbstractVel3D`:

        >>> q0 = cx.CartesianPos3D.from_(xyz0)
        >>> p0 = cx.CartesianVel3D.from_(v_xyz)
        >>> w0 = (q0, p0)

        >>> soln = solver.solve(field, w0, t0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `coordinax.vecs.FourVector`, `coordinax.vecs.AbstractVel3D`:

        >>> w0 = (cx.vecs.FourVector(q=q0, t=t0), p0)

        >>> soln = solver.solve(field, w0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `coordinax.KinematicSpace`:

        >>> w0 = cx.KinematicSpace(length=q0, speed=p0)

        >>> soln = solver.solve(field, w0, t0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        >>> w0 = cx.KinematicSpace(length=cx.vecs.FourVector(q=q0, t=t0), speed=p0)

        >>> soln = solver.solve(field, w0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `coordinax.frames.AbstractCoordinate`:

        >>> w0 = cx.Coordinate({"length": q0, "speed": p0}, gc.frames.simulation_frame)

        >>> soln = solver.solve(field, w0, t0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `galax.coordinates.PhaseSpacePosition` (without time):

        >>> w0 = gc.PhaseSpacePosition(q=q0, p=p0)

        >>> soln = solver.solve(field, w0, t0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `galax.coordinates.PhaseSpaceCoordinate`: we've seen this before!

        >>> w0 = gc.PhaseSpaceCoordinate(q=q0, p=p0, t=t0)

        >>> soln = solver.solve(field, w0, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        - `galax.coordinates.AbstractCompositePhaseSpaceCoordinate`:

        >>> w01 = gc.PhaseSpaceCoordinate(q=u.Quantity([10, 0, 0], "kpc"),
        ...                               p=u.Quantity([0, 200, 0], "km/s"),
        ...                               t=u.Quantity(0, "Gyr"))
        >>> w02 = gc.PhaseSpaceCoordinate(q=u.Quantity([0, 10, 0], "kpc"),
        ...                               p=u.Quantity([-200, 0, 0], "km/s"),
        ...                               t=u.Quantity(10, "Myr"))
        >>> w0s = gc.CompositePhaseSpaceCoordinate(w01=w01, w02=w02)

        >>> soln = solver.solve(field, w0s, t1, unbatch_time=True)
        >>> soln
        {'w01': Solution( t0=f64[], t1=f64[], ts=f64[],
                        ys=(f64[3], f64[3]), ... ),
        'w02': Solution( t0=f64[], t1=f64[], ts=f64[],
                        ys=(f64[3], f64[3]), ... )}

        - `galax.dynamics.solve.SolveState` from ``.init()``:

        >>> state = solver.init(field, w0, u.Quantity(0, "Gyr"), None)
        >>> soln = solver.solve(field, state, t1, unbatch_time=True)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[],
                  ys=(f64[3], f64[3]), ...)

        ## Solving at times

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

        ## Batched solutions

        A set of initial conditions can be solved at once. The resulting
        `diffrax.Solution` has a `ys` shape of ([time], *shape, 3),

        >>> w0s = (u.Quantity([[8, 0, 0], [9, 0, 0]], "kpc"),
        ...        u.Quantity([[0, 220, 0], [0, 230, 0]], "km/s"))

        >>> soln = solver.solve(field, w0s, t0, t1)
        >>> soln
        Solution( t0=f64[], t1=f64[], ts=f64[1],
                  ys=(f64[1,2,3], f64[1,2,3]), ... )
        >>> soln.ys
        (Array([[[-6.91453717,  1.64014587,  0.        ],
                 [-0.09424643, -8.71835893,  0.        ]]], dtype=float64),
         Array([[[-0.24701013, -0.20172583,  0.        ],
                 [ 0.24385017,  0.09506092,  0.        ]]], dtype=float64))

        This can be batched with a set of times that can be broadcasted to the
        shape of ``w0s``.

        >>> t1 = u.Quantity([1, 1.1], "Gyr")
        >>> soln = solver.solve(field, w0, t0, t1)
        >>> soln
        Solution( t0=f64[2], t1=f64[2], ts=f64[2,1],
                  ys=(f64[2,1,3], f64[2,1,3]), ... )
        >>> soln.ys
        (Array([[[-6.91453518,  1.64014782,  0.        ]],
           [[ 2.47763667, -6.45713646,  0.        ]]], dtype=float64),
         Array([[[-0.24701038, -0.20172576,  0.        ]],
           [[ 0.31968144, -0.10665539,  0.        ]]], dtype=float64))

        All these examples can be interpolated as dense solutions. If the time
        or position are batched, then the `diffrax.DenseInterpolation` needs to
        be vectorized into a
        `galax.dynamics.integrate.VectorizedDenseInterpolation`. This can be
        done by passing ``vectorize_interpolation=True``. To emphasize the
        differences, let's do this on a batched solve.

        >>> t0 = u.Quantity([-1, -0.5], "Gyr")
        >>> soln = solver.solve(field, w0s, t0, t1, dense=True,
        ...                     vectorize_interpolation=True)
        >>> newq, newp = soln.evaluate(0.5)  # Myr
        >>> print((newq.round(3), newp.round(3)))
        (Array([[-7.034,  1.538,  0.   ],
                [ 3.428, -0.286,  0.   ]], dtype=float64),
         Array([[-0.232, -0.205,  0.   ],
                [ 0.366,  0.587,  0.   ]], dtype=float64))

        """
        raise NotImplementedError  # pragma: no cover


# ===============================================
# Init Dispatches


@OrbitSolver.init.dispatch
def init(
    self: OrbitSolver,
    field: AbstractOrbitField,
    qp: Any,
    t0: gt.BBtQuSz0 | gt.BBtLikeSz0,
    args: gt.OptArgs = None,
    /,
) -> SolveState:
    """Initialize from terms, unit/array tuple, and time."""
    # Parse the inputs to jax arrays
    t0, y0 = parse_to_t_y(None, t0, qp, ustrip=field.units)  # TODO: frame
    # Initialize the state
    return self._init_impl(field, t0, y0, args, field.units)


@OrbitSolver.init.dispatch
def init(
    self: OrbitSolver,
    field: AbstractOrbitField,
    tqp: Any,
    args: gt.OptArgs = None,
    /,
) -> SolveState:
    t0, y0 = parse_to_t_y(None, tqp, ustrip=field.units)  # TODO: frame
    return self.init(field, y0, t0, args)


# Composite PSPs
@OrbitSolver.init.dispatch
def init(
    self: OrbitSolver,
    field: AbstractOrbitField,
    w0s: gc.AbstractCompositePhaseSpaceCoordinate,
    args: gt.OptArgs = None,
    /,
) -> dict[str, SolveState]:
    return {k: self.init(field, w0, args) for k, w0 in w0s.items()}


# ===============================================
# Run Dispatches


@OrbitSolver.run.dispatch
def run(
    self: OrbitSolver,
    field: AbstractOrbitField,
    state: SolveState,
    t1: Any,
    args: PyTree,
    /,
    **solver_kw: Any,
) -> SolveState:
    t1 = u.ustrip(AllowValue, state.units["time"], t1)  # Parse the time
    # Validate the solver keyword arguments
    solver_kw = eqx.error_if(
        solver_kw, "saveat" in solver_kw, "`saveat` is not allowed in run"
    )
    solver_kw.setdefault("dt0", None)

    # Run the solver
    # Broadcast state.t, t1, and the PyTree state.y to have the same batch
    # dimensions, where t, t1 are scalars and state.y can have additional
    # shape dimensions.
    t0_bbt, t1_bbt = jnp.broadcast_arrays(state.t, t1)
    ndim0, batch = state.t.ndim, t0_bbt.shape
    y0 = jtu.map(lambda x: jnp.broadcast_to(x, batch + x.shape[ndim0:]), state.y)
    t0_f, t1_f = t0_bbt.reshape(-1), t1_bbt.reshape(-1)
    y0_f = jtu.map(lambda x: x.reshape(-1, *x.shape[ndim0:]), y0)

    # vmap over the solver
    @ft.partial(jax.vmap, in_axes=(0, 0, jtu.map(lambda _: 0, y0)))
    def runner(t0: gt.Sz0, t1: gt.Sz0, y0: PyTree, /) -> dfx.Solution:
        return self(field, t0=t0, t1=t1, y0=y0, args=args, **solver_kw)

    soln = runner(t0_f, t1_f, y0_f)

    # Unbatch the time batch shape
    if soln.ts.shape[soln.t0.ndim] == 1:
        slc = (slice(None),) * soln.t0.ndim + (0,)
        soln = eqx.tree_at(lambda tree: tree.ts, soln, soln.ts[slc])
        soln = eqx.tree_at(
            lambda tree: tree.ys, soln, jtu.map(lambda y: y[slc], soln.ys)
        )
        # TODO: unbatch the solver_state

    # Reshape the solution (possibly unbatching)
    soln = jtu.map(lambda x: x.reshape(batch + x.shape[1:]), soln)

    # Unpack into Solution into a SolveState
    return SolveState(
        t=soln.ts,
        y=soln.ys,
        solver_state=soln.solver_state,
        success=soln.result,
        units=state.units,
    )


@OrbitSolver.run.dispatch
def run(
    self: OrbitSolver,
    field: AbstractOrbitField,
    state: dict[str, SolveState],
    t1: Any,
    args: PyTree,
    /,
) -> dict[str, SolveState]:
    return {k: self.run(field, v, t1, args) for k, v in state.items()}


# ===============================================
# Solve Dispatches

# -------------------------------------
# Scalar solve - JAX arrays
# TODO: check if this is any faster than the `init-run` pattern.


@OrbitSolver.solve.dispatch(precedence=1)  # type: ignore[misc]
@ft.partial(eqx.filter_jit)
def solve(
    self: OrbitSolver,
    field: AbstractOrbitField,
    qp0: tuple[gdt.BBtQarr, gdt.BBtParr],
    t0: gt.LikeSz0,
    t1: gt.LikeSz0,
    /,
    *,
    args: Any = None,
    saveat: Any = default_saveat,
    unbatch_time: bool = False,
    **solver_kw: Any,
) -> dfx.Solution:
    """Solve for batch position tuple, scalar start, end time."""
    # Parse inputs
    usys = field.units

    # Run the solver
    solver_kw["saveat"] = parse_saveat(usys, saveat, dense=solver_kw.pop("dense", None))
    solver_kw.setdefault("dt0", None)
    soln = self(field, t0=t0, t1=t1, y0=qp0, args=args, **solver_kw)

    # Possibly unbatch in the time dimension.
    if unbatch_time and soln.ts.shape[soln.t0.ndim] == 1:
        soln = eqx.tree_at(lambda tree: tree.ts, soln, soln.ts[0])
        soln = eqx.tree_at(lambda tree: tree.ys, soln, jtu.map(lambda y: y[0], soln.ys))

    return soln


scalar_solver = OrbitSolver.solve.invoke(
    OrbitSolver,
    AbstractOrbitField,
    tuple[gdt.BBtQarr, gdt.BBtParr],
    gt.LikeSz0,
    gt.LikeSz0,
)

# ---------------------------


# NOTE: The scalar solve doesn't depend on the loop strategy.
@OrbitSolver.solve.dispatch(precedence=1)
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.AbstractLoopStrategy],  # noqa: ARG001
    field: AbstractOrbitField,
    qp0: tuple[gdt.BBtQarr, gdt.BBtParr],
    t0: gt.LikeSz0,
    t1: gt.LikeSz0,
    /,
    **kw: Any,
) -> dfx.Solution:
    """Solve for batch position tuple, scalar start, end time."""
    return scalar_solver(self, field, qp0, t0, t1, **kw)


# -------------------------------------
# Batched solve - JAX arrays


@OrbitSolver.solve.dispatch
def solve(
    self: OrbitSolver,
    field: AbstractOrbitField,
    qp: tuple[gdt.BBtQarr, gdt.BBtParr],
    t0: gt.BBtLikeSz0,
    t1: gt.BBtLikeSz0,
    /,
    **kw: Any,
) -> dfx.Solution:
    """Solve for batch position tuple, batched start, end time."""
    return self.solve(lstrat.Determine, field, qp, t0, t1, **kw)


# ---------------------------


@OrbitSolver.solve.dispatch
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.Determine],  # noqa: ARG001
    field: AbstractOrbitField,
    qp: tuple[gdt.BBtQarr, gdt.BBtParr],
    t0: gt.BBtLikeSz0,
    t1: gt.BBtLikeSz0,
    /,
    **kw: Any,
) -> dfx.Solution:
    """Solve for batch position tuple, batched start, end time."""
    # Determine the loop strategy
    platform = jex.backend.get_backend().platform
    loop_strat = lstrat.Scan if platform == "cpu" else lstrat.Vectorize

    # Call solver with appropriate loop strategy
    return self.solve(loop_strat, field, qp, t0, t1, **kw)


# ---------------------------


def _is_saveat_arr(saveat: Any, /) -> bool:
    dfx_check = (
        isinstance(saveat, dfx.SaveAt)
        and isinstance(saveat.subs, dfx.SubSaveAt)
        and (saveat.subs.ts is not None and saveat.subs.ts.ndim in (0, 1))
    )
    arr_check = hasattr(saveat, "ndim") and saveat.ndim in (0, 1)
    return dfx_check or arr_check


@OrbitSolver.solve.dispatch
@ft.partial(eqx.filter_jit)
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.Vectorize],  # noqa: ARG001
    field: AbstractOrbitField,
    qp: tuple[gdt.BBtQarr, gdt.BBtParr],
    t0: gt.BBtLikeSz0,
    t1: gt.BBtLikeSz0,
    /,
    saveat: Any = default_saveat,
    **kw: Any,
) -> dfx.Solution:
    # Parse kwargs
    # Because of how JAX does its tree building over vmaps, vectorizing the
    # interpolation within the loop does not work, it needs to be done after.
    vectorize_interpolation = kw.pop("vectorize_interpolation", False)
    kw.setdefault("dt0", None)

    # Build the vectorized solver. To get the right broadcasting the Q, P need
    # to be split up, then recombined.
    @ft.partial(jnp.vectorize, signature="(3),(3),(),()->()")
    def batched_call(q: gdt.Qarr, p: gdt.Parr, t0: gt.Sz0, t1: gt.Sz0) -> dfx.Solution:
        return scalar_solver(self, field, (q, p), t0, t1, saveat=saveat, **kw)

    # Solve the batched problem
    soln = batched_call(qp[0], qp[1], t0, t1)

    # NOTE: this is a heuristic that can be improved!
    # The saveat was not vmapped over, so it got erroneously broadcasted.
    # Let's try to unbatch it if it looks like it was broadcasted.
    if _is_saveat_arr(saveat):
        slc = (0,) * soln.t0.ndim
        soln = eqx.tree_at(lambda tree: tree.ts, soln, soln.ts[slc])

    # Now possibly vectorize the interpolation
    if vectorize_interpolation:
        soln = dfxtra.VectorizedDenseInterpolation.apply_to_solution(soln)

    return soln


# ---------------------------


@OrbitSolver.solve.dispatch
@ft.partial(eqx.filter_jit)
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.VMap],  # noqa: ARG001
    field: AbstractOrbitField,
    qp0: tuple[gdt.BBtQarr, gdt.BBtParr],
    t0: gt.BBtLikeSz0,
    t1: gt.BBtLikeSz0,
    /,
    saveat: Any = default_saveat,
    **kw: Any,
) -> dfx.Solution:
    # Parse kwargs
    # Because of how JAX does its tree building over vmaps, vectorizing the
    # interpolation within the loop does not work, it needs to be done after.
    vectorize_interpolation = kw.pop("vectorize_interpolation", False)
    kw.setdefault("dt0", None)

    # Broadcast t0, t1, qp
    q0, p0 = qp0
    batch = jnp.broadcast_shapes(t0.shape, t1.shape, q0.shape[:-1], p0.shape[:-1])
    t0_b, t1_b = jnp.broadcast_to(t0, batch), jnp.broadcast_to(t1, batch)
    q0_b, p0_b = (jnp.broadcast_to(q0, (*batch, 3)), jnp.broadcast_to(p0, (*batch, 3)))

    # flatten the batch dimensions
    t0_f, t1_f = t0_b.reshape(-1), t1_b.reshape(-1)
    q0_f, p0_f = q0_b.reshape(-1, 3), p0_b.reshape(-1, 3)

    # Build the vectorized solver.
    @ft.partial(jax.vmap, in_axes=(0, 0, 0, 0))
    def batched_call(q: gdt.Qarr, p: gdt.Parr, t0: gt.Sz0, t1: gt.Sz0) -> dfx.Solution:
        return scalar_solver(self, field, (q, p), t0, t1, saveat=saveat, **kw)

    # Solve the batched problem
    soln = batched_call(q0_f, p0_f, t0_f, t1_f)

    # Unflatten the batch dimensions
    soln = jtu.map(lambda x: x.reshape(batch + x.shape[1:]), soln)

    # Possibly unbatch the time dimension
    if _is_saveat_arr(saveat):
        slc = (0,) * soln.t0.ndim
        soln = eqx.tree_at(lambda tree: tree.ts, soln, soln.ts[slc])

    # Now possibly vectorize the interpolation
    if vectorize_interpolation:
        soln = dfxtra.VectorizedDenseInterpolation.apply_to_solution(soln)

    return soln


# ---------------------------


@OrbitSolver.solve.dispatch
@ft.partial(eqx.filter_jit)
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.Scan],  # noqa: ARG001
    field: AbstractOrbitField,
    qp0: tuple[gdt.BBtQarr, gdt.BBtParr],
    t0: gt.BBtLikeSz0,
    t1: gt.BBtLikeSz0,
    /,
    saveat: Any = default_saveat,
    **kw: Any,
) -> dfx.Solution:
    # Parse kwargs
    # Because of how JAX does its tree building over vmaps, vectorizing the
    # interpolation within the loop does not work, it needs to be done after.
    vectorize_interpolation = kw.pop("vectorize_interpolation", False)
    kw.setdefault("dt0", None)

    # Broadcast t0, t1, qp
    q0, p0 = qp0
    batch = jnp.broadcast_shapes(t0.shape, t1.shape, q0.shape[:-1], p0.shape[:-1])
    t0_b, t1_b = jnp.broadcast_to(t0, batch), jnp.broadcast_to(t1, batch)
    q0_b, p0_b = (jnp.broadcast_to(q0, (*batch, 3)), jnp.broadcast_to(p0, (*batch, 3)))

    # flatten the batch dimensions
    t0_f, t1_f = t0_b.reshape(-1), t1_b.reshape(-1)
    q0_f, p0_f = q0_b.reshape(-1, 3), p0_b.reshape(-1, 3)

    # Build the body.
    @jax.jit
    def body(carry: tuple[int], _: int) -> tuple[tuple[int], dfx.Solution]:
        i = carry[0]
        w0_i = (q0_f[i], p0_f[i])
        soln = scalar_solver(self, field, w0_i, t0_f[i], t1_f[i], saveat=saveat, **kw)
        return [i + 1], soln

    # Solve the batched problem
    _, soln = jax.lax.scan(body, [0], jnp.arange(len(t0_f)))

    # Unflatten the batch dimensions
    soln = jtu.map(lambda x: x.reshape(batch + x.shape[1:]), soln)

    # Possibly unbatch the time dimension
    if _is_saveat_arr(saveat):
        slc = (0,) * soln.t0.ndim
        soln = eqx.tree_at(lambda tree: tree.ts, soln, soln.ts[slc])

    # Now possibly vectorize the interpolation
    if vectorize_interpolation:
        soln = dfxtra.VectorizedDenseInterpolation.apply_to_solution(soln)

    return soln


# -------------------------------------
# Generic pass through on missing loop strategy


@OrbitSolver.solve.dispatch(precedence=-1)
def solve(
    self: OrbitSolver, field: AbstractOrbitField | Terms, *args: Any, **kw: Any
) -> Any:
    """Solve generically, determining loop strategy."""
    return self.solve(lstrat.Determine, field, *args, **kw)


# -------------------------------------
# From State


@OrbitSolver.solve.dispatch
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.AbstractLoopStrategy],
    field: AbstractOrbitField | Terms,
    state: SolveState,
    t1: gt.BBtQuSz0,
    /,
    **kw: Any,
) -> dfx.Solution:
    """Solve for state, end time."""
    kw["solver_state"] = state.solver_state
    return self.solve(loop_strategy, field, state.y, state.t, t1, **kw)


# -------------------------------------


@OrbitSolver.solve.dispatch
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.AbstractLoopStrategy],
    field: AbstractOrbitField | Terms,
    tqp0: Any,
    t1: gt.BBtQuSz0 | gt.BBtLikeSz0,
    **kw: Any,
) -> dfx.Solution:
    # Parse the inputs
    t0, qp0 = parse_to_t_y(None, tqp0, ustrip=field.units)  # TODO: frame
    t1 = u.ustrip(AllowValue, field.units["time"], t1)
    return self.solve(loop_strategy, field, qp0, t0, t1, **kw)


@OrbitSolver.solve.dispatch
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.AbstractLoopStrategy],
    field: AbstractOrbitField | Terms,
    qp0: Any,
    t0: gt.BBtQuSz0 | gt.BBtLikeSz0,
    t1: gt.BBtQuSz0 | gt.BBtLikeSz0,
    **kw: Any,
) -> dfx.Solution:
    # Parse the inputs
    usys = getattr(field, "units", None)
    t0, qp0 = parse_to_t_y(None, t0, qp0, ustrip=usys)  # TODO: frame
    t1 = u.ustrip(AllowValue, usys["time"], jnp.asarray(t1))
    return self.solve(loop_strategy, field, qp0, t0, t1, **kw)


# -------------------------------------


@OrbitSolver.solve.dispatch
def solve(
    self: OrbitSolver,
    loop_strategy: type[lstrat.AbstractLoopStrategy],
    field: AbstractOrbitField | Terms,
    w0s: gc.AbstractCompositePhaseSpaceCoordinate,
    t1: gt.BBtQuSz0,
    /,
    args: Any = None,
    **solver_kw: Any,
) -> dict[str, dfx.Solution]:
    """Solve for AbstractCompositePhaseSpaceCoordinate, start, end time."""
    return {
        k: self.solve(loop_strategy, field, w0, t1, args=args, **solver_kw)
        for k, w0 in w0s.items()
    }
