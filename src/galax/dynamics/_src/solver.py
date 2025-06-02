"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractSolver", "SolveState", "integrate_field"]


import abc
import functools as ft
from dataclasses import fields
from typing import Any, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree, Real

import diffraxtra as dfxtra
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from .fields import AbstractField

USys: TypeAlias = u.AbstractUnitSystem
DenseInfo: TypeAlias = dict[str, PyTree[Array]]
Terms: TypeAlias = PyTree
DfxRealScalarLike: TypeAlias = Real[int | float | Array | np.ndarray[Any, Any], ""]

# =========================================================
# SolveState


class SolveState(eqx.Module, strict=True):  # type: ignore[misc, call-arg]
    """State of the solver.

    This is used as the return value for `galax.dynamics.AbstractSolver.init`
    and `galax.dynamics.AbstractSolver.step`. It is used as the argument to
    `galax.dynamics.AbstractSolver.step`, `galax.dynamics.AbstractSolver.run`,
    and can also be passed to `galax.dynamics.AbstractSolver.solve`.

    """

    #: Current time.
    t: gt.Sz0

    # ---- diffrax step outputs ----
    #: Current solution at `t`.
    y: PyTree

    # TODO: figure out how to extract this from `diffrax.Solution`
    # #: A local error estimate made during the step
    # err: PyTree | None  # noqa: ERA001

    # TODO: figure out how to extract this from `diffrax.Solution`
    # #: Save information. This is a dictionary of information that is passed to
    # # the solver's interpolation routine to calculate dense output. (Used with
    # # SaveAt(ts=...) or SaveAt(dense=...).)
    # save_info: DenseInfo  # noqa: ERA001
    #: The value of the solver state at t1.
    solver_state: Any

    #: Step success. An integer (corresponding to diffrax.RESULTS) indicating
    # whether the step happened successfully, or if (unusually) it failed for
    # some reason.
    success: dfx.RESULTS

    # --- reconstruction info ---
    units: USys = eqx.field(static=True)

    @classmethod
    def from_step_output(
        cls,
        t: Any,
        obj: tuple[PyTree, PyTree | None, DenseInfo, Any, dfx.RESULTS],
        units: USys,
        /,
    ) -> "SolveState":
        return cls(
            t=t,
            y=obj[0],
            # err=obj[1],  # noqa: ERA001
            # save_info=obj[2],  # noqa: ERA001
            solver_state=obj[3],
            success=obj[4],
            units=units,
        )


# =========================================================
# Abstract Solver


class AbstractSolver(dfxtra.AbstractDiffEqSolver, strict=True):  # type: ignore[call-arg,misc]
    """ABC for solvers.

    Notes
    -----
    The ``init``, ``step``, and ``solve`` methods are abstract and should be
    implemented by subclasses.

    """

    @ft.partial(jnp.vectorize, excluded=(0, 1, 3, 4, 5))
    @ft.partial(eqx.filter_jit)
    def _init_impl(
        self,
        field: AbstractField,
        t0: gt.SzAny,
        y0: PyTree,
        args: Any,
        units: USys,
        /,
    ) -> SolveState:
        """`init` helper."""
        terms = field.terms(self)
        # Initializes the state from diffrax. Steps from t0 to t0!
        solver_state = self.solver.init(terms, t0, t0, y0, args)
        # Step from t0 to t0, which is a no-op but sets the state
        step_output = self.solver.step(
            terms, t0, t0, y0, args=args, solver_state=solver_state, made_jump=False
        )
        return SolveState.from_step_output(t0, step_output, units)

    @abc.abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> SolveState:
        """Initialize the solver."""
        raise NotImplementedError

    # -----------------------

    def _step_impl_scalar(
        self,
        field: AbstractField | Terms,
        state: SolveState,
        t1: gt.Sz0,
        args: Any,
        step_kw: dict[str, Any],
    ) -> SolveState:
        terms = _parse_field(field, self)
        t0 = state.t
        t0 = eqx.error_if(t0, t0.ndim != 0, "t0 must be a scalar")
        step_output = self.solver.step(
            terms,
            t0,
            t1,
            state.y,
            args=args,
            solver_state=state.solver_state,
            **step_kw,
        )
        return SolveState.from_step_output(t1, step_output, state.units)

    @abc.abstractmethod
    def step(
        self,
        terms: Any,
        state: SolveState,
        t1: Any,
        /,
        args: PyTree,
        **step_kwargs: Any,  # e.g. solver_state, made_jump
    ) -> SolveState:
        """Step the solver."""
        raise NotImplementedError

    # ----------------

    @abc.abstractmethod
    def run(
        self, terms: Any, state: SolveState, t1: Any, args: Any, **solver_kw: Any
    ) -> SolveState:
        """Run the solver."""
        raise NotImplementedError  # pragma: no cover

    # ----------------

    @abc.abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Solve, initializing and stepping to the solution."""
        raise NotImplementedError


# =========================================================
# Constructors


@AbstractSolver.from_.dispatch  # type: ignore[misc]
def from_(cls: type[AbstractSolver], solver: dfxtra.DiffEqSolver) -> AbstractSolver:
    """Create a new solver from a `diffraxtra.DiffeqSolver`.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import diffraxtra as dfxtra
    >>> import galax.dynamics as gd

    >>> solver = dfxtra.DiffEqSolver(dfx.Dopri5())

    >>> new_solver = gd.OrbitSolver.from_(solver)
    >>> new_solver
    OrbitSolver(
      solver=Dopri5(),
      stepsize_controller=ConstantStepSize(),
      adjoint=RecursiveCheckpointAdjoint(),
      max_steps=4096
    )

    """
    return cls(**{f.name: getattr(solver, f.name) for f in fields(cls)})


def _parse_field(field: AbstractField | Terms, solver: AbstractSolver) -> Terms:
    return field.terms(solver) if isinstance(field, AbstractField) else field


# ==================================================================


default_solver = dfxtra.DiffEqSolver(
    solver=dfx.Dopri8(scan_kind="bounded"),
    stepsize_controller=dfx.PIDController(
        rtol=1e-7, atol=1e-7, dtmin=0.05, dtmax=None, force_dtmin=True
    ),
    max_steps=16**3,
)


def _parse_t0_t1(
    ts: gt.SzTime, t0: gt.LikeSz0, t1: gt.LikeSz0, /
) -> tuple[gt.Sz0, gt.Sz0]:
    # Parse t0, t1
    t0, t1 = jnp.asarray(t0), jnp.asarray(t1)

    # Handle t0, t1:
    # - t0 == t1, t0 and t1 are computed using `ts` array (default)
    # - t0 != t1, the user-specified values are utilized
    def false_func(ts: gt.SzTime) -> tuple[gt.Sz0, gt.Sz0]:
        """Integrating forward in time: t1 > t0."""
        return ts.min(), ts.max()

    def true_func(ts: gt.SzTime) -> tuple[gt.Sz0, gt.Sz0]:
        """Integrating backwards in time: t1 < t0."""
        return ts.max(), ts.min()

    def t0_t1_are_same() -> tuple[gt.Sz0, gt.Sz0]:
        backwards_int = ts[-1] < ts[0]
        t0, t1 = jax.lax.cond(backwards_int, true_func, false_func, ts)
        return t0, t1

    def t0_t1_are_different() -> tuple[gt.Sz0, gt.Sz0]:
        return t0, t1

    t0, t1 = jax.lax.cond(t0 != t1, t0_t1_are_different, t0_t1_are_same)
    return t0, t1


@ft.partial(eqx.filter_jit)
def integrate_field(
    field: AbstractField,
    y0: PyTree[Array],
    ts: Real[Array, "time"],
    /,
    *,
    solver: dfxtra.AbstractDiffEqSolver | dfx.AbstractSolver = default_solver,
    dense: bool = False,
    args: PyTree[Any] = None,
    t0: gt.RealScalarLike = 0.0,
    t1: gt.RealScalarLike = 0.0,
    **solver_kwargs: Any,
) -> dfx.Solution:
    """Integrate a trajectory on a field.

    If you want complete control over the integration use
    `diffraxtra.DiffEqSolver` directly, or the `diffrax.diffeqsolve` function
    which it intelligently wraps.

    Parameters
    ----------
    y0:
        PyTree initial conditions. Shape must be compatible with the input
        field.
    ts : Array[real, (n>2,)]
        Array of saved times. Must be at least length 2, specifying a minimum
        and maximum time. This does *not* determine the timestep
    field : `galax.dynamics.fields.AbstractField`
        Specifies the field that we are integrating on. This field must have a
        `terms` method that returns the correct PyTree of `diffrax.AbstractTerm`
        objects given the solver.
    solver : `diffraxtra.AbstractDiffEqSolver`
        Solver object. If not a `diffraxtra.AbstractDiffEqSolver`, e.g.
        `diffraxtra.DiffEqSolver` or `galax.dynamics.OrbitSolver`, it will be
        converted using `diffraxtra.DiffEqSolver.from_`.
    dense : bool
        When False (default), return orbit at times ts. When True, return dense
        interpolation of orbit between the min and max of ``ts``.
    args : PyTree[Any]
        Additional arguments to pass to the field.

    t0, t1 : RealScalarLike
        Start and end times for the integration. If t0 == t1, the times are
        computed from the ts array. If t0 != t1, the user-specified values are
        utilized.

    solver_kwargs : Any
        Additional keyword arguments to pass to the `solver`. See
        `diffraxtra.AbstractDiffEqSolver` for the complete information. This can
        be used to override (a partial list):

        - `saveat` : `diffrax.SaveAt` object
        - `event` : `diffrax.Event` | None
        - `max_steps` : int | None
        - `throw` : bool
        - `progress_meter` : `diffrax.AbstractProgressMeter`
        - `solver_state` : PyTree[Array] | None
        - `controller_state` : PyTree[Array] | None
        - `made_jump` : BoolSz0Like | None

        - `vectorize_interpolation` : bool
            A flag to vectorize the interpolation using
            `diffraxtra.VectorizedDenseInterpolation`.

    See Also
    --------
    - `galax.dynamics.orbit.OrbitSolver` : Orbit solver supporting many more
      input types.
    - `galax.dynamics.cluster.MassSolver` : Cluster solver supporting many
        more input types.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    - From `jax.Array`:

    >>> y0 = (jnp.array([8., 0, 0]), jnp.array([0, 0.22499668, 0]))  # [kpc, kpc/Myr]
    >>> ts = jnp.linspace(0, 1_000, 100)  # [Myr]

    >>> soln = gd.integrate_field(field, y0, ts)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[100],
              ys=(f64[100,3], f64[100,3]),
              interpolation=None, ... )
    >>> soln.ys
    (Array([[ 8.        ,  0.        ,  0.        ],
            [ 3.73351942,  1.73655426,  0.        ],
            ...
            [ 7.99298322, -0.09508821,  0.        ],
            [ 4.15712296,  1.73084745,  0.        ]], dtype=float64),
     Array([[ 0.        ,  0.22499668,  0.        ],
            [-1.05400822, -0.00813413,  0.        ],
            [ 0.39312592,  0.19388413,  0.        ],
            ...
            [ 0.02973396,  0.22484028,  0.        ],
            [-0.96062032,  0.03302397,  0.        ]], dtype=float64))

    - From `unxt.Quantity`:

    >>> y0 = (u.Quantity([8., 0, 0], "kpc"), u.Quantity([0, 220, 0], "km/s"))
    >>> ts = u.Quantity(jnp.linspace(0, 1, 100), "Gyr")

    >>> soln = gd.integrate_field(field, y0, ts)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[100],
              ys=(f64[100,3], f64[100,3]),
              interpolation=None, ... )
    >>> soln.ys
    (Array([[ 8.        ,  0.        ,  0.        ],
            [ 3.73351941,  1.73655423,  0.        ],
            ...
            [ 7.9929833 , -0.09508739,  0.        ],
            [ 4.15711932,  1.73084753,  0.        ]], dtype=float64),
     Array([[ 0.        ,  0.22499668,  0.        ],
            [-1.05400822, -0.00813413,  0.        ],
            ...
            [ 0.0297337 ,  0.22484028,  0.        ],
            [-0.96062109,  0.03302365,  0.        ]], dtype=float64))

    Additional input types are dependent on the `field`.
    `galax.dynamics.orbit.AbstractOrbitField` can handle many more input types.
    See that class and `galax.dynamics.orbit.OrbitSolver` for more information.
    As a single example, here is how to use a
    `galax.coordinates.PhaseSpaceCoordinate`:

    >>> import galax.coordinates as gc

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([8, 0, 0], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"), t=u.Quantity(0, "Gyr"))

    >>> soln = gd.integrate_field(field, w0, ts)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[100],
              ys=(f64[100,3], f64[100,3]),
              interpolation=None, ... )

    >>> soln.ys
    (Array([[ 8.        ,  0.        ,  0.        ],
            [ 3.73351941,  1.73655423,  0.        ],
            ...
            [ 7.9929833 , -0.09508739,  0.        ],
            [ 4.15711932,  1.73084753,  0.        ]], dtype=float64),
     Array([[ 0.        ,  0.22499668,  0.        ],
            [-1.05400822, -0.00813413,  0.        ],
            ...
            [ 0.0297337 ,  0.22484028,  0.        ],
            [-0.96062109,  0.03302365,  0.        ]], dtype=float64))

    """
    # -----------------------
    # Parse inputs

    # Create the solver instance
    solver_: dfxtra.AbstractDiffEqSolver = (
        dfxtra.DiffEqSolver.from_(solver)
        if not isinstance(solver, dfxtra.AbstractDiffEqSolver)
        else solver
    )
    # Parse t0, y0. Important for Quantities
    _, y0 = field.parse_inputs(ts[0], y0, ustrip=True)  # Parse inputs

    # Make SaveAt
    u_t = field.units["time"]
    ts = u.ustrip(AllowValue, u_t, ts)  # Convert ts
    saveat = dfx.SaveAt(t0=False, t1=False, ts=ts, dense=dense)  # Make saveat

    # Start and end times for the integration
    # - t0 == t1, t0 and t1 are computed using `ts` array (default)
    # - t0 != t1, the user-specified values are utilized
    t0 = u.ustrip(AllowValue, u_t, t0)
    t1 = u.ustrip(AllowValue, u_t, t1)
    t0, t1 = _parse_t0_t1(ts, t0, t1)

    # Call solver
    return solver_(
        field, t0=t0, t1=t1, dt0=None, y0=y0, args=args, saveat=saveat, **solver_kwargs
    )
