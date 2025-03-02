"""Fields."""

__all__ = ["AbstractField"]

from collections.abc import Callable
from dataclasses import KW_ONLY
from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real
from plum import dispatch

import diffraxtra as dfxtra
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt


class AbstractField(eqx.Module, strict=True):  # type: ignore[misc,call-arg]
    """Abstract base class for fields.

    Methods
    -------
    - `__call__` : evaluates the field.
    - `terms` : returns the `diffrax.AbstractTerm` wrapper of the `jax.jit`-ed
      ``__call__`` for integration with `diffrax.diffeqsolve`. `terms` takes as
      input the `diffrax.AbstractSolver` object (or something that wraps it,
      like a `diffraxtra.DiffEqSolver`), to determine the correct
      `diffrax.AbstractTerm` and its `jaxtyping.PyTree` structure.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    Define a Hamiltonian field:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Evaluate the field at a given coordinate:

    >>> t = u.Quantity(0, "Gyr")
    >>> x = u.Quantity([8., 0, 0], "kpc")
    >>> v = u.Quantity([0, 220, 0], "km/s")

    >>> field(t, x, v)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    This can also be done with `jax.Array` directly, but care must be taken to
    ensure the units are correct. In this case ``x`` is in the right units, but
    ``t``, ``v`` are not. We use `unxt.ustrip` to correctly convert and remove
    the units:

    >>> field(t.ustrip("Myr"), x.value, v.ustrip("kpc/Myr"))
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    Field evaluation is very flexible and can work with a large variety of
    inputs. For more information, see the
    `galax.dynamics.fields.HamiltonianField` class.

    For integration with `diffrax.diffeqsolve` the ``terms`` method returns the
    correctly-structured `diffrax.AbstractTerm` `jaxtyping.PyTree`. The term is
    a wrapper around the ``__call__`` method, which is `jax.jit`-ed for
    performance.

    >>> field.terms(dfx.Dopri8())
    ODETerm(vector_field=<wrapped function __call__>)

    >>> field.terms(dfx.SemiImplicitEuler())
    (ODETerm(...), ODETerm(...))

    """

    __call__: eqx.AbstractClassVar[Callable[..., Any]]

    _: KW_ONLY
    #: unit system of the field.
    units: eqx.AbstractVar[u.AbstractUnitSystem]

    @dispatch.abstract
    def terms(self, solver: Any, /) -> PyTree[dfx.AbstractTerm]:
        """Return the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration."""
        raise NotImplementedError  # pragma: no cover

    # TODO: consider the frame information, like in `parse_to_t_y`
    @dispatch.abstract
    def parse_inputs(self, *args: Any, **kwargs: Any) -> Any:
        """Parse inputs for the field.

        Dispatches to this method should at least support the following:

        - `parse_inputs(self, t: Array, y0: PyTree[Array], /, *, ustrip: bool) -> tuple[Array, PyTree[Array]]`
        - `parse_inputs(self, t: Quantity, y0: PyTree[Quantity], /, *, ustrip: bool) -> tuple[Array, PyTree[Array]]`

        Where the output types are suitable for use with `diffrax`.

        """  # noqa: E501
        raise NotImplementedError


# ==================================================================


@AbstractField.terms.dispatch  # type: ignore[misc]
def terms(
    self: AbstractField, wrapper: dfxtra.AbstractDiffEqSolver, /
) -> PyTree[dfx.AbstractTerm]:
    """Return diffeq terms, redispatching to the solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.solve.DiffEqSolver(dfx.Dopri8())

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    """
    return self.terms(wrapper.solver)


# ==================================================================


default_solver = dfxtra.DiffEqSolver(
    solver=dfx.Dopri8(scan_kind="bounded"),
    stepsize_controller=dfx.PIDController(
        rtol=1e-7, atol=1e-7, dtmin=0.05, dtmax=None, force_dtmin=True
    ),
    max_steps=16**3,
)


def _parse_t0_t1(
    ts: gt.SzTime, t0: gt.LikeSz0, t1: gt.LikeSz0, /, *, backwards_int: bool
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
        t0, t1 = jax.lax.cond(backwards_int, true_func, false_func, ts)
        return t0, t1

    def t0_t1_are_different() -> tuple[gt.Sz0, gt.Sz0]:
        return t0, t1

    t0, t1 = jax.lax.cond(t0 != t1, t0_t1_are_different, t0_t1_are_same)
    return t0, t1


@partial(jax.jit, static_argnames=("solver" "dense", "backwards_int"))
def integrate_field(
    y0: PyTree[dfx.AbstractTerm],
    ts: Real[Array, "time"],
    /,
    field: AbstractField,
    *,
    solver: dfxtra.AbstractDiffEqSolver | dfx.AbstractSolver = default_solver,
    dense: bool = False,
    args: PyTree[Any] = None,
    backwards_int: bool = False,
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
    backwards_int : bool
        If True, integrate backwards in time.

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

    >>> soln = gd.integrate_field(y0, ts, field=field)
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

    >>> soln = gd.integrate_field(y0, ts, field=field)
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

    >>> soln = gd.integrate_field(w0, ts, field=field)
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
    # Solve for the terms given the solver
    terms_ = field.terms(solver)
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
    t0, t1 = _parse_t0_t1(ts, t0, t1, backwards_int=backwards_int)

    # Call solver
    return solver_(
        terms_, t0=t0, t1=t1, dt0=None, y0=y0, args=args, saveat=saveat, **solver_kwargs
    )
