"""Experimental dynamics."""

__all__: list[str] = []

from functools import partial
from typing import Any, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.extend as jex
from jaxtyping import Array, Real
from plum import dispatch

import diffraxtra as dfxtra
import quaxed.numpy as jnp

import galax._custom_types as gt
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
import galax.utils.loop_strategies as lstrat
from galax.dynamics._src.orbit.field_base import AbstractOrbitField
from galax.dynamics._src.orbit.field_hamiltonian import HamiltonianField

BQParr: TypeAlias = tuple[Real[gdt.Qarr, "B"], Real[gdt.Parr, "B"]]

default_solver = dfxtra.DiffEqSolver(
    solver=dfx.Dopri8(scan_kind="bounded"),
    stepsize_controller=dfx.PIDController(
        rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, force_dtmin=True, jump_ts=None
    ),
    max_steps=10_000,
    # adjoint=ForwardMode(),  # noqa: ERA001
)


def parse_t0_t1_saveat(
    t0: gt.LikeSz0 | None,
    t1: gt.LikeSz0 | None,
    saveat: gt.LikeSz0 | None,
    *,
    dense: bool,
) -> tuple[gt.LikeSz0, gt.LikeSz0, dfx.SaveAt]:
    """Parse t0, t1, and saveat."""
    # Parse times / saveat TODO: see if this can be simplified with a PR to
    # https://github.com/patrick-kidger/diffrax/blob/14baa1edddcacf27c0483962b3c9cf2e86e6e5b6/diffrax/_saveat.py#L18
    # There are 3 cases to consider:
    # 1. saveat is None, in which case we integrate from t0 to t1, returning the
    #    solution at t1. Neither t0 nor t1 can be None.
    # 2. saveat is a 0-D array (a scalar), in which case we integrate from t0 to
    #    saveat, skipping the need for t1. t0 cannot be None.
    # 3. saveat is a length>2 1-D array, in which case we integrate from t0 to
    #    t1, saving at the times specified in saveat. t0 and t1 can be None, in
    #    which case they are inferred from saveat.
    ts = jnp.array(saveat) if saveat is not None else None
    if ts is None:
        t0 = eqx.error_if(
            t0, t0 is None or t1 is None, "t0, t1 must be specified if saveat is None"
        )
        saver = dfx.SaveAt(
            t0=False, t1=True if not dense else None, ts=None, dense=dense, steps=False
        )

    elif ts.ndim == 0:
        t0 = eqx.error_if(t0, t0 is None, "t0 must be specified if saveat is a scalar")
        t1 = ts
        saver = dfx.SaveAt(
            t0=False,
            t1=False,
            ts=jnp.array([t0, ts]) if not dense else None,
            dense=dense,
            steps=False,
        )

    else:
        ts = eqx.error_if(
            ts,
            len(ts) < 2 and (t0 is None or t1 is None),
            "if t0 or t1 are None, saveat must be [t0, ..., t1]",
        )
        t0 = jnp.min(ts) if t0 is None else t0
        t1 = jnp.max(ts) if t1 is None else t1
        saver = dfx.SaveAt(
            t0=False, t1=False, ts=ts if not dense else None, dense=dense, steps=False
        )

    return t0, t1, saver


# =============================================================================


@dispatch.multi(
    (
        type[lstrat.AbstractLoopStrategy],
        gp.AbstractPotential | AbstractOrbitField,
        gdt.QParr,
        gt.LikeSz0 | None,
        gt.LikeSz0 | None,
    ),
    (
        type[lstrat.NoLoop],
        gp.AbstractPotential | AbstractOrbitField,
        BQParr,
        gt.LikeSz0 | None,
        gt.LikeSz0 | None,
    ),
)
@partial(
    jax.jit,
    static_argnums=(0,),
    static_argnames=("solver", "solver_kwargs", "dense"),
)
def integrate_orbit(
    loop_strategy: type[lstrat.AbstractLoopStrategy],  # noqa: ARG001
    pot: gp.AbstractPotential | AbstractOrbitField,
    qp0: gdt.QParr,
    /,
    t0: gt.LikeSz0 | None = None,
    t1: gt.LikeSz0 | None = None,
    *,
    saveat: gt.LikeSz0 | None = None,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
    dense: bool = False,
) -> dfx.Solution:
    """Integrate orbit associated with potential function.

    Parameters
    ----------
    loop_strategy:
        Loop strategy to use for the integration. Because of multiple dispatch,
        this argument can be omitted.
    pot:
        The potential in which to integrate the orbit.
    qp0:
        The initial conditions of the orbit. Should be a tuple of two arrays,
        the first containing the initial positions in Cartesian coordinates and
        the second containing the initial velocities in Cartesian coordinates.
    t0,t1,ts:
        Start time, end time, and save times. There are 3 cases:

        1. saveat is None, in which case we integrate from t0 to t1, returning
           the solution at t1. Neither t0 nor t1 can be None.
        2. saveat is a 0-D array (a scalar), in which case we integrate from t0
           to saveat, skipping the need for t1. t0 cannot be None.
        3. saveat is a length>2 1-D array, in which case we integrate from t0 to
           t1, saving at the times specified in saveat. t0 and t1 can be None,
           in which case they are inferred from saveat.

        In all cases, if `dense` is `True`, the solution is interpolated between
        `t0` and `t1` and no save times are used.

    solver:
        Solver to use for the integration. See `diffraxtra.AbstractDiffEqSolver`
        for more information.
    solver_kwargs:
        Additional keyword arguments to pass to the solver, e.g. 'max_steps',
        'stepsize_controller', etc. See `diffraxtra.AbstractDiffEqSolver` for
        more information.
    dense:
        When `False` (default), return orbit at times ts. When `True`, return
        dense interpolation of orbit between `t0` and `t1`.

    """
    field = pot if isinstance(pot, AbstractOrbitField) else HamiltonianField(pot)
    terms = field.terms(solver)

    t0, t1, saver = parse_t0_t1_saveat(t0, t1, saveat, dense=dense)

    soln: dfx.Solution = solver(
        terms, t0=t0, t1=t1, dt0=None, y0=qp0, saveat=saver, **(solver_kwargs or {})
    )
    return soln


# ---------------------------
# auto-determine


@dispatch(precedence=-1)
def integrate_orbit(
    pot: gp.AbstractPotential | AbstractOrbitField,
    qp0: Any,
    /,
    t0: gt.LikeSz0 | None = None,
    t1: gt.LikeSz0 | None = None,
    **kwargs: Any,
) -> dfx.Solution:
    """Integrate the orbit.

    This function re-dispatches to the correct integrator based on the
    determined loop strategy.

    """
    return integrate_orbit(lstrat.Determine, pot, qp0, t0, t1, **kwargs)


@dispatch(precedence=-1)
def integrate_orbit(
    loop_strategy: type[lstrat.Determine],  # noqa: ARG001
    pot: gp.AbstractPotential | AbstractOrbitField,
    qp0: BQParr,
    /,
    t0: gt.LikeSz0 | None = None,
    t1: gt.LikeSz0 | None = None,
    **kw: Any,
) -> dfx.Solution:
    """Re-dispatch based on the determined loop strategy."""
    # Determine the loop strategy
    platform = jex.backend.get_backend().platform
    loop_strat = lstrat.Scan if platform == "cpu" else lstrat.VMap

    # Call solver with appropriate loop strategy
    return integrate_orbit(loop_strat, pot, qp0, t0, t1, **kw)


# ---------------------------

ScanCarry: TypeAlias = list[int]


@dispatch
@partial(
    jax.jit,
    static_argnums=(0,),
    static_argnames=("solver", "solver_kwargs", "dense"),
)
def integrate_orbit(
    loop_strategy: type[lstrat.Scan],  # noqa: ARG001
    pot: gp.AbstractPotential | AbstractOrbitField,
    qp0: BQParr,
    /,
    t0: gt.LikeSz0 | None = None,
    t1: gt.LikeSz0 | None = None,
    *,
    saveat: Real[Array, "time"] | Real[Array, "B time"],
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
    dense: bool = False,
) -> dfx.Solution:
    """Integrate a batch of orbits using scan [best for CPU usage].

    Parameters
    ----------
    qp0:
        shape ((B,3), (B,3)) array of initial conditions
    ts:
        array of save times. Can either be 1D array (same for all trajectories),
        or (B x T) array, where T is the number of saved times for each
        trajectory.

    Returns
    -------
    `dfx.Solution`:
        The solution to the differential equation. The ``ys`` attribute contains
        the solution at the saved times ``ts`` and will have shape (B, T, 6).

    Notes
    -----
    `diffrax.diffeqsolve` can solve a batched `y0` array, but this function is
    actually faster, at the expense of only being able to solve for a single
    batch axis, but at the gain of being able to batch over the `saveat`. If you
    want to speed compare against raw `diffrax.diffeqsolve`, you can use the
    `galax.utils.loop_strategies.NoLoop` loop strategy.

    """

    @partial(jax.jit)
    def body(carry: ScanCarry, _: float) -> tuple[ScanCarry, dfx.Solution]:
        i = carry[0]
        saveat_i = saveat if len(saveat.shape) == 1 else saveat[i]
        soln = integrate_orbit(
            pot,
            (qp0[0][i], qp0[1][i]),
            t0,
            t1,
            saveat=saveat_i,
            dense=dense,
            solver=solver,
            solver_kwargs=solver_kwargs,
        )
        return [i + 1], soln

    init_carry = [0]
    _, state = jax.lax.scan(body, init_carry, jnp.arange(len(qp0)))
    soln: dfx.Solution = state
    return soln


@dispatch
@partial(
    jax.jit,
    static_argnums=(0,),
    static_argnames=("solver", "solver_kwargs", "dense"),
)
def integrate_orbit(
    loop_strategy: type[lstrat.VMap],  # noqa: ARG001
    pot: gp.AbstractPotential | AbstractOrbitField,
    qp0: BQParr,
    /,
    t0: gt.LikeSz0 | None = None,
    t1: gt.LikeSz0 | None = None,
    *,
    saveat: Real[Array, "time"] | Real[Array, "B time"],
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
    dense: bool = False,
) -> dfx.Solution:
    """Integrate a batch of orbits using scan [best for GPU usage].

    Parameters
    ----------
    qp0:
        shape ((B,3), (B,3)) array of initial conditions
    ts:
        array of save times. Can either be 1D array (same for all trajectories),
        or (B x T) array, where T is the number of saved times for each
        trajectory.

    Returns
    -------
    `dfx.Solution`:
        The solution to the differential equation. The ``ys`` attribute contains
        the solution at the saved times ``ts`` and will have shape (B, T, 6).

    Notes
    -----
    `diffrax.diffeqsolve` can solve a batched `y0` array, but this function is
    actually faster, at the expense of only being able to solve for a single
    batch axis, but at the gain of being able to batch over the `saveat`. If you
    want to speed compare against raw `diffrax.diffeqsolve`, you can use the
    `galax.utils.loop_strategies.NoLoop` loop strategy.

    """
    integrator = lambda qp0, saveat: integrate_orbit(
        pot,
        qp0,
        t0,
        t1,
        saveat=saveat,
        dense=dense,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )
    in_axes = ((0, 0), (None if saveat.ndim == 1 else 0))
    integrator_mapped = jax.vmap(integrator, in_axes=in_axes)

    soln: dfx.Solution = integrator_mapped(qp0, saveat)
    return soln
