"""Experimental dynamics."""

__all__: list[str] = []

from functools import partial
from typing import Any, TypeAlias

import diffrax as dfx
import jax
from jaxtyping import Array, Real

import diffraxtra as dfxtra
import quaxed.numpy as jnp

import galax._custom_types as gt
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
from galax.dynamics._src.orbit.field_hamiltonian import HamiltonianField

NQParr: TypeAlias = tuple[Real[gdt.Qarr, "N"], Real[gdt.Parr, "N"]]

default_solver = dfxtra.DiffEqSolver(
    solver=dfx.Dopri8(scan_kind="bounded"),
    stepsize_controller=dfx.PIDController(
        rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, force_dtmin=True, jump_ts=None
    ),
    max_steps=10_000,
    # adjoint=ForwardMode(),  # noqa: ERA001
)


@partial(jax.jit, static_argnames=("solver", "solver_kwargs", "dense"))
def integrate_orbit(
    pot: gp.AbstractPotential,
    w0: gdt.QParr,
    /,
    ts: gt.LikeSz0,
    t0: gt.LikeSz0 | None = None,
    t1: gt.LikeSz0 | None = None,
    *,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
    dense: bool = False,
) -> dfx.Solution:
    """Integrate orbit associated with potential function.

    Parameters
    ----------
    w0:
        length 6 array [x,y,z,vx,vy,vz]
    ts:
        array of saved times. Must be at least length 2, specifying a minimum
        and maximum time. This does _not_ determine the timestep

    solver:
        Solver to use for the integration.
    solver_kwargs:
        Additional keyword arguments to pass to the solver, e.g. 'max_steps',
        'stepsize_controller', etc.
    dense:
        When `False`, return orbit at times ts. When `True`, return dense
        interpolation of orbit between ts.min() and ts.max().

    """
    terms = HamiltonianField(pot).terms(solver)

    saveat = dfx.SaveAt(
        t0=False, t1=False, ts=ts if not dense else None, dense=dense, steps=False
    )

    soln: dfx.Solution = solver(
        terms,
        t0=ts.min() if t0 is None else t0,
        t1=ts.max() if t1 is None else t1,
        dt0=None,
        y0=w0,
        saveat=saveat,
        **(solver_kwargs or {}),
    )
    return soln


@partial(jax.jit, static_argnames=("solver", "solver_kwargs", "dense"))
def integrate_orbit_batch_scan(
    pot: gp.AbstractPotential,
    w0: NQParr,
    ts: Real[Array, "batch time"],
    /,
    t0: gt.LikeSz0 | None = None,
    t1: gt.LikeSz0 | None = None,
    *,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
    dense: bool = False,
) -> dfx.Solution:
    """Integrate a batch of orbits using scan [best for CPU usage].

    w0: shape ((N,3), (N,3)) array of initial conditions ts: array of saved
    times. Can either be 1D array (same for all trajectories), or N x M array,
    where M is the number of saved times for each trajectory.

    """

    @partial(jax.jit)
    def body(carry: list[int], _: float) -> tuple[list[int], dfx.Solution]:
        i = carry[0]
        w0_i = (w0[0][i], w0[1][i])
        ts_i = ts if len(ts.shape) == 1 else ts[i]
        soln = integrate_orbit(
            pot,
            w0_i,
            ts_i,
            t0=t0,
            t1=t1,
            dense=dense,
            solver=solver,
            solver_kwargs=solver_kwargs,
        )
        return [i + 1], soln

    init_carry = [0]
    _, state = jax.lax.scan(body, init_carry, jnp.arange(len(w0)))
    soln: dfx.Solution = state
    return soln


@partial(jax.jit, static_argnames=("solver", "solver_kwargs", "dense"))
def integrate_orbit_batch_vmap(
    pot: gp.AbstractPotential,
    w0: NQParr,
    ts: Real[Array, "batch time"],
    /,
    t0: gt.LikeSz0 | None = None,
    t1: gt.LikeSz0 | None = None,
    *,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
    dense: bool = False,
) -> dfx.Solution:
    """Integrate a batch of orbits using vmap [best for GPU usage].

    w0: shape ((N,3), (N,3)) array of initial conditions ts: array of saved
    times. Can either be 1D array (same for all trajectories), or N x M array,
    where M is the number of saved times for each trajectory.

    """
    integrator = lambda w0, ts: integrate_orbit(
        pot,
        w0,
        ts,
        t0=t0,
        t1=t1,
        dense=dense,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )

    if len(ts.shape) == 1:
        func = lambda w0: integrator(w0, ts)  # type: ignore[no-untyped-call]
        integrator_mapped = jax.vmap(func, in_axes=((0, 0),))
    else:
        func = jax.vmap(integrator, in_axes=((0, 0), 0))
        integrator_mapped = lambda w0: func(w0, ts)  # type: ignore[call-arg, no-untyped-call]

    soln: dfx.Solution = integrator_mapped(w0)
    return soln
