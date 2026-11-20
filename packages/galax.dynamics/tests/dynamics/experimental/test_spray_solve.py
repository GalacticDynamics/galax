"""Tests for the experimental spray solver settings and progenitor orbit."""

from jaxtyping import Array

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diffraxtra as dfxtra
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from galax.dynamics._src.experimental.schedule import Uniform
from galax.dynamics._src.experimental.solve import (
    default_solver,
    host_terms,
    is_lattice,
    progenitor_states,
    solve_on,
)

POT = gp.MilkyWayPotential(units="galactic")
W0 = jnp.asarray([8.0, 0.0, 1.0, 0.0, 0.2, 0.05])  # kpc, kpc/Myr
SCHED = Uniform(jnp.asarray(1000.0), 5, 20, jnp.asarray(0.0))
ADAPTIVE = dfxtra.DiffEqSolver(
    dfx.Dopri8(),
    stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-10),
    adjoint=dfx.RecursiveCheckpointAdjoint(),
    max_steps=10_000,
)


def test_default_solver_is_lattice() -> None:
    assert is_lattice(default_solver())
    assert not is_lattice(ADAPTIVE)


def test_progenitor_returns_today_leapfrog() -> None:
    """The backward then forward solve is reversible, ending at ``w_today``."""
    ws = progenitor_states(POT, W0, SCHED, default_solver())
    assert ws.shape == (SCHED.n_steps() + 1, 6)
    np.testing.assert_allclose(ws[-1], W0, rtol=0, atol=1e-12)


def test_progenitor_adaptive_matches_evaluate_orbit() -> None:
    ws = progenitor_states(POT, W0, SCHED, ADAPTIVE)
    lattice = SCHED.lattice()
    w_start = ws[0]
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(w_start[:3], "kpc"),
        p=u.Q(w_start[3:], "kpc/Myr"),
        t=u.Q(lattice[0], "Myr"),
    )
    integrator = gd.integrate.Integrator(
        dfxtra.DiffEqSolver(
            dfx.Dopri8(),
            stepsize_controller=dfx.PIDController(rtol=1e-12, atol=1e-12),
            max_steps=100_000,
        )
    )
    orbit = gd.evaluate_orbit(POT, w0, t=u.Q(lattice, "Myr"), integrator=integrator)
    ref = jnp.concat(
        [
            u.ustrip("kpc", orbit.q.x)[:, None],
            u.ustrip("kpc", orbit.q.y)[:, None],
            u.ustrip("kpc", orbit.q.z)[:, None],
        ],
        axis=1,
    )
    np.testing.assert_allclose(ws[:, :3], ref, rtol=0, atol=1e-8)


def test_lattice_steps_exact() -> None:
    """A StepTo solve over a lattice slice reproduces the saved states."""
    solver = default_solver()
    ws = progenitor_states(POT, W0, SCHED, solver)
    lattice = SCHED.lattice()
    i = 7
    out = solve_on(
        solver,
        host_terms(POT, solver),
        (ws[i, :3], ws[i, 3:]),
        lattice[i:],
    )
    np.testing.assert_allclose(jnp.concat(out), ws[-1], rtol=0, atol=1e-12)


def test_rca_needs_int_max_steps() -> None:
    solver = dfxtra.DiffEqSolver(
        dfx.Dopri5(),
        stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
        max_steps=None,
    )
    with pytest.raises(ValueError, match="max_steps"):
        progenitor_states(POT, W0, SCHED, solver)


def test_failed_solve_gives_nan() -> None:
    """Too few ``max_steps`` gives NaN, under jit and with one lane failing."""
    solver = dfxtra.DiffEqSolver(
        dfx.Dopri5(),
        stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-10),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
        max_steps=3,
    )
    term = dfx.ODETerm(lambda t, y, args: jnp.asarray([y[1], -y[0]]))  # noqa: ARG005
    y0 = jnp.asarray([1.0, 0.0])

    def run(span: Array) -> Array:
        return solve_on(solver, term, y0, jnp.stack([0.0, span]))

    out = jax.jit(run)(jnp.asarray(100.0))
    assert np.all(np.isnan(out))

    spans = jnp.asarray([0.01, 100.0])
    good, bad = jax.jit(jax.vmap(run))(spans)
    np.testing.assert_allclose(good, [np.cos(0.01), -np.sin(0.01)], atol=1e-8)
    assert np.all(np.isnan(bad))
