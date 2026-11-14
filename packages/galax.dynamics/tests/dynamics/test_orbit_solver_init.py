"""`OrbitSolver.init` with batched start times (GalacticDynamics/galax#1027)."""

import diffrax as dfx
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp

FIELD = gd.fields.HamiltonianField(
    gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
)
Q = u.Q([[8.0, 0, 9], [9, 0, 3]], "kpc")
P = u.Q([[0.0, 220, 0], [0, 200, 10]], "km/s")
T0 = u.Q([0.0, 10.0], "Myr")
T1 = u.Q(100.0, "Myr")


def test_one_start_time_per_particle():
    """Each start time goes with its own particle, not with every particle."""
    state = gd.OrbitSolver().init(FIELD, (Q, P), T0, None)
    assert state.t.shape == (2,)
    assert state.y[0].shape == (2, 3)
    assert state.y[1].shape == (2, 3)
    assert jnp.array_equal(state.t, u.ustrip("Myr", T0))
    assert jnp.allclose(state.y[0], u.ustrip("kpc", Q), rtol=1e-15, atol=0)


def test_one_particle_at_several_start_times():
    """A single particle with batched start times is that particle at each."""
    state = gd.OrbitSolver().init(FIELD, (Q[0], P[0]), T0, None)
    assert state.t.shape == (2,)
    assert state.y[0].shape == (2, 3)
    assert jnp.allclose(state.y[0][0], state.y[0][1], rtol=0, atol=0)


def test_solve_from_batched_time_state():
    """Solving from the state equals solving from the initial conditions.

    This uses a solver without a solver state. A solver with one (e.g. the
    default Dopri8) also needs a batched solve not to hand every particle the
    whole batched solver state, which #1025 fixes.
    """
    s = gd.OrbitSolver(
        dfx.SemiImplicitEuler(), stepsize_controller=dfx.ConstantStepSize()
    )
    got = s.solve(FIELD, s.init(FIELD, (Q, P), T0, None), T1, dt0=0.5).ys
    want = s.solve(FIELD, gc.PhaseSpaceCoordinate(q=Q, p=P, t=T0), T1, dt0=0.5).ys
    assert got[0].shape == want[0].shape
    assert jnp.allclose(got[0], want[0], rtol=1e-12, atol=0)
    assert jnp.allclose(got[1], want[1], rtol=1e-12, atol=0)


def test_scalar_start_time_unchanged():
    """A scalar start time with a batch of particles is one batched state."""
    state = gd.OrbitSolver().init(FIELD, (Q, P), T0[0], None)
    assert state.t.shape == ()
    assert state.y[0].shape == (2, 3)


def test_mismatched_batches_raise():
    """Start times and particles that don't broadcast are an error."""
    q3 = jnp.concat([Q, Q[:1]])
    p3 = jnp.concat([P, P[:1]])
    with pytest.raises(ValueError, match="broadcast"):
        gd.OrbitSolver().init(FIELD, (q3, p3), T0, None)
