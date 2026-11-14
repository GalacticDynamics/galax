"""Tests for Lyapunov exponents from the variational equations."""

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import pytest

import unxt as u

import galax.dynamics as gd
import galax.potential as gp
from ._variational_helpers import LinearDragField, make_w0
from galax.dynamics._src.orbit.lyapunov import compute_lyapunov
from galax.dynamics._src.orbit.variational import compute_variational

TIGHT = gd.OrbitSolver(stepsize_controller=dfx.PIDController(rtol=1e-11, atol=1e-11))
POT = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
XV0 = jnp.array([8.0, 0, 1, 0, 0.2, 0.02])  # kpc, kpc/Myr


def test_k1_matches_unrenormalized_growth():
    """Renormalizing rescales the deviation, so the growth is unchanged."""
    t1 = 2000.0
    ts, lam = compute_lyapunov(
        POT, make_w0(XV0), u.Q(t1, "Myr"), n_renorm=20, solver=TIGHT
    )
    assert ts.shape == (20,)
    assert lam.shape == (20, 1)
    assert jnp.allclose(ts[-1], t1)

    q0 = jnp.linalg.qr(jnp.ones((6, 1)) + jnp.eye(6, 1))[0]
    _, D = compute_variational(
        POT, make_w0(XV0), u.Q([t1], "Myr"), dw0=q0, solver=TIGHT
    )
    assert jnp.allclose(
        lam[-1, 0] * t1, jnp.log(jnp.linalg.norm(D[0, :, 0])), rtol=1e-6
    )


def test_spectrum_sum_is_log_det():
    """For a conservative field det M = 1, so the exponents sum to zero."""
    pot = gp.MilkyWayPotential()
    _, lam = compute_lyapunov(
        pot, make_w0(XV0), u.Q(2000, "Myr"), n_renorm=20, k=6, solver=TIGHT
    )
    assert lam.shape == (20, 6)
    assert jnp.abs(lam[-1].sum()) < 1e-8


def test_harmonic_oscillator_exponents_vanish():
    pot = gp.HarmonicOscillatorPotential(omega=u.Q(0.1, "1/Myr"), units="galactic")
    t1 = 1e4
    _, lam = compute_lyapunov(
        pot, make_w0(XV0), u.Q(t1, "Myr"), n_renorm=100, k=6, solver=TIGHT
    )
    assert jnp.max(jnp.abs(lam[-1])) < 10 / t1


def test_drag_field_spectrum_sum():
    """With drag, det M = exp(-3 gamma t), so the exponents sum to -3 gamma."""
    gamma = 0.01
    field = LinearDragField(POT, gamma)
    _, lam = compute_lyapunov(
        field, make_w0(XV0), u.Q(500, "Myr"), n_renorm=10, k=6, solver=TIGHT
    )
    assert jnp.allclose(lam[-1].sum(), -3 * gamma, rtol=1e-6)


def test_batched_initial_conditions():
    xv = jnp.stack([XV0, XV0 * jnp.array([1.1, 1, 1, 1, 0.9, 1])])
    ts, lam = compute_lyapunov(POT, make_w0(xv), u.Q(500, "Myr"), n_renorm=5)
    _, lam1 = compute_lyapunov(POT, make_w0(xv[1]), u.Q(500, "Myr"), n_renorm=5)
    assert ts.shape == (2, 5)
    assert lam.shape == (2, 5, 1)
    assert jnp.allclose(lam[1], lam1, rtol=1e-12)


def test_constant_step_backward():
    solver = gd.OrbitSolver(
        gd.experimental.Leapfrog(), stepsize_controller=dfx.ConstantStepSize()
    )
    _, lam = compute_lyapunov(
        POT,
        make_w0(XV0, t=500.0),
        u.Q(0, "Myr"),
        n_renorm=5,
        k=6,
        solver=solver,
        dt0=0.5,
    )
    assert jnp.all(jnp.isfinite(lam))
    assert jnp.abs(lam[-1].sum()) < 1e-10


def test_backward_exponents_match_qr_of_stm():
    """Backward exponents are log|diag R| of QR(M q0), divided by |t1 - t0|.

    The product of the per-interval R factors is the R factor of M q0, so the
    final estimate equals a single QR of the state-transition matrix. This
    pins each of the k exponents, and their sign for a backward run.
    """
    t0, t1, k = 500.0, 0.0, 3
    _, lam = compute_lyapunov(
        POT, make_w0(XV0, t=t0), u.Q(t1, "Myr"), n_renorm=4, k=k, solver=TIGHT
    )
    q0 = jnp.linalg.qr(jnp.ones((6, k)) + jnp.eye(6, k))[0]
    _, D = compute_variational(
        POT, make_w0(XV0, t=t0), u.Q([t1], "Myr"), dw0=q0, solver=TIGHT
    )
    r = jnp.linalg.qr(D[0])[1]
    expected = jnp.log(jnp.abs(jnp.diag(r))) / abs(t1 - t0)
    assert jnp.allclose(lam[-1], expected, rtol=1e-6, atol=1e-10)


def test_zero_duration_raises():
    with pytest.raises(eqx.EquinoxRuntimeError, match="t1 must differ from w0.t"):
        compute_lyapunov(POT, make_w0(XV0, t=100.0), u.Q(100, "Myr"), n_renorm=3)
