"""Tests for the experimental spray orbit fields."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diffraxtra as dfxtra

import galax.fields as gf
import galax.potential as gp
from galax.dynamics._src.experimental.fields import (
    _CopiesField,
    _TrackField,
    _WholeStreamField,
)

HOST = gp.MilkyWayPotential(units="galactic")
PROG = gp.PlummerPotential(m_tot=1e5, r_s=0.004, units="galactic")
PROG_PC = gp.PlummerPotential(m_tot=1e5, r_s=4.0, units=("pc", "Myr", "Msun", "radian"))
Q = jnp.asarray([[8.0, 0.0, 0.1], [8.01, 0.02, 0.1], [7.9, 0.1, -0.05]])
P = jnp.asarray([[0.0, 0.2, 0.0], [0.01, 0.2, 0.0], [0.0, 0.19, 0.01]])
T = jnp.asarray(3.0)
LEAP = dfxtra.DiffEqSolver(gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize())
DOPRI = dfxtra.DiffEqSolver(dfx.Dopri8())


def _acc(pot, q):
    return -gp.gradient(pot, q, T)


def _dp(field, solver, q, p, args=None):
    """Return (dq/dt, dp/dt) from the field's terms, for either solver."""
    terms = field.terms(solver)
    if isinstance(solver.solver, gf.Leapfrog):
        return terms[0].vf(T, p, args), terms[1].vf(T, q, args)
    dw = terms.vf(T, jnp.concat([q, p], axis=-1), args)
    return dw[..., :3], dw[..., 3:]


@pytest.mark.parametrize("solver", [LEAP, DOPRI])
def test_field_accelerations(solver) -> None:
    host = _acc(HOST, Q)
    kick = lambda x: _acc(PROG, x)

    # whole stream: row 0 host only; tracers host + progenitor at row 0
    f = _WholeStreamField(HOST, PROG)
    act = jnp.ones(2, bool)
    dq, dp = _dp(f, solver, Q, P, {"active": act})
    np.testing.assert_allclose(dq, P, rtol=1e-14)
    want = host.at[1:].add(kick(Q[1:] - Q[0]))
    np.testing.assert_allclose(dp, want, rtol=1e-14)

    # no progenitor potential: host only
    _, dp = _dp(_WholeStreamField(HOST, None), solver, Q, P, {"active": act})
    np.testing.assert_allclose(dp, host, rtol=1e-14)

    # copies: rows [particle, progenitor]
    _, dp = _dp(_CopiesField(HOST, PROG), solver, Q[:2], P[:2])
    want = host[:2].at[0].add(kick(Q[0] - Q[1]))
    np.testing.assert_allclose(dp, want, rtol=1e-14)

    # track: one particle about track(t)
    centre = jnp.asarray([7.0, 0.5, 0.0])
    f = _TrackField(HOST, PROG, lambda t: centre + 0 * t)
    dq, dp = _dp(f, solver, Q[:1], P[:1])
    np.testing.assert_allclose(dq, P[:1], rtol=1e-14)
    want = host[:1] + kick(Q[:1] - centre)
    np.testing.assert_allclose(dp, want, rtol=1e-14)


@pytest.mark.parametrize("solver", [LEAP, DOPRI])
def test_inactive_tracers_frozen(solver) -> None:
    f = _WholeStreamField(HOST, PROG)
    on = jnp.asarray([True, True])
    off = jnp.asarray([True, False])
    dq1, dp1 = _dp(f, solver, Q, P, {"active": on})
    dq2, dp2 = _dp(f, solver, Q, P, {"active": off})
    assert jnp.all(dq2[2] == 0)
    assert jnp.all(dp2[2] == 0)
    np.testing.assert_array_equal(dq2[:2], dq1[:2])
    np.testing.assert_array_equal(dp2[:2], dp1[:2])


@pytest.mark.parametrize(
    "make",
    [
        lambda p: _WholeStreamField(HOST, p),
        lambda p: _CopiesField(HOST, p),
        lambda p: _TrackField(HOST, p, lambda t: t * jnp.ones(3)),
    ],
)
def test_units_must_match(make) -> None:
    with pytest.raises(ValueError, match="same unit system"):
        make(PROG_PC)


@pytest.mark.parametrize("solver", [LEAP, DOPRI])
def test_inactive_tracer_at_progenitor_grad_finite(solver) -> None:
    q = Q.at[2].set(Q[0])  # inactive tracer exactly at the progenitor
    act = jnp.asarray([True, False])

    def f(q, p, m):
        field = _WholeStreamField(HOST, gp.KeplerPotential(m_tot=m, units="galactic"))
        dq, dp = _dp(field, solver, q, p, {"active": act})
        return jnp.sum(dq) + jnp.sum(dp)

    gq, gp_, gm = jax.grad(f, argnums=(0, 1, 2))(q, P, jnp.asarray(1e5))
    for g in (gq, gp_, gm):
        assert jnp.all(jnp.isfinite(g))
