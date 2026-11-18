"""Test `galax.fields.Leapfrog` on a harmonic oscillator, with no orbit code."""

import diffrax as dfx
import jax.numpy as jnp
import pytest

import galax.fields as gf

# dq/dt = p, dp/dt = -q; exact solution q = cos t, p = -sin t from (1, 0).
TERMS = (dfx.ODETerm(lambda _t, p, _args: p), dfx.ODETerm(lambda _t, q, _args: -q))
Y0 = (jnp.array(1.0), jnp.array(0.0))


def _solve(t1: float, dt0: float, y0=Y0):
    sol = dfx.diffeqsolve(
        TERMS, gf.Leapfrog(), t0=0.0, t1=t1, dt0=dt0, y0=y0, max_steps=100_000
    )
    return sol.ys[0][-1], sol.ys[1][-1]


def test_second_order() -> None:
    """Halving the step cuts the error by about 4."""
    errs = []
    for dt in (0.1, 0.05, 0.025):
        q, p = _solve(1.0, dt)
        errs.append(jnp.hypot(q - jnp.cos(1.0), p + jnp.sin(1.0)))
    ratios = [float(errs[i] / errs[i + 1]) for i in range(2)]
    assert ratios == pytest.approx([4.0, 4.0], rel=0.05)


def test_energy_bounded() -> None:
    """The energy error stays small over many periods instead of drifting."""
    q, p = _solve(200 * 2 * jnp.pi, 0.05)
    assert abs(float(0.5 * (q**2 + p**2)) - 0.5) < 1e-3


def test_backward_round_trip() -> None:
    """Integrating forward then backward returns to the start."""
    y1 = _solve(10.0, 0.01)
    sol = dfx.diffeqsolve(TERMS, gf.Leapfrog(), t0=10.0, t1=0.0, dt0=-0.01, y0=y1)
    atol = 1e3 * float(jnp.finfo(sol.ys[0].dtype).eps)  # float32 or float64
    assert float(sol.ys[0][-1]) == pytest.approx(1.0, abs=atol)
    assert float(sol.ys[1][-1]) == pytest.approx(0.0, abs=atol)


def test_func() -> None:
    """`func` returns the time derivative ``(dq/dt, dp/dt) = (p, -q)``."""
    qdot, pdot = gf.Leapfrog().func(TERMS, 0.0, (jnp.array(2.0), jnp.array(3.0)), None)
    assert (float(qdot), float(pdot)) == (3.0, -2.0)
