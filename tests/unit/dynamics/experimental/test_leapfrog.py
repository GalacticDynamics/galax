# ruff: noqa: ARG005

import diffrax as dfx
import jax.numpy as jnp
import pytest

from galax.dynamics import experimental


def shaped_allclose(x, y, **kwargs):
    return jnp.shape(x) == jnp.shape(y) and jnp.allclose(x, y, **kwargs)


@pytest.mark.parametrize("solver", [experimental.Leapfrog()])
def test_symplectic_solvers(solver):
    dq_dt = dfx.ODETerm(lambda t, p, args: p)
    dp_dt = dfx.ODETerm(lambda t, q, args: -q)
    y0 = (1.0, -0.5)
    dt0 = 0.00001
    sol1 = dfx.diffeqsolve(
        (dq_dt, dp_dt),
        solver,
        0,
        1,
        dt0,
        y0,
        max_steps=100000,
    )
    term_combined = dfx.ODETerm(lambda t, y, args: (y[1], -y[0]))
    sol2 = dfx.diffeqsolve(term_combined, dfx.Tsit5(), 0, 1, 0.001, y0)
    assert shaped_allclose(sol1.ys[0], sol2.ys[0])
    assert shaped_allclose(sol1.ys[1], sol2.ys[1])
