"""Tests for the Stäckel-fudge numerical utilities."""

import jax
import numpy as np

import quaxed.numpy as jnp

from galax.actionangle._src._utils import (
    chandrupatla,
    safe_div,
    tang,
)

ROOTS = np.array([0.3, 0.5 ** (1 / 3), np.log(2.0)])


def _f(x):
    assert x.shape == (3,)  # one call on the full width-3 vector
    return jnp.stack([x[0] - 0.3, x[1] ** 3 - 0.5, jnp.exp(x[2]) - 2.0])


def _solve(f=_f, a=None, b=None, fa=None, fb=None, x0=None, done0=None, n_iter=16):
    a = jnp.zeros(3) if a is None else a
    b = jnp.ones(3) if b is None else b
    fa = f(a) if fa is None else fa
    fb = f(b) if fb is None else fb
    x0 = 0.5 * (a + b) if x0 is None else x0
    done0 = jnp.zeros(3, dtype=bool) if done0 is None else done0
    solve = jax.jit(chandrupatla, static_argnums=(0, 7))
    return solve(f, a, b, fa, fb, x0, done0, n_iter)


def test_chandrupatla_roots():
    x = _solve()
    np.testing.assert_allclose(x, ROOTS, rtol=1e-14, atol=0)


def test_chandrupatla_interpolates():
    """Superlinear: 8 iterations suffice (bisection would need ~50)."""
    x = _solve(n_iter=8)
    np.testing.assert_allclose(x, ROOTS, rtol=1e-14, atol=0)


def test_chandrupatla_roots_off_centre_start():
    x = _solve(x0=jnp.asarray([0.9, 0.1, 0.999]))
    np.testing.assert_allclose(x, ROOTS, rtol=1e-14, atol=0)


def test_chandrupatla_no_bracket_nan():
    a = jnp.asarray([0.0, 0.9, 0.0])  # lane 1: f(0.9) > 0 and f(1) > 0
    x = _solve(a=a)
    assert np.isnan(x[1])
    np.testing.assert_allclose(np.asarray(x)[[0, 2]], ROOTS[[0, 2]], rtol=1e-14, atol=0)


def test_chandrupatla_nan_endpoint_nan():
    fa = _f(jnp.zeros(3)).at[2].set(jnp.nan)
    x = _solve(fa=fa)
    assert np.isnan(x[2])
    np.testing.assert_allclose(x[:2], ROOTS[:2], rtol=1e-14, atol=0)


def test_chandrupatla_inf_endpoint():
    fb = jnp.full(3, -jnp.inf)

    def f(x):  # decreasing, so f(1) < 0 is consistent with fb = -inf
        assert x.shape == (3,)
        return -_f(x)

    x = _solve(f=f, fb=fb)
    np.testing.assert_allclose(x, ROOTS, rtol=1e-14, atol=0)


def test_chandrupatla_root_at_endpoint():
    x = _solve(b=jnp.asarray([0.3, 1.0, 1.0]))
    np.testing.assert_array_equal(x[0], 0.3)
    np.testing.assert_allclose(x[1:], ROOTS[1:], rtol=1e-14, atol=0)


def test_chandrupatla_done_lane_untouched():
    x0 = jnp.asarray([0.123, 0.5, 0.5])
    done0 = jnp.asarray([True, False, False])
    x = _solve(x0=x0, done0=done0)
    np.testing.assert_array_equal(x[0], 0.123)
    np.testing.assert_allclose(x[1:], ROOTS[1:], rtol=1e-14, atol=0)


def test_safe_div_value():
    np.testing.assert_array_equal(
        safe_div(jnp.asarray([1.0, 3.0]), jnp.asarray([0.0, 2.0])), [0.0, 1.5]
    )


def test_safe_div_grad_nan_free():
    g = jax.grad(lambda x: safe_div(x, 0.0))(1.0)
    assert g == 0.0
    ga, gb = jax.grad(safe_div, argnums=(0, 1))(1.0, 0.0)
    assert ga == 0.0
    assert gb == 0.0


def test_tang():
    x = jnp.asarray(2.0)
    assert tang(x) == 0.0
    assert jax.grad(lambda y: tang(y) ** 2 + tang(3.0 * y))(x) == 3.0


def test_chandrupatla_nan_f_poisons_lane():
    def f(x):  # lane 0 is NaN beyond 0.6; its first trial (0.65) lands there
        y = _f(x)
        return y.at[0].set(jnp.where(x[0] > 0.6, jnp.nan, y[0]))

    x = _solve(f=f, fb=_f(jnp.ones(3)), x0=jnp.asarray([0.65, 0.5, 0.5]))
    assert np.isnan(x[0])
    np.testing.assert_allclose(x[1:], ROOTS[1:], rtol=1e-14, atol=0)


def test_chandrupatla_x0_sanitized():
    x0 = jnp.asarray([jnp.nan, -1.0, 1.0])  # NaN, outside, on the endpoint
    x = _solve(x0=x0)
    np.testing.assert_allclose(x, ROOTS, rtol=1e-14, atol=0)


def test_chandrupatla_converged_lane_held():
    """A lane that converges early is held at a fixed trial point.

    `f` is still called on every lane each iteration, but a converged lane's
    trial point and result no longer change.
    """
    calls = []

    def f(x):
        jax.debug.callback(lambda v: calls.append(np.asarray(v)), x, ordered=True)
        return _f(x)

    x8 = _solve(f=f, n_iter=8)
    calls.clear()
    x16 = _solve(f=f, n_iter=16)
    np.testing.assert_array_equal(x8, x16)
    # After convergence (≤ 7 iterations here) each lane's trial point repeats.
    late = np.stack(calls[-8:])
    np.testing.assert_array_equal(late, np.broadcast_to(late[0], late.shape))


def test_chandrupatla_inf_endpoint_jacrev_nan_free():
    fb = jnp.full(3, -jnp.inf)

    def solve(a, b, fa, x0):
        def f(x):
            return -_f(x)

        return chandrupatla(f, a, b, fa, fb, x0, jnp.zeros(3, dtype=bool))

    a, b = jnp.zeros(3), jnp.ones(3)
    jac = jax.jacrev(solve, argnums=(0, 1, 2, 3))(a, b, -_f(a), 0.5 * (a + b))
    for j in jac:
        assert np.all(np.isfinite(j))


def test_chandrupatla_settled_lane_not_poisoned():
    """A lane settled at init (f(a) = 0) whose held trial gives NaN keeps a."""
    b = jnp.asarray([1.0, 1.0, 2.0])  # lane 2: stand-in bracket [0, 2]

    def f(x):  # lane 2 is NaN at its held midpoint trial
        return _f(x).at[2].set(jnp.where(x[2] == 1.0, jnp.nan, x[2]))

    fa = f(jnp.zeros(3))
    assert fa[2] == 0
    x = _solve(f=f, b=b, fa=fa, fb=f(b))
    np.testing.assert_array_equal(x[2], 0.0)
    np.testing.assert_allclose(x[:2], ROOTS[:2], rtol=1e-14, atol=0)


def test_chandrupatla_bisection_converges_to_4_ulp():
    """A step function defeats interpolation: bisection to 4 ulp.

    The width test (|b - a| <= 4 eps |x|) alone decides convergence.
    """
    r = jnp.asarray([0.3, 1 / 3, np.pi / 10])

    def f(x):
        return jnp.where(x < r, -1.0, 1.0)

    a, b = r - 1e-6, r + 3.7e-6
    x = _solve(f=f, a=a, b=b, n_iter=60)
    np.testing.assert_array_less(np.abs(x - r), 4 * np.spacing(np.asarray(r)))


def test_chandrupatla_result_carries_no_gradient():
    """The root is stop-gradient: callers reattach the implicit-function tangent."""

    def solve(a, b, x0):
        return chandrupatla(_f, a, b, _f(a), _f(b), x0, jnp.zeros(3, dtype=bool))

    a, b = jnp.zeros(3), jnp.ones(3)
    for jac in (jax.jacfwd, jax.jacrev):
        for j in jac(solve, argnums=(0, 1, 2))(a, b, 0.5 * (a + b)):
            np.testing.assert_array_equal(j, 0.0)
