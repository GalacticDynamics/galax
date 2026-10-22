"""Tests for the Stäckel-fudge numerical utilities."""

import jax
import numpy as np
import pytest

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


def _nan_beyond(x):  # lane 0 is NaN beyond 0.6; its first trial (0.65) lands there
    y = _f(x)
    return y.at[0].set(jnp.where(x[0] > 0.6, jnp.nan, y[0]))


def _settled(x):  # lane 2: f(0) = 0, NaN at its held midpoint trial 1.0
    return _f(x).at[2].set(jnp.where(x[2] == 1.0, jnp.nan, x[2]))


def _neg_f(x):  # decreasing, so f(1) < 0 is consistent with fb = -inf
    return -_f(x)


# (solve kwargs, {lane: expected}); the other lanes are ROOTS to 1e-14
_LANES = {
    "off_centre_start": ({"x0": [0.9, 0.1, 0.999]}, {}),
    "x0_sanitized": ({"x0": [np.nan, -1.0, 1.0]}, {}),  # NaN, outside, endpoint
    "inf_endpoint": ({"f": _neg_f, "fb": [-np.inf] * 3}, {}),
    "no_bracket": ({"a": [0.0, 0.9, 0.0]}, {1: np.nan}),  # f(0.9), f(1) > 0
    "nan_endpoint": ({"fa": [-0.3, -0.5, np.nan]}, {2: np.nan}),
    "nan_f_poisons_lane": (
        {"f": _nan_beyond, "fb": list(_f(jnp.ones(3))), "x0": [0.65, 0.5, 0.5]},
        {0: np.nan},
    ),
    "root_at_endpoint": ({"b": [0.3, 1.0, 1.0]}, {0: 0.3}),
    "done_lane_untouched": (
        {"x0": [0.123, 0.5, 0.5], "done0": [True, False, False]},
        {0: 0.123},
    ),
    # a lane settled at init (f(a) = 0) whose held trial gives NaN keeps a
    "settled_lane_not_poisoned": (
        {"f": _settled, "b": [1.0, 1.0, 2.0]},  # lane 2: stand-in bracket [0, 2]
        {2: 0.0},
    ),
}


@pytest.mark.parametrize("case", list(_LANES))
def test_chandrupatla_lanes(case):
    kw, special = _LANES[case]
    kw = {k: v if callable(v) else jnp.asarray(v) for k, v in kw.items()}
    x = np.asarray(_solve(**kw))
    rest = [i for i in range(3) if i not in special]
    np.testing.assert_allclose(x[rest], ROOTS[rest], rtol=1e-14, atol=0)
    for i, v in special.items():
        np.testing.assert_array_equal(x[i], v)


def test_safe_div():
    """0 where b = 0, with a NaN-free gradient."""
    np.testing.assert_array_equal(
        safe_div(jnp.asarray([1.0, 3.0]), jnp.asarray([0.0, 2.0])), [0.0, 1.5]
    )
    ga, gb = jax.grad(safe_div, argnums=(0, 1))(1.0, 0.0)
    assert ga == 0.0
    assert gb == 0.0


def test_tang():
    x = jnp.asarray(2.0)
    assert tang(x) == 0.0
    assert jax.grad(lambda y: tang(y) ** 2 + tang(3.0 * y))(x) == 3.0


def test_chandrupatla_interpolates_and_holds():
    """Superlinear, and a converged lane is held at a fixed trial point.

    8 iterations reach the roots to 1e-14 (bisection would need ~50), and 16
    give the same result: `f` is still called on every lane each iteration,
    but a converged lane's trial point and result no longer change.
    """
    calls = []

    def f(x):
        jax.debug.callback(lambda v: calls.append(np.asarray(v)), x, ordered=True)
        return _f(x)

    x8 = _solve(f=f, n_iter=8)
    np.testing.assert_allclose(x8, ROOTS, rtol=1e-14, atol=0)
    calls.clear()
    x16 = _solve(f=f, n_iter=16)
    np.testing.assert_array_equal(x8, x16)
    # After convergence (≤ 7 iterations here) each lane's trial point repeats.
    late = np.stack(calls[-8:])
    np.testing.assert_array_equal(late, np.broadcast_to(late[0], late.shape))


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
    """The root is stop-gradient: callers reattach the implicit-function tangent.

    Zero (so also NaN-free) in both modes, with an ``-inf`` endpoint.
    """
    fb = jnp.full(3, -jnp.inf)

    def solve(a, b, fa, x0):
        return chandrupatla(_neg_f, a, b, fa, fb, x0, jnp.zeros(3, dtype=bool))

    a, b = jnp.zeros(3), jnp.ones(3)
    for jac in (jax.jacfwd, jax.jacrev):
        for j in jac(solve, argnums=(0, 1, 2, 3))(a, b, _neg_f(a), 0.5 * (a + b)):
            np.testing.assert_array_equal(j, 0.0)
