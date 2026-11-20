"""Tests for the O2GF action and angle linear solves."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from . import _o2gf_ref as ref
from galax.actionangle._src._o2gf_nvecs import LOOP_STEPS, n_vectors
from galax.actionangle._src._o2gf_solve import action_system, angle_system, solve_o2gf

T = 200


def _close(a, e):
    """Compare at rtol 1e-12 with an absolute floor of 1e-12 of the local scale.

    The scale is the largest entry of the row for a matrix and of the whole
    vector for a vector, so small matrix entries are held to 1e-12 of their row.
    """
    a, e = np.asarray(a), np.asarray(e)
    scale = np.abs(e).max(axis=-1, keepdims=True) * np.ones_like(e)
    excess = np.abs(a - e) - 1e-12 * (np.abs(e) + scale)
    assert (excess <= 0).all(), f"worst abs diff {np.abs(a - e).max():.3e}"


NVECS = n_vectors(4, LOOP_STEPS)


def _data(seed=0, T=T):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 30.0, T)
    J = 1.0 + 0.1 * rng.normal(size=(T, 3))
    theta = np.cumsum(rng.uniform(0.1, 0.5, size=(T, 3)), axis=0)  # unwrapped
    return J, theta, t


def test_action_system_matches_gala():
    J, theta, _ = _data()
    A, b = action_system(J, theta, NVECS, jnp.ones(T))
    Ar, br = ref.action_prepare(np.hstack([J, theta]).T, NVECS)
    _close(A, Ar)
    _close(b, br)


def test_angle_system_matches_gala():
    J, theta, t = _data()
    A, b = angle_system(theta, t, NVECS, jnp.ones(T))
    Ar, br = ref.angle_prepare(np.hstack([J, theta]).T, t, NVECS)
    _close(A, Ar)
    _close(b, br)


def test_nan_sample_masked():
    J, theta, t = _data()
    wrapped = np.mod(theta, 2 * np.pi)
    Jn, wn = J.copy(), wrapped.copy()
    Jn[57] = np.nan
    wn[57] = np.nan
    keep = np.arange(T) != 57
    out = solve_o2gf(Jn, wn, t, NVECS)
    exp = solve_o2gf(J[keep], wrapped[keep], t[keep], NVECS)
    assert int(out[6]) == 1
    assert int(exp[6]) == 0
    for a, e in zip(out[:5], exp[:5], strict=True):
        _close(a, e)
    assert np.isfinite(np.asarray(out[0])).all()


@pytest.mark.parametrize("n", [20, 60, 200])
def test_coverage_counts_match_gala(n):
    J, theta, t = _data()
    J, theta, t = J[:n], theta[:n], t[:n]
    out = solve_o2gf(J, np.mod(theta, 2 * np.pi), t, NVECS)
    failed, _ = ref.check_angle_sampling(
        NVECS, np.unwrap(np.mod(theta, 2 * np.pi), axis=0).T
    )
    assert int(out[5]) == len(failed)
    assert n != 20 or len(failed) > 0


def test_solve_grad_finite():
    J, theta, t = _data()
    wrapped = np.mod(theta, 2 * np.pi)
    jac = jax.jacrev(lambda j: solve_o2gf(j, wrapped, t, NVECS)[0])(jnp.asarray(J))
    assert jac.shape == (3, T, 3)
    assert np.isfinite(np.asarray(jac)).all()


def test_solve_grad_finite_with_nan_sample():
    J, theta, t = _data()
    wrapped = np.mod(theta, 2 * np.pi)
    wrapped[57] = np.nan
    Jn = jnp.asarray(J).at[57].set(jnp.nan)
    jac = jax.jacrev(lambda j: solve_o2gf(j, wrapped, t, NVECS)[0])(Jn)
    assert np.isfinite(np.asarray(jac)).all()
    np.testing.assert_array_equal(np.asarray(jac)[:, 57], 0.0)


def test_coarse_sampling_counted():
    """Sparse samples with big angle steps fail rule 2 (span / T > pi)."""
    n = 30
    t = np.linspace(0.0, 30.0, n)
    theta = np.cumsum(np.full((n, 3), 3.0), axis=0) + np.arange(3) * 0.1
    J = 1.0 + 0.01 * np.sin(np.arange(n))[:, None] * np.ones(3)
    out = solve_o2gf(J, np.mod(theta, 2 * np.pi), t, NVECS)
    X = np.unwrap(np.mod(theta, 2 * np.pi), axis=0).T
    failed, kinds = ref.check_angle_sampling(NVECS, X)
    assert (kinds == 1).any()
    # gala counts the exactly resonant n (all components equal rates) as failed
    assert int(out[5]) + int(out[7]) == len(failed)
    assert int(out[7]) > 0
