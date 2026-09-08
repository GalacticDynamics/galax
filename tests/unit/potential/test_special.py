"""Tests for `galax.potential._src.special`.

`incomplete_beta` sums two fixed-length series rather than calling
`jax.scipy.special.hyp2f1`, so it is checked against independent references
over the whole domain the potential models use.
"""

import jax
import numpy as np
import pytest
from scipy.special import beta as scipy_beta, betainc as scipy_betainc, hyp2f1

import quaxed.numpy as jnp

from galax.potential._src.special import incomplete_beta

# `a > 0` always (e.g. Zhao uses a = alpha*(3-gamma) or alpha*(beta-2)); `b` is
# any real (b = alpha*(beta-3) <= 0 for the infinite-mass models, and
# alpha*(2-gamma) <= 0 for gamma >= 2, e.g. Jaffe).
AS = [0.25, 0.5, 1.0, 1.31, 2.0, 4.5]
BS = [-2.5, -1.0, -0.1, 0.0, 0.3, 1.31, 4.0]
ZS = [1e-6, 1e-3, 0.1, 0.3, 0.5, 0.5 + 1e-7, 0.7, 0.9, 0.99, 1 - 1e-6]


@pytest.mark.parametrize("a", AS)
@pytest.mark.parametrize("b", BS)
def test_matches_hyp2f1_identity(a: float, b: float) -> None:
    """Must match B(a,b,z) = z^a/a * 2F1(a, 1-b; a+1; z) (DLMF 8.17.7)."""
    z = np.array(ZS)
    got = np.asarray(incomplete_beta(a, b, jnp.asarray(z)))
    expect = z**a / a * hyp2f1(a, 1.0 - b, a + 1.0, z)
    np.testing.assert_allclose(got, expect, rtol=1e-11)


@pytest.mark.parametrize("a", AS)
@pytest.mark.parametrize("b", [b for b in BS if b > 0])
def test_matches_regularized(a: float, b: float) -> None:
    """For b > 0 it must match `beta(a, b) * betainc(a, b, z)`."""
    z = np.array(ZS)
    got = np.asarray(incomplete_beta(a, b, jnp.asarray(z)))
    expect = scipy_beta(a, b) * scipy_betainc(a, b, z)
    np.testing.assert_allclose(got, expect, rtol=1e-11)


@pytest.mark.parametrize("b", [-1.0, 0.0, 2.0])
def test_finite_where_complete_beta_diverges(b: float) -> None:
    """`b <= 0` must stay finite: the divergence is in `beta(a, b)` alone."""
    got = incomplete_beta(2.0, b, jnp.asarray([0.1, 0.5, 0.9, 1 - 1e-6]))
    assert jnp.all(jnp.isfinite(got))


@pytest.mark.parametrize("a", [0.5, 2.0])
@pytest.mark.parametrize("b", [1.0, 2.5])
def test_endpoints(a: float, b: float) -> None:
    """B(a,b,0) = 0, and for b > 0, B(a,b,1) is the complete beta function."""
    got = incomplete_beta(a, b, jnp.asarray([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(got[0]), 0.0, atol=1e-300)
    np.testing.assert_allclose(np.asarray(got[1]), scipy_beta(a, b), rtol=1e-11)


@pytest.mark.parametrize("a", [0.5, 2.0])
@pytest.mark.parametrize("b", [-1.0, 0.0, 1.5])
def test_derivative(a: float, b: float) -> None:
    """The custom JVP must equal the integrand, z^(a-1) (1-z)^(b-1)."""
    for z in [0.1, 0.49, 0.51, 0.9]:
        got = jax.grad(lambda zz: incomplete_beta(a, b, zz))(jnp.asarray(z))
        expect = z ** (a - 1.0) * (1.0 - z) ** (b - 1.0)
        np.testing.assert_allclose(np.asarray(got), expect, rtol=1e-10)


@pytest.mark.parametrize("a", [0.5, 2.0])
@pytest.mark.parametrize("b", [-1.0, 0.5])
def test_parameter_derivatives(a: float, b: float) -> None:
    """The a/b tangents must match finite differences of the primal."""
    z = jnp.asarray(0.7)
    eps = 1e-6
    for argnum, (da, db) in enumerate([(eps, 0.0), (0.0, eps)]):
        got = jax.grad(incomplete_beta, argnums=argnum)(a, b, z)
        expect = (
            incomplete_beta(a + da, b + db, z) - incomplete_beta(a - da, b - db, z)
        ) / (2 * eps)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-5)


def test_second_derivative_composes() -> None:
    """`custom_jvp` (not `custom_vjp`) must survive `jacfwd(jacrev(...))`."""
    f = lambda zz: incomplete_beta(2.0, 1.5, zz)
    got = jax.jacfwd(jax.jacrev(f))(jnp.asarray(0.3))
    # d2/dz2 B(a,b,z) = (a-1) z^(a-2) (1-z)^(b-1) - (b-1) z^(a-1) (1-z)^(b-2)
    a, b, z = 2.0, 1.5, 0.3
    expect = (a - 1) * z ** (a - 2) * (1 - z) ** (b - 1) - (b - 1) * z ** (a - 1) * (
        1 - z
    ) ** (b - 2)
    np.testing.assert_allclose(np.asarray(got), expect, rtol=1e-8)


def test_batched_matches_scalar() -> None:
    """A batched call must equal the scalar calls, elementwise."""
    z = jnp.asarray(ZS)
    batched = incomplete_beta(1.31, -0.5, z)
    scalar = jnp.stack([incomplete_beta(1.31, -0.5, zz) for zz in z])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(scalar), rtol=1e-14)
