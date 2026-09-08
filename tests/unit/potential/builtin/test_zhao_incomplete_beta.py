"""Tests for `ZhaoPotential`'s incomplete beta function and analytic derivatives.

`incomplete_beta` implements Zhao (1996) Eq. 43 with two fixed-length series
rather than `jax.scipy.special.hyp2f1`, so it needs checking against
independent references over the whole domain the model uses.

The model's `gradient`/`hessian`/`laplacian` are likewise written analytically
(from Eqs. 15 and 1) rather than by autodiff of `potential`, so they are
checked against autodiff of the potential here.
"""

import jax
import numpy as np
import pytest
from scipy.special import beta as scipy_beta, betainc as scipy_betainc, hyp2f1

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.builtin.zhao import incomplete_beta

# a > 0 always (Zhao uses a = alpha*(3-gamma) or alpha*(beta-2)); b is any real
# (b = alpha*(beta-3) <= 0 for the infinite-mass models, alpha*(2-gamma) <= 0
# for gamma >= 2, e.g. Jaffe).
AS = [0.25, 0.5, 1.0, 1.31, 2.0, 4.5]
BS = [-2.5, -1.0, -0.1, 0.0, 0.3, 1.31, 4.0]
ZS = [1e-6, 1e-3, 0.1, 0.3, 0.5, 0.5 + 1e-7, 0.7, 0.9, 0.99, 1 - 1e-6]


@pytest.mark.parametrize("a", AS)
@pytest.mark.parametrize("b", BS)
def test_incomplete_beta_matches_hyp2f1_identity(a: float, b: float) -> None:
    """Must match B(a,b,z) = z^a/a * 2F1(a, 1-b; a+1; z) (DLMF 8.17.7)."""
    z = np.array(ZS)
    got = np.asarray(incomplete_beta(a, b, jnp.asarray(z)))
    expect = z**a / a * hyp2f1(a, 1.0 - b, a + 1.0, z)
    np.testing.assert_allclose(got, expect, rtol=1e-11)


@pytest.mark.parametrize("a", AS)
@pytest.mark.parametrize("b", [b for b in BS if b > 0])
def test_incomplete_beta_matches_regularized(a: float, b: float) -> None:
    """For b > 0 it must match `beta(a, b) * betainc(a, b, z)`."""
    z = np.array(ZS)
    got = np.asarray(incomplete_beta(a, b, jnp.asarray(z)))
    expect = scipy_beta(a, b) * scipy_betainc(a, b, z)
    np.testing.assert_allclose(got, expect, rtol=1e-11)


@pytest.mark.parametrize("b", [-1.0, 0.0, 2.0])
def test_incomplete_beta_finite_where_complete_beta_diverges(b: float) -> None:
    """`b <= 0` must stay finite: the divergence is in `beta(a, b)` alone."""
    got = incomplete_beta(2.0, b, jnp.asarray([0.1, 0.5, 0.9, 1 - 1e-6]))
    assert jnp.all(jnp.isfinite(got))


@pytest.mark.parametrize("a", [0.5, 2.0])
@pytest.mark.parametrize("b", [-1.0, 0.0, 1.5])
def test_incomplete_beta_derivative(a: float, b: float) -> None:
    """The custom JVP must equal the Eq. 43 integrand, z^(a-1) (1-z)^(b-1)."""
    for z in [0.1, 0.49, 0.51, 0.9]:
        got = jax.grad(lambda zz: incomplete_beta(a, b, zz))(jnp.asarray(z))
        expect = z ** (a - 1.0) * (1.0 - z) ** (b - 1.0)
        np.testing.assert_allclose(np.asarray(got), expect, rtol=1e-10)


# ===================================================================
# Analytic derivatives vs autodiff of the potential

ABG = [
    (1.0, 4.0, 1.0),  # Hernquist
    (1.0, 4.0, 2.0),  # Jaffe (p0 = 0)
    (0.5, 5.0, 0.0),  # Plummer
    (1.0, 3.0, 1.0),  # NFW (q0 = 0, infinite mass)
    (0.9, 4.31, 1.2),  # generic
]
RADII = [0.05, 0.5, 1.0, 5.0, 50.0]


def _pot(abg: tuple[float, float, float]) -> gp.ZhaoPotential:
    alpha, beta, gamma = abg
    return gp.ZhaoPotential(
        m=u.Quantity(1e12, "Msun"),
        r_s=u.Quantity(8.0, "kpc"),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        units="galactic",
    )


@pytest.mark.parametrize("abg", ABG)
@pytest.mark.parametrize("r", RADII)
def test_analytic_gradient_matches_autodiff(abg, r: float) -> None:
    """The shell-theorem gradient must match `jax.grad` of the potential."""
    pot = _pot(abg)
    xyz = jnp.asarray([0.3, -0.5, 0.81]) * r
    xyz = xyz / jnp.linalg.norm(xyz) * r

    got = pot.gradient(xyz, t=0)
    expect = jax.grad(lambda q: pot._potential(q, 0.0))(xyz)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-8)


@pytest.mark.parametrize("abg", ABG)
@pytest.mark.parametrize("r", RADII)
def test_analytic_hessian_matches_autodiff(abg, r: float) -> None:
    """The analytic hessian must match `jax.hessian` of the potential."""
    pot = _pot(abg)
    xyz = jnp.asarray([0.3, -0.5, 0.81])
    xyz = xyz / jnp.linalg.norm(xyz) * r

    got = pot.hessian(xyz, t=0)
    expect = jax.hessian(lambda q: pot._potential(q, 0.0))(xyz)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-6)


@pytest.mark.parametrize("abg", ABG)
@pytest.mark.parametrize("r", RADII)
def test_laplacian_is_poisson(abg, r: float) -> None:
    """Poisson's equation: the laplacian must be 4 pi G rho (Eq. 1)."""
    pot = _pot(abg)
    xyz = jnp.asarray([0.3, -0.5, 0.81])
    xyz = xyz / jnp.linalg.norm(xyz) * r

    lap = pot.laplacian(xyz, t=0)
    rho = pot.density(xyz, t=0)
    expect = 4 * jnp.pi * pot.constants["G"].value * rho
    np.testing.assert_allclose(np.asarray(lap), np.asarray(expect), rtol=1e-10)
