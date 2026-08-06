"""Tests for potential derivatives at exactly r=0.

`jnp.sqrt` has an infinite derivative at zero, so a radius built as
``sqrt(x**2 + y**2 + z**2)`` made `gradient` and `hessian` NaN at the origin for
every potential defined in terms of ``r``. See `galax.potential._src.utils.safe_sqrt`.
"""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.utils import safe_sqrt

# Potentials that are regular at the origin, with the analytic value of
# d^2Phi/dx^2 there (the Hessian is isotropic, so the diagonal suffices).
M_TOT = u.Quantity(1e12, "Msun")
R_S = u.Quantity(5.0, "kpc")

REGULAR = [
    # Plummer: Phi = -GM / sqrt(r^2 + b^2)  =>  d2Phi/dx2(0) = GM / b^3
    (gp.PlummerPotential(m_tot=M_TOT, r_s=R_S, units="galactic"), 3),
    # Isochrone: Phi = -GM / (b + sqrt(r^2 + b^2))  =>  d2Phi/dx2(0) = GM / (4 b^3)
    (gp.IsochronePotential(m_tot=M_TOT, r_s=R_S, units="galactic"), 12),
]

# Cuspy but still expected to be non-NaN: the net force at the exact centre of a
# spherically symmetric system is zero by symmetry.
CUSPY = [
    gp.HernquistPotential(m_tot=M_TOT, r_s=R_S, units="galactic"),
    gp.TriaxialHernquistPotential(
        m_tot=M_TOT, r_s=R_S, q1=1.0, q2=0.9, units="galactic"
    ),
    gp.LogarithmicPotential(v_c=u.Quantity(220, "km/s"), r_s=R_S, units="galactic"),
    gp.StoneOstriker15Potential(
        m_tot=M_TOT, r_h=u.Quantity(10.0, "kpc"), r_c=R_S, units="galactic"
    ),
]

# Known remaining limitation, from a *second* and independent cause: these
# profiles are written in a form whose r-derivative is itself 0/0 at r=0 (e.g.
# NFW's d/dr[log1p(x)/x] = [x/(1+x) - log1p(x)]/x^2). Making the radius
# differentiable does not help; the profile formula needs a small-x series.
# Their potential *values* at the origin are correct.
STILL_NAN_GRADIENT = [
    gp.NFWPotential(m=M_TOT, r_s=R_S, units="galactic"),
    gp.BurkertPotential(m=M_TOT, r_s=R_S, units="galactic"),
]

ORIGIN = jnp.zeros(3)


def test_safe_sqrt_value_is_exact() -> None:
    """`safe_sqrt` must not perturb the value of `jnp.sqrt`."""
    x = jnp.asarray([1e-30, 1e-6, 1.0, 4.0, 1e30])
    assert jnp.array_equal(safe_sqrt(x), jnp.sqrt(x))


def test_safe_sqrt_derivative_is_finite_at_zero() -> None:
    """...but its derivative at zero must be finite, unlike `jnp.sqrt`."""
    assert jnp.isinf(jax.grad(jnp.sqrt)(jnp.asarray(0.0)))
    assert jnp.isfinite(jax.grad(safe_sqrt)(jnp.asarray(0.0)))


@pytest.mark.parametrize(("pot", "denom"), REGULAR, ids=lambda p: type(p).__name__)
def test_regular_potential_hessian_at_origin(
    pot: gp.AbstractPotential, denom: int
) -> None:
    """Cored potentials have the exact analytic gradient & Hessian at r=0."""
    G = pot.constants["G"].value
    m_tot = pot.m_tot(0.0, ustrip=pot.units["mass"])
    r_s = pot.r_s(0.0, ustrip=pot.units["length"])

    assert jnp.allclose(pot.gradient(ORIGIN, 0.0), jnp.zeros(3), atol=1e-12)
    expected = G * m_tot / (denom / 3 * r_s**3)
    assert jnp.allclose(pot.hessian(ORIGIN, 0.0), expected * jnp.eye(3), rtol=1e-8)


@pytest.mark.parametrize("pot", CUSPY, ids=lambda p: type(p).__name__)
def test_cuspy_potential_gradient_at_origin(pot: gp.AbstractPotential) -> None:
    """Cuspy potentials give zero net force at the centre, not NaN."""
    assert jnp.all(jnp.isfinite(pot.gradient(ORIGIN, 0.0)))
    assert jnp.allclose(pot.gradient(ORIGIN, 0.0), jnp.zeros(3), atol=1e-12)


def test_potential_value_matches_limit_at_origin() -> None:
    """NFW's Phi = -G m log1p(r/r_s) / r has a removable singularity at r=0."""
    pot = gp.NFWPotential(m=M_TOT, r_s=R_S, units="galactic")
    G = pot.constants["G"].value
    expected = -G * 1e12 / 5.0
    assert jnp.allclose(pot.potential(ORIGIN, 0.0), expected, rtol=1e-8)


@pytest.mark.parametrize(
    "pot", [p for p, _ in REGULAR] + CUSPY, ids=lambda p: type(p).__name__
)
def test_gradient_is_continuous_into_the_origin(pot: gp.AbstractPotential) -> None:
    """The value just off the origin stays finite, i.e. no NaN boundary layer."""
    near = jnp.asarray([1e-12, 0.0, 0.0])
    assert jnp.all(jnp.isfinite(pot.gradient(near, 0.0)))
    assert jnp.isfinite(pot.potential(near, 0.0))


@pytest.mark.parametrize("pot", STILL_NAN_GRADIENT, ids=lambda p: type(p).__name__)
def test_profile_formula_singularity_is_a_separate_issue(
    pot: gp.AbstractPotential,
) -> None:
    """Pin the known limitation so that fixing the profiles trips this test.

    The potential value at the origin is correct; only the gradient is NaN.
    """
    assert jnp.isfinite(pot.potential(ORIGIN, 0.0))
    assert jnp.all(jnp.isnan(pot.gradient(ORIGIN, 0.0)))
