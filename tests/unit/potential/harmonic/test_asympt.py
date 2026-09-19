"""Tests for the asymptotic power-law continuation of the radial profiles."""

import math

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp

import galax.potential as gp
from galax.potential._src.harmonic.asympt import (
    _pow_diff,
    asymptotic_coeffs,
    eval_log_spline_asympt,
)
from galax.potential._src.harmonic.poisson import solve_poisson_lm
from galax.potential._src.harmonic.project import (
    default_angular_resolution,
    harmonic_coeffs,
    lm_keys,
)
from galax.potential._src.harmonic.spline import (
    eval_log_spline,
    fit_log_spline,
    radial_grid,
)
from galax.potential._src.symmetry import Symmetry

R_MIN, R_MAX, N_R = 0.05, 20.0, 128
"""A deliberately small ``r_max`` so that 2x, 5x and 10x are all real space."""

Y00 = 1.0 / math.sqrt(4.0 * math.pi)
EPS = float(np.finfo(np.float64).eps)

# Closed-form monopoles, with G = M = 1 and a scale radius of 2.
PROFILES = {
    "hernquist": (lambda r: -1.0 / (r + 2.0), gp.HernquistPotential, "m_tot"),
    "plummer": (lambda r: -1.0 / jnp.sqrt(r**2 + 4.0), gp.PlummerPotential, "m_tot"),
    "nfw": (lambda r: -jnp.log1p(0.5 * r) / r, gp.NFWPotential, "m"),
}


def _potential(name, m=1e12):
    """Return the `galax` potential whose monopole is ``PROFILES[name][0]``."""
    _, cls, mass_kw = PROFILES[name]
    return cls(**{mass_kw: m}, r_s=2, units="galactic")


def _splined(phi_of_r, n_r=N_R):
    """Knots, values and natural-spline derivatives of a closed-form monopole."""
    r = radial_grid(n_r, jnp.asarray(R_MIN), jnp.asarray(R_MAX))
    log_r = jnp.log(r)
    values = phi_of_r(r)[:, None]
    return log_r, values, fit_log_spline(log_r, values)


def _expansion(pot):
    """Run the whole build: density -> rho_lm -> Phi_lm -> spline -> tails."""
    keys = lm_keys(0, Symmetry.SPHERICAL)
    n_theta, n_phi = default_angular_resolution(0)
    r = radial_grid(N_R, jnp.asarray(R_MIN), jnp.asarray(R_MAX))
    log_r = jnp.log(r)
    rho_lm = harmonic_coeffs(
        # Not inlinable: `harmonic_coeffs` takes `rho_fn` as a static
        # argument, and an `equinox` bound method is not hashable.
        lambda xyz, t: pot._density(xyz, t),
        r,
        0,
        keys,
        n_theta,
        n_phi,
        jnp.asarray(0.0),
    )
    l = jnp.asarray([0.0])
    phi_lm = solve_poisson_lm(r, rho_lm, l, jnp.asarray(pot.constants["G"].value))
    derivs = fit_log_spline(log_r, phi_lm)
    return log_r, phi_lm, derivs, asymptotic_coeffs(log_r, phi_lm, derivs, l)


# ---------------------------------------------------------------------------
# The four hard requirements.


def test_interior_is_bit_identical_to_the_plain_spline() -> None:
    """Inside the knots the continuation must not perturb anything."""
    log_r, values, derivs = _splined(PROFILES["hernquist"][0])
    coefs = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0]))

    log_rq = jnp.linspace(log_r[0], log_r[-1], 1001)
    got = eval_log_spline_asympt(log_r, values, derivs, coefs, log_rq)
    expect = eval_log_spline(log_r, values, derivs, log_rq)

    assert jnp.array_equal(got, expect)


@pytest.mark.parametrize("name", list(PROFILES))
def test_c1_at_both_boundaries(name) -> None:
    """Value *and* slope join the spline to round-off at ``r_min``/``r_max``.

    Measured one ulp outside the knot range, which is the closest a query can
    get to the join while still taking the continuation branch. A jump here
    is a discontinuous force for an orbit crossing the boundary.
    """
    log_r, values, derivs, coefs = _expansion(_potential(name))

    def phi(log_rq):
        return eval_log_spline_asympt(
            log_r, values, derivs, coefs, jnp.atleast_1d(log_rq)
        )[0, 0]

    for i, toward in ((0, -jnp.inf), (-1, jnp.inf)):
        log_rq = jnp.nextafter(log_r[i], jnp.asarray(toward))
        d_value = abs(float(phi(log_rq) - values[i, 0]) / float(values[i, 0]))
        d_slope = abs(float(jax.grad(phi)(log_rq) - derivs[i, 0]) / float(derivs[i, 0]))
        assert d_value < 8.0 * EPS, (name, i, d_value / EPS)
        assert d_slope < 8.0 * EPS, (name, i, d_slope / EPS)


def test_hessian_is_finite_across_and_beyond_the_boundary() -> None:
    """`hessian` and `tidal_tensor` differentiate through this."""
    log_r, values, derivs, coefs = _expansion(_potential("hernquist"))

    def phi(xyz):
        r = jnp.sqrt(jnp.sum(jnp.atleast_2d(xyz) ** 2, axis=-1))
        return jnp.sum(
            eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(r))[..., 0]
        )

    for boundary in (R_MIN, R_MAX):
        for f in (1e-3, 0.5, 0.999, 1.0, 1.001, 2.0, 10.0, 100.0):
            # Off-axis, so every Hessian entry is exercised.
            xyz = jnp.asarray([0.6, 0.8, 0.0]) * boundary * f
            assert jnp.all(jnp.isfinite(jax.hessian(phi)(xyz))), (boundary, f)


def test_kepler_monopole_is_exact() -> None:
    r"""``Phi = -GM/r`` is precisely the ``v = -l-1`` solution, with ``U = Q = 0``.

    The sharpest available check on the continuation algebra: outward there is
    nothing for the ``x^s`` or ``x^2`` terms to do, so anything but round-off
    means the parametrization is structurally wrong. Inward the same profile
    is the :math:`s = -1` limit of the density-slope guard -- the steepest
    cusp with a convergent enclosed mass -- so it too must come out exact.

    It has to be posed at the coefficient level rather than through
    `harmonic_coeffs`: `KeplerPotential`'s density is a delta function at the
    origin and is identically zero on every grid node, so the projection sees
    nothing at all.
    """
    r = radial_grid(N_R, jnp.asarray(R_MIN), jnp.asarray(R_MAX))
    log_r = jnp.log(r)
    values = (-1.0 / r)[:, None]
    derivs = (1.0 / r)[:, None]  # dPhi/dln r, exact
    coefs = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0]))

    # Outward this is exact: the round-off guard sees that the knot values
    # *are* the solid harmonic and drops the x^s term entirely.
    rq = jnp.asarray([R_MAX * f for f in (2.0, 5.0, 10.0)])
    got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[:, 0]
    assert jnp.allclose(got, -1.0 / rq, rtol=4.0 * EPS, atol=0.0)
    assert float(coefs[1, 2, 0]) == 0.0  # B, the x^s amplitude
    assert float(coefs[1, 3, 0]) == 0.0  # Q, the x^2 amplitude

    # Inward the fit does run, and lands on the guard slope s = -1. Its
    # accuracy is set by how well two adjacent knots pin a slope, ~1e-13
    # here, not by the continuation algebra.
    rq = jnp.asarray([R_MIN * f for f in (0.5, 0.2, 0.1)])
    got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[:, 0]
    assert jnp.allclose(got, -1.0 / rq, rtol=1e-11, atol=0.0)
    assert abs(float(coefs[0, 1, 0]) + 1.0) < 1e-10
    assert float(coefs[0, 3, 0]) == 0.0  # Q


# ---------------------------------------------------------------------------
# Physics: closed-form comparisons outside the grid.


@pytest.mark.parametrize(
    ("name", "out_tol", "in_tol"),
    [("hernquist", 0.08, 0.02), ("plummer", 0.01, 1e-3), ("nfw", 0.20, 0.01)],
)
def test_continuation_matches_closed_form(name, out_tol, in_tol) -> None:
    """Continue an exactly-known monopole and compare against it.

    Only the continuation is under test here: the knot values are the closed
    form itself, so there is no projection or Poisson-solve error to hide
    behind. The residual is set by `fit_log_spline`'s *natural* end condition,
    which forces the second log-derivative to zero exactly where the
    continuation reads its slope; handed analytic boundary derivatives instead
    the same code is 1e-5 to 1e-3 (see the module report).
    """
    phi_of_r = PROFILES[name][0]
    log_r, values, derivs = _splined(phi_of_r)
    coefs = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0]))

    for base, factors, tol in (
        (R_MAX, (2.0, 5.0, 10.0), out_tol),
        (R_MIN, (0.5, 0.2, 0.1), in_tol),
    ):
        rq = jnp.asarray([base * f for f in factors])
        expect = phi_of_r(rq)
        got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[:, 0]
        rel = np.abs(np.asarray(got / expect - 1.0))

        assert jnp.all(got < 0.0), (name, got)  # never flips sign
        assert rel.max() < tol, (name, rel)
        # Monotone: the further from the boundary, the worse -- never better.
        assert np.all(np.diff(rel) > 0.0), (name, rel)

    # The density-slope guards hold: rho no steeper than r^-3 inward,
    # falling faster than r^-2 outward.
    assert float(coefs[0, 1, 0]) >= -1.0
    assert float(coefs[1, 1, 0]) <= 0.0


@pytest.mark.parametrize("name", list(PROFILES))
def test_end_to_end_beats_the_edge_cubic(name) -> None:
    """The whole build, against the analytic potential it was made from.

    The edge-cubic continuation it replaces is not merely inaccurate out
    here: it changes sign and then diverges. This asserts the new tail keeps
    the right sign and is at least an order of magnitude closer.
    """
    pot = _potential(name)
    log_r, values, derivs, coefs = _expansion(pot)

    rq = jnp.asarray([R_MAX * f for f in (2.0, 5.0, 10.0)])
    xyz = jnp.stack([rq, jnp.zeros_like(rq), jnp.zeros_like(rq)], axis=-1)
    expect = pot._potential(xyz, jnp.asarray(0.0))

    new = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[:, 0] * Y00
    old = eval_log_spline(log_r, values, derivs, jnp.log(rq))[:, 0] * Y00

    assert jnp.all(new < 0.0)
    rel_new = np.abs(np.asarray(new / expect - 1.0))
    rel_old = np.abs(np.asarray(old / expect - 1.0))
    assert rel_new.max() < 0.1, (name, rel_new)
    assert np.all(rel_new < 0.1 * rel_old), (name, rel_new, rel_old)


def test_the_edge_cubic_really_does_change_sign() -> None:
    """Pins the behaviour being replaced, so the comparison above has teeth."""
    log_r, values, derivs = _splined(PROFILES["hernquist"][0])
    rq = jnp.asarray(np.geomspace(R_MAX, 10.0 * R_MAX, 40))
    old = eval_log_spline(log_r, values, derivs, jnp.log(rq))[:, 0]
    assert jnp.any(old > 0.0)


# ---------------------------------------------------------------------------
# The degenerate exponents, which are ordinary points here rather than branches.


def test_pow_diff_is_smooth_through_the_degenerate_exponent() -> None:
    r"""``(x^a - x^b)/(a-b) -> x^b ln x`` as ``a -> b``.

    This is the :math:`s = v` case -- the generic outer monopole of an
    :math:`r^{-3}` halo -- and the ``v = 2`` case of the inward quadrupole.
    Both must be finite, accurate and twice differentiable.
    """
    ln_x = jnp.asarray([-3.0, -0.5, 0.0, 0.5, 3.0])
    b = jnp.asarray(-1.0)

    assert jnp.allclose(_pow_diff(b, b, ln_x), jnp.exp(b * ln_x) * ln_x, atol=1e-15)

    for delta in (1e-12, 1e-6, 1e-3, 1e-1):
        got = _pow_diff(b + delta, b, ln_x)
        # Continuity in the exponent, and agreement with the plain quotient
        # wherever that quotient is not itself cancellation noise.
        assert jnp.all(jnp.isfinite(got))
        assert jnp.allclose(got, jnp.exp(b * ln_x) * ln_x, rtol=delta * 2.0, atol=1e-14)

    d2 = jax.hessian(lambda t: jnp.sum(_pow_diff(b, b, jnp.atleast_1d(t))))
    for t in (-1.0, 0.0, 1.0):
        assert jnp.isfinite(d2(jnp.asarray(t)))


def test_the_q_term_carries_a_cored_inner_monopole() -> None:
    r"""The four-parameter ``v = 0`` inward fit, and what it is for.

    A cored profile has :math:`\Phi \simeq \Phi(0) + \frac{2\pi G\rho_0}{3}r^2`
    near the centre; that :math:`r^2` is the ``Q`` term, and without it the
    three-parameter form has to spend its single power law on it. Handed
    accurate boundary derivatives the fit takes the four-parameter branch on
    a Plummer monopole and gains five orders of magnitude inward.

    It is rejected, and the three-parameter form used, when the boundary
    derivatives are the natural-spline ones -- ``agama``'s density-sign
    acceptance test doing its job on data that cannot support four
    parameters. That fallback is the *only* path a realistic build takes
    today; see the report on `fit_log_spline`'s end condition.
    """
    phi_of_r = PROFILES["plummer"][0]
    r = radial_grid(N_R, jnp.asarray(R_MIN), jnp.asarray(R_MAX))
    log_r = jnp.log(r)
    values = phi_of_r(r)[:, None]
    exact = (jax.vmap(jax.grad(phi_of_r))(r) * r)[:, None]

    rq = jnp.asarray([R_MIN * f for f in (0.5, 0.2, 0.1)])
    expect = phi_of_r(rq)

    def err(derivs):
        coefs = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0]))
        got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[:, 0]
        return float(coefs[0, 3, 0]), np.abs(np.asarray(got / expect - 1.0)).max()

    q_exact, err_exact = err(exact)
    q_spline, err_spline = err(fit_log_spline(log_r, values))

    assert q_exact > 0.0  # positive: a positive central density
    assert err_exact < 1e-9
    assert q_spline == 0.0  # rejected, three-parameter form instead
    assert err_spline < 1e-3
