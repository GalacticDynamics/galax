"""Tests for the asymptotic power-law continuation of the radial profiles."""

import itertools
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
from galax.potential._src.harmonic.spline import eval_log_spline, fit_log_spline
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
    """Knots, values and splined derivatives of a closed-form monopole."""
    r = jnp.geomspace(R_MIN, R_MAX, n_r)
    log_r = jnp.log(r)
    values = phi_of_r(r)[:, None]
    return log_r, values, fit_log_spline(log_r, values)


def _expansion(pot):
    """Run the whole build: density -> rho_lm -> Phi_lm -> spline -> tails."""
    keys = lm_keys(0, Symmetry.SPHERICAL)
    n_theta, n_phi = default_angular_resolution(0)
    r = jnp.geomspace(R_MIN, R_MAX, N_R)
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
    r = jnp.geomspace(R_MIN, R_MAX, N_R)
    log_r = jnp.log(r)
    values = (-1.0 / r)[:, None]
    derivs = (1.0 / r)[:, None]  # dPhi/dln r, exact
    coefs = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0]))

    # Outward this is exact: the round-off guard sees that the knot values
    # *are* the solid harmonic and drops the x^s term entirely.
    rq = jnp.asarray([R_MAX * f for f in (2.0, 5.0, 10.0)])
    got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[:, 0]
    assert jnp.allclose(got, -1.0 / rq, rtol=4.0 * EPS, atol=0.0)
    # Only B exists to check: the default build carries no Q row, and JAX
    # clamps an out-of-range index rather than raising, so asserting on
    # `coefs[1, 3, 0]` would silently re-read B and could never fail.
    assert float(coefs[1, 2, 0]) == 0.0  # B, the x^s amplitude

    # Inward the fit does run, and lands on the guard slope s = -1. Its
    # accuracy is set by how well two adjacent knots pin a slope, ~1e-13
    # here, not by the continuation algebra.
    rq = jnp.asarray([R_MIN * f for f in (0.5, 0.2, 0.1)])
    got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[:, 0]
    assert jnp.allclose(got, -1.0 / rq, rtol=1e-11, atol=0.0)
    assert abs(float(coefs[0, 1, 0]) + 1.0) < 1e-10
    assert coefs.shape[1] == 3, "no Q row without `cored_monopole`"


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
    behind. What is left is the boundary derivative `fit_log_spline` hands
    over, which the continuation reads its slope from -- see that function
    on why the end condition is load-bearing rather than incidental.
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
    here: it changes sign and then diverges.

    Close to the boundary the two are comparable -- both read the same knot
    derivatives, and with a not-a-knot end condition those are good -- so the
    tail is not asserted to win at 2 r_max. What it must do is not fall
    apart: the cubic's error grows by two orders across 2 -> 10 r_max while
    the tail's stays flat, and the tail never changes sign.

        r/r_max      tail     edge cubic     (hernquist)
            2.0    0.0101         0.0092
            5.0    0.0176         0.1297
           10.0    0.0228         1.4320

    The NFW bound is looser, and not because of the continuation: at
    ``r_max / r_s = 10`` an NFW encloses nowhere near its (logarithmically
    divergent) mass, so `solve_poisson_lm`'s own outer-tail truncation
    already leaves ``Phi_00`` 4.6e-2 wrong *at* ``r_max`` -- against 5.7e-3
    for Hernquist and 4.2e-4 for Plummer. The continuation faithfully
    extrapolates a boundary value that is itself off. On the closed-form NFW
    monopole, where there is no such error, the tail is 2.5e-2 at 10 r_max
    and falls to 1e-4 as the grid widens.
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
    tol = 0.25 if name == "nfw" else 0.1
    assert rel_new.max() < tol, (name, rel_new)
    # Far out, where the cubic has diverged, the tail must win decisively.
    # Four-fold rather than ten, because for NFW both read the same
    # boundary value and that value is itself 4.6e-2 wrong, which caps how
    # far apart they can get.
    assert rel_new[-1] < 0.25 * rel_old[-1], (name, rel_new, rel_old)
    # And it must stay bounded where the cubic does not.
    assert rel_new[-1] < 3.0 * rel_new[0], (name, rel_new)
    assert rel_old[-1] > 10.0 * rel_old[0], (name, rel_old)


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

    The branch is opt-in (``cored_monopole=True``) and must stay off for
    spline-derived derivatives, where it latches onto a bracket endpoint and
    makes refinement *worse* -- see `_inner_monopole_q`. So this checks both
    halves: opted in with exact derivatives it fires and wins five orders;
    opted in with spline derivatives the density-sign test still rejects it;
    and by default it is not reached at all.
    """
    phi_of_r = PROFILES["plummer"][0]
    r = jnp.geomspace(R_MIN, R_MAX, N_R)
    log_r = jnp.log(r)
    values = phi_of_r(r)[:, None]
    exact = (jax.vmap(jax.grad(phi_of_r))(r) * r)[:, None]

    rq = jnp.asarray([R_MIN * f for f in (0.5, 0.2, 0.1)])
    expect = phi_of_r(rq)

    def err(derivs, **kw):
        coefs = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0]), **kw)
        got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[:, 0]
        q = float(coefs[0, 3, 0]) if coefs.shape[1] == 4 else 0.0
        return q, np.abs(np.asarray(got / expect - 1.0)).max()

    spline = fit_log_spline(log_r, values)
    q_exact, err_exact = err(exact, cored_monopole=True)
    q_spline, err_spline = err(spline, cored_monopole=True)

    assert q_exact > 0.0  # positive: a positive central density
    assert err_exact < 1e-9
    assert q_spline == 0.0  # rejected, three-parameter form instead
    assert err_spline < 1e-3

    # Off by default, so neither input can reach the branch at all.
    assert err(exact)[0] == 0.0
    assert err(spline)[0] == 0.0


@pytest.mark.parametrize("p", [-0.9, -0.5, -0.2])
def test_monopole_clip_is_skipped_when_there_is_no_monopole(p: float) -> None:
    """The cross-l clip needs a monopole to clip against.

    Every `lm_keys` set contains ``(0, 0)``, but `asymptotic_coeffs` is public
    and takes ``l_per_mode`` directly, so a caller can hand it a subset that
    does not. The masked mean then has nothing to average: ``n_mono`` falls
    back to 1 and the sum is empty, so ``s0`` is a fictitious 0 rather than
    the monopole's slope.

    Outward that is invisible -- ``_S_OUTER`` already caps the slope at 0 --
    but inward ``_S_INNER`` reaches to -1, so a genuinely negative slope was
    clipped away. These columns are exactly ``r**p``, so the fit must recover
    ``p``; before the guard it returned 0.
    """
    log_r = jnp.log(jnp.geomspace(1e-2, 300.0, 64))
    col = jnp.exp(p * (log_r - log_r[0]))
    values = jnp.stack([col, col], axis=-1)
    derivs = jnp.stack([p * col, p * col], axis=-1)

    s = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([2.0, 4.0]))[0, 1]
    assert np.allclose(np.asarray(s), p)


def test_monopole_clip_still_applies_inward_when_a_monopole_is_present() -> None:
    """The no-monopole guard must not disable the inward clip for sets that have one."""
    log_r = jnp.log(jnp.geomspace(1e-2, 300.0, 64))
    # A flat monopole and a steeper l=2: the l=2 inner slope must not fall below it.
    mono = jnp.ones_like(log_r)
    steep = jnp.exp(3.0 * (log_r - log_r[0]))
    values = jnp.stack([mono, steep], axis=-1)
    derivs = jnp.stack([jnp.zeros_like(log_r), 3.0 * steep], axis=-1)

    coefs = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0, 2.0]))
    s_mono, s_l2 = coefs[0, 1]
    assert s_l2 >= s_mono - 1e-12, "the inward clip must still bind"


def test_outward_slope_is_not_clipped_against_the_monopole() -> None:
    """Outward, a mode's fit must not depend on what else is in the set.

    The cross-l clip is inward-only. `_S_OUTER` caps `s` at 0, so outward no
    mode can diverge and there is nothing to guard against; and `s0` is the
    slope of the monopole's *correction* term, not of its leading `x^-1`, so
    clipping to it has no physical content. Measured on a q = 0.6 flattened
    NFW against a 20x wider grid, it cost a factor of 3-5 on the modes it
    bound -- (2,0) 0.855 -> 0.271 and (4,0) 0.941 -> 0.192 relative error --
    because what the fitted `s` reports outward is that the mass does not
    stop where the grid does.
    """
    log_r = jnp.log(jnp.geomspace(1e-2, 300.0, 96))
    x = jnp.exp(log_r - log_r[-1])
    mono = -1.0 / (1.0 + x)
    l2 = 1e-3 * x ** (-1.3)  # decays far slower than the l=2 solid harmonic
    values = jnp.stack([mono, l2], axis=-1)
    derivs = fit_log_spline(log_r, values)

    together = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0, 2.0]))
    alone = asymptotic_coeffs(log_r, values[:, 1:], derivs[:, 1:], jnp.asarray([2.0]))
    assert jnp.isclose(
        together[1, 1, 1], alone[1, 1, 0], rtol=1e-12
    ), "the outward fit must be independent of the monopole"


def _plummer_modes(*, cored_monopole: bool = False):
    """Build a monopole + l=2 pair on a log grid, with tail coefficients."""
    log_r = jnp.log(jnp.geomspace(0.05, 20.0, 64))
    r = jnp.exp(log_r)
    values = jnp.stack([-1.0 / jnp.sqrt(1 + r**2), 0.1 * r**2 / (1 + r**2) ** 2.5], -1)
    derivs = fit_log_spline(log_r, values)
    coefs = asymptotic_coeffs(
        log_r, values, derivs, jnp.asarray([0.0, 2.0]), cored_monopole=cored_monopole
    )
    return log_r, values, derivs, coefs


def test_tail_is_finite_at_the_origin() -> None:
    r"""``r = 0`` arrives as ``log_rq = -inf`` and must give the tail's limit.

    REGRESSION: clamping :math:`L` by sign alone left the monopole's
    :math:`v = 0` computing ``exp(0 * -inf)``, so :math:`\Phi` and its
    gradient both came back `nan` -- at a point an orbit integrator can
    actually reach. The limit is finite and analytic,
    :math:`W = P_1 - B/s - Q`.
    """
    log_r, values, derivs, coefs = _plummer_modes()
    got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.asarray([-jnp.inf]))[
        0
    ]
    assert jnp.all(jnp.isfinite(got))

    _v, s, B = coefs[0][:3, 0]
    assert jnp.isclose(got[0], values[0, 0] - B / s, rtol=1e-12)

    grad = jax.grad(
        lambda lq: eval_log_spline_asympt(log_r, values, derivs, coefs, lq).sum()
    )(jnp.asarray([-jnp.inf]))
    assert jnp.all(jnp.isfinite(grad)), "a nan here poisons a whole vmapped batch"


@pytest.mark.parametrize("cored_monopole", [False, True])
def test_tail_is_finite_arbitrarily_far_outside_the_grid(cored_monopole) -> None:
    """The outward ``Q (x^2 - x^v)`` term must not overflow.

    REGRESSION: ``Q`` is structurally zero outward, but the ``x^2`` beside it
    was still evaluated, so ``0 * inf`` gave `nan` past ``r/r_max ~ 1e154``.

    Parametrized because the default build now carries no ``Q`` row at all,
    so it cannot reach the code this is meant to guard -- only the
    ``cored_monopole=True`` build takes ``q_term=True`` inward.
    """
    log_r, values, derivs, coefs = _plummer_modes(cored_monopole=cored_monopole)
    assert coefs.shape[1] == (4 if cored_monopole else 3)

    far = log_r[-1] + jnp.log(jnp.asarray([1e10, 1e100, 1e200]))
    got = eval_log_spline_asympt(log_r, values, derivs, coefs, far)
    assert jnp.all(jnp.isfinite(got))
    for edge in (jnp.inf, -jnp.inf):
        assert jnp.all(
            jnp.isfinite(
                eval_log_spline_asympt(
                    log_r, values, derivs, coefs, jnp.asarray([edge])
                )
            )
        )


def test_refining_the_grid_never_makes_the_inner_tail_worse() -> None:
    """Refinement must improve the answer, monotonically.

    REGRESSION: the four-parameter inward fit used to be taken by default.
    Its acceptance gate compares an O(h^4) residual against an O(1) scale, so
    past a certain resolution the whole bracket passed and bisection latched
    onto an endpoint -- ``s = 8`` (no root) or the spurious exact root
    ``s = 2``. Measured on this exact configuration, the relative error at
    ``r_min / 2`` went 1.1e-5, 1.1e-5, **1.0e-3**, 3.0e-4 across
    n_r = 1024...8192: a 92x degradation from one refinement, and not even
    monotone in the wrong direction. It is opt-in now (`cored_monopole`).
    """
    phi_of_r = PROFILES["nfw"][0]
    errs = []
    for n_r in (1024, 2048, 4096, 8192):
        r = jnp.geomspace(R_MIN, R_MAX, n_r)
        log_r = jnp.log(r)
        values = phi_of_r(r)[:, None]
        derivs = fit_log_spline(log_r, values)
        coefs = asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0]))

        assert coefs.shape[1] == 3, "the Q branch must not be reached"

        rq = jnp.asarray([R_MIN * 0.5])
        got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.log(rq))[0, 0]
        errs.append(abs(float(got / phi_of_r(rq)[0] - 1.0)))

    assert all(b <= a for a, b in itertools.pairwise(errs)), errs
    assert errs[-1] < 2e-5, errs


def test_the_fitted_slope_is_differentiable_through_the_root_find() -> None:
    """`_polish` exists so the bisected ``s`` carries a derivative.

    Bisection is a `fori_loop` over `jnp.where`, whose output has zero
    gradient with respect to the data -- the root is selected, never
    computed. `_polish` adds ``(r - stop_gradient(r)) / stop_gradient(dr)``,
    which is identically 0 in value and carries the implicit-function
    derivative ``-(dR/dtheta)/(dR/ds)`` in its tangent.

    It therefore reads as ``s - 0`` and is exactly the kind of line that gets
    "simplified" away, so this pins both halves: that the gradient exists,
    and that it is the *right* gradient.

    The perturbed parameter has to be one the root actually depends on. An
    overall amplitude will not do: the residual is homogeneous of degree one
    in it, so ``s`` is amplitude-invariant (it agrees to 11 digits either
    side of a 1e-6 step) and both autodiff and finite differences return
    round-off. Perturbing the Plummer scale instead moves the root properly.
    """
    log_r = jnp.log(jnp.geomspace(0.05, 20.0, 64))
    r = jnp.exp(log_r)

    def fitted_s(a):
        values = (-1.0 / jnp.sqrt(a + r**2))[:, None]
        derivs = fit_log_spline(log_r, values)
        return asymptotic_coeffs(log_r, values, derivs, jnp.asarray([0.0]))[0, 1, 0]

    one = jnp.asarray(1.0)
    auto = float(jax.grad(fitted_s)(one))
    h = 1e-6
    fd = float((fitted_s(one + h) - fitted_s(one - h)) / (2 * h))

    # ~7.0e-3; the point is that it is a real derivative, not round-off.
    assert abs(fd) > 1e-4, f"the probe must actually move the root, got {fd}"
    assert auto != 0.0, "the root find must not be gradient-blind"
    assert abs(auto - fd) < 1e-3 * abs(fd), (auto, fd)
