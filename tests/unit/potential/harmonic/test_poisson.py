"""Tests for the radial Poisson solve."""

import pytest

import quaxed.numpy as jnp

from galax.potential._src.harmonic.poisson import solve_poisson_lm


@pytest.mark.parametrize("l", [1, 2, 3, 4, 6])
def test_solve_poisson_lm_matches_the_analytic_power_law(l: int) -> None:
    r"""The Green's-function solve against a closed form, for l > 0.

    For :math:`\rho_l(r) = r^p` both radial integrals are elementary, and

    .. math::

        \Phi_l(r) = \frac{4 \pi G r^{p+2}}{(p + l + 3)(p + 2 - l)}

    whenever :math:`p + l + 3 > 0` (the inner integral converges at 0) and
    :math:`p + 2 - l < 0` (the outer one converges at infinity). This pins
    the :math:`l`-dependence, the :math:`4 \pi G / (2l + 1)` prefactor and
    both boundary tails at once: a power law is exactly what the three-point
    tail fit is meant to reproduce, so any error here is the interior
    quadrature, which `test_..._converges_at_second_order` pins separately.
    """
    p, G = -1.5, 1.3
    assert p + l + 3 > 0  # the inner integral converges at the origin
    assert p + 2 - l < 0  # the outer one converges at infinity

    r = jnp.exp(jnp.linspace(jnp.log(1e-3), jnp.log(1e3), 1024))
    rho_lm = (r**p)[:, None]
    got = solve_poisson_lm(r, rho_lm, jnp.asarray([float(l)]), jnp.asarray(G))[:, 0]
    expect = 4.0 * jnp.pi * G * r ** (p + 2) / ((p + l + 3) * (p + 2 - l))

    interior = slice(4, -4)
    assert jnp.allclose(got[interior], expect[interior], rtol=1e-3)


def test_solve_poisson_lm_converges_at_second_order() -> None:
    """The interior quadrature is the trapezoid rule, so error ~ h^2.

    Measured against the closed form above: halving the step must quarter the
    error. A first-order slip (a mis-centred weight, an off-by-one in the
    cumulative sums) would show up here as a ratio near 2, while still
    passing a loose fixed tolerance.
    """
    p, G, l = -1.5, 1.0, 2.0
    errs = []
    for n in (256, 512, 1024):
        r = jnp.exp(jnp.linspace(jnp.log(1e-3), jnp.log(1e3), n))
        got = solve_poisson_lm(r, (r**p)[:, None], jnp.asarray([l]), jnp.asarray(G))
        expect = 4.0 * jnp.pi * G * r ** (p + 2) / ((p + l + 3) * (p + 2 - l))
        interior = slice(4, -4)
        errs.append(float(jnp.max(jnp.abs(got[interior, 0] / expect[interior] - 1.0))))

    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(3.5 < q < 4.5 for q in ratios), f"not second order: {ratios}"


def test_solve_poisson_lm_monopole_is_the_hernquist_potential() -> None:
    """Independent physics check, not a self-consistency check.

    Hernquist with M=1, a=1 has rho = 1/(2 pi) / (r (1+r)^3) and the closed
    form Phi = -1/(1+r). The monopole solve must reproduce it, since
    Phi = Phi_00 Y_00 = Phi_00 / sqrt(4 pi).
    """
    r = jnp.exp(jnp.linspace(jnp.log(1e-3), jnp.log(1e3), 512))
    rho = 1.0 / (2.0 * jnp.pi) / (r * (1.0 + r) ** 3)
    rho_00 = jnp.sqrt(4.0 * jnp.pi) * rho

    phi_00 = solve_poisson_lm(r, rho_00[:, None], jnp.asarray([0.0]), jnp.asarray(1.0))[
        :, 0
    ]
    got = phi_00 / jnp.sqrt(4.0 * jnp.pi)
    expect = -1.0 / (1.0 + r)

    # Interior only: the outermost knots carry the truncated-tail error.
    interior = slice(8, -8)
    assert jnp.allclose(got[interior], expect[interior], rtol=5e-4)


def test_solve_poisson_lm_scales_linearly_in_G() -> None:
    r = jnp.exp(jnp.linspace(jnp.log(1e-2), jnp.log(1e2), 64))
    rho_lm = (1.0 / (r * (1.0 + r) ** 2))[:, None]
    a = solve_poisson_lm(r, rho_lm, jnp.asarray([0.0]), jnp.asarray(1.0))
    b = solve_poisson_lm(r, rho_lm, jnp.asarray([0.0]), jnp.asarray(2.5))
    assert jnp.allclose(b, 2.5 * a, rtol=1e-14)


def test_outer_tail_applies_to_negative_modes() -> None:
    """The outer tail must be sign-agnostic.

    Gating on `rho_lm[-1] > 0.0` rather than `!= 0.0` would silently drop the
    correction for negative modes -- routine for l >= 1 -- since the sign says
    nothing about whether the tail converges. A mode and its negation must
    give exactly opposite Phi_lm, because the solve is linear in rho_lm, and
    that is what this checks.
    """
    r = jnp.exp(jnp.linspace(jnp.log(1e-2), jnp.log(3e2), 128))
    rho = 1.0 / (r * (1.0 + r) ** 2)
    l = jnp.asarray([2.0])

    pos = solve_poisson_lm(r, rho[:, None], l, jnp.asarray(1.0))
    neg = solve_poisson_lm(r, -rho[:, None], l, jnp.asarray(1.0))

    assert jnp.allclose(neg, -pos, rtol=1e-14)


def test_inner_tail_is_dropped_inside_the_clamped_slope_window() -> None:
    """A clamped denominator must zero the tail, not scale it arbitrarily.

    For l = 0 the inner tail carries a 1/(alpha_in + 3) factor. At
    alpha_in = -3 + 1e-9 that denominator is clamped to `_SLOPE_TOL` = 1e-6,
    so the tail came out ~1e3 times too small -- silently wrong rather than
    conservatively zero. It is now dropped, matching the treatment just
    across the convergence boundary at alpha_in <= -3.
    """
    r = jnp.exp(jnp.linspace(jnp.log(1e-2), jnp.log(3e2), 128))
    l = jnp.asarray([0.0])

    # Inside the clamped window: exp_in = alpha_in + 3 = 1e-9.
    inside = solve_poisson_lm(r, (r**-2.999999999)[:, None], l, jnp.asarray(1.0))
    # Just across the convergence boundary, where the tail is already zero.
    across = solve_poisson_lm(r, (r**-3.000000001)[:, None], l, jnp.asarray(1.0))

    # Continuous across the boundary: both drop the tail, so Phi agrees to
    # the difference between the two densities themselves.
    assert jnp.allclose(inside, across, rtol=1e-6)


def test_outer_tail_is_load_bearing() -> None:
    """Guard against a gate that disables the tail for every mode.

    Reusing the inner gate's `1e-8 * max|rho_lm|` here would zero every tail:
    that scale is the per-mode maximum, set by the inner cusp, and is ~1e9
    larger than rho_lm(r_max).

    Physically, for l = 0,
    ``Phi(r) = -4 pi G [ M(<r)/(4 pi r) + int_r^inf rho r' dr' ]``
    and the exterior term is strictly negative, so |Phi(r_max)| must exceed
    the enclosed-mass-only value by a clear margin. Measured at ~20% for NFW.
    """
    r = jnp.exp(jnp.linspace(jnp.log(1e-2), jnp.log(3e2), 128))
    rho = 1.0 / (r * (1.0 + r) ** 2)
    rho_00 = jnp.sqrt(4.0 * jnp.pi) * rho

    phi_00 = solve_poisson_lm(r, rho_00[:, None], jnp.asarray([0.0]), jnp.asarray(1.0))[
        :, 0
    ]
    phi = phi_00 / jnp.sqrt(4.0 * jnp.pi)

    m_enclosed = (
        4.0
        * jnp.pi
        * jnp.concatenate(
            [
                jnp.zeros(1),
                jnp.cumsum(
                    0.5 * (rho[:-1] * r[:-1] ** 2 + rho[1:] * r[1:] ** 2) * jnp.diff(r)
                ),
            ]
        )
    )
    enclosed_only = -m_enclosed / r

    assert abs(float(phi[-1])) > 1.1 * abs(float(enclosed_only[-1]))


@pytest.mark.parametrize(("lo", "hi"), [(1e17, 1e21), (1e-21, 1e-17)])
def test_solve_poisson_lm_survives_a_large_radius_magnitude(lo, hi) -> None:
    """The solve must not care where the grid sits, only how wide it is.

    REGRESSION: ``f_in`` carried ``exp((l+2) log r)`` and ``phi_col`` divided
    it back out, so the intermediate overflowed for large ``(l+2)|log r|``
    even though the ratio is ordinary -- and ``inf * 0`` is `nan`. At l=16
    over [1e19, 1e23] every one of 1024 knots came back non-finite. A length
    unit with a large numeric magnitude (metres: |log r| ~ 50) reaches this
    at l_max ~ 12; kpc does not, which is why nothing caught it.
    """
    p, l, G = -1.5, 16.0, 1.0
    r = jnp.exp(jnp.linspace(jnp.log(lo), jnp.log(hi), 1024))
    got = solve_poisson_lm(r, (r**p)[:, None], jnp.asarray([l]), jnp.asarray(G))[:, 0]
    expect = 4.0 * jnp.pi * G * r ** (p + 2) / ((p + l + 3) * (p + 2 - l))

    assert jnp.all(jnp.isfinite(got))
    interior = slice(4, -4)
    assert jnp.allclose(got[interior], expect[interior], rtol=1e-2)


def test_solve_poisson_lm_is_equivariant_under_rescaling_r() -> None:
    """Phi(e^c r) must equal e^(2c) Phi(r) at fixed rho.

    That is the exact property the centred grid relies on, so it is pinned
    rather than assumed.
    """
    p, l, G = -1.5, 3.0, 1.0
    r = jnp.exp(jnp.linspace(jnp.log(1e-2), jnp.log(1e2), 512))
    rho = (r**p)[:, None]
    base = solve_poisson_lm(r, rho, jnp.asarray([l]), jnp.asarray(G))

    c = jnp.log(1e6)
    scaled = solve_poisson_lm(r * jnp.exp(c), rho, jnp.asarray([l]), jnp.asarray(G))
    assert jnp.allclose(scaled, base * jnp.exp(2.0 * c), rtol=1e-12)


def test_inner_tail_survives_a_huge_fitted_slope() -> None:
    """A round-off mode must not take the whole build down.

    REGRESSION: the inner tail was formed as ``A_in * r_min**exp_in`` with
    ``A_in = rho_0 * r_min**-alpha_in``, which builds
    ``exp(-alpha_in * log r_min)`` as an intermediate. ``alpha_in`` is a
    three-point slope, and on a mode that is pure round-off it is routinely
    in the hundreds -- measured +420 to +459 on ordinary flattened-NFW builds
    (l_max = 8, r in [1, 50] kpc, n_r = 1024, axis ratios 0.9 and 0.6) -- so
    with the grid centred, ``-alpha_in * log r_min`` exceeded 709 and that
    intermediate overflowed to ``inf``. ``inf * exp(-large)`` is ``nan``,
    both boundary gates pass, and `fit_log_spline` then failed on the whole
    matrix: one negligible mode killed every mode.

    The sign matters: a steeply *falling* first three points give a large
    negative slope, which underflows to zero harmlessly. It is the *rising*
    case that overflows, which is what a round-off mode produces.
    """
    r = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(50.0), 1024))
    # Rising across the first three knots -> large positive fitted slope.
    rho = jnp.full_like(r, 1e-18).at[0].set(1e-24).at[1].set(1e-22).at[2].set(1e-20)

    log_r = jnp.log(r)
    alpha = float(jnp.mean(jnp.diff(jnp.log(jnp.abs(rho[:3]))) / jnp.diff(log_r[:3])))
    # The old form overflowed once `-alpha * (log r_min - log r_mid)` > 709.
    half_range = 0.5 * float(log_r[-1] - log_r[0])
    assert alpha * half_range > 709.0, f"slope {alpha} is not extreme enough to bite"

    got = solve_poisson_lm(r, rho[:, None], jnp.asarray([8.0]), jnp.asarray(1.0))
    assert jnp.all(jnp.isfinite(got)), "one round-off mode must not poison the column"
