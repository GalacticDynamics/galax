"""Tests for the radial Poisson solve."""

import jax
import pytest

import quaxed.numpy as jnp

from galax.potential._src.harmonic.poisson import (
    _active_tol,
    _log_floor,
    gl_log_nodes,
    solve_poisson_lm,
    solve_poisson_profiles,
)


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
    quadrature, which `test_..._converges_at_fourth_order` pins separately.
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


def test_solve_poisson_lm_converges_at_fourth_order() -> None:
    """The interior quadrature is cubic-Hermite in log r, so error ~ h^4.

    Measured against the closed form above: halving the step must cut the
    error by sixteen. The band is wide enough to survive round-off at the
    finest step but far from the trapezoid's ratio of 4, so dropping the
    endpoint-slope correction -- or applying it with the wrong sign or
    spacing -- fails here while still passing a loose fixed tolerance.
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
    assert all(12.0 < q < 20.0 for q in ratios), f"not fourth order: {ratios}"


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


@pytest.mark.parametrize("n_r", [1, 2])
def test_too_few_radial_knots_is_rejected(n_r: int) -> None:
    """A grid too short for the boundary slope fits must fail loudly.

    It does not fail on its own. JAX's static slicing clamps rather than
    raising, so ``rho_col[:3]`` on a 2-knot grid silently degrades to a
    2-point slope fit, and a 1-knot grid returns ``-0.0`` -- a meaningless
    answer that no exception marks as one.
    """
    r = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10.0), n_r))
    rho = jnp.ones((n_r, 1))

    with pytest.raises(ValueError, match="at least 3 radial knots"):
        solve_poisson_lm(r, rho, jnp.asarray([0.0]), jnp.asarray(1.0))


def test_interior_quadrature_is_accurate_in_float32() -> None:
    """The working dtype users actually get, not the one the suite forces.

    `galax` defaults to float32, while ``pyproject.toml`` turns on x64 for
    tests -- so every other check here runs in a precision the library does
    not use by default. Float32 *inputs* are not enough to test that:
    internal literals promote back to x64, and the solve silently returns
    float64. `jax.enable_x64(False)` is what actually pins the arithmetic,
    and the dtype assertion below is what keeps this test honest.

    A wide bracket holds the boundary tails well under the interior error, so
    this measures the quadrature. The trapezoid this replaced gives 1.1e-3 on
    the same grid, three orders above the bound here, while the Hermite rule
    sits on its float32 round-off floor of a few times ``eps``.
    """
    with jax.enable_x64(False):  # noqa: FBT003
        r = jnp.geomspace(1e-4, 1e4, 256).astype(jnp.float32)
        rho = (1.0 / (2.0 * jnp.pi) / (r * (1.0 + r) ** 3)).astype(jnp.float32)
        got = solve_poisson_lm(r, rho[:, None], jnp.asarray([0.0]), jnp.asarray(1.0))[
            :, 0
        ]

        assert got.dtype == jnp.float32  # no silent promotion to x64
        expect = -1.0 / (1.0 + r)
        interior = (r > 1e-2) & (r < 1e2)
        err = jnp.max(
            jnp.abs(got[interior] - expect[interior]) / jnp.abs(expect[interior])
        )

    assert float(err) < 1e-6, f"float32 interior error {float(err):.2e}"


def test_zero_mode_gradient_is_finite_in_float32() -> None:
    """An identically-zero mode must not poison the gradient in float32.

    REGRESSION: the floor inside ``log|rho|`` was ``1e-300``, a float64
    constant that underflows to *exactly zero* in float32 -- which is what a
    caller gets, since `galax` does not enable x64 on import. The floor then
    floored nothing, ``log(0)`` gave ``-inf``, and the value survived only
    because the gates masked it while the gradient came back ``nan``. Six of
    them, on one zero column. Zero columns are routine for ``l >= 1``.
    """
    r = jnp.geomspace(1e-2, 1e2, 64).astype(jnp.float32)
    live = (1.0 / (r * (1.0 + r) ** 3)).astype(jnp.float32)
    rho = jnp.stack([live, jnp.zeros_like(live)], axis=-1)
    l_per_mode = jnp.asarray([0.0, 2.0], dtype=jnp.float32)

    def loss(rho_lm):
        return jnp.sum(
            solve_poisson_lm(r, rho_lm, l_per_mode, jnp.asarray(1.0, dtype=jnp.float32))
            ** 2
        )

    grad = jax.grad(loss)(rho)

    assert grad.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(grad)), f"{int(jnp.sum(jnp.isnan(grad)))} nan in grad"


def test_thresholds_follow_the_working_dtype() -> None:
    """Both float64 constants underflow or vanish at float32 precision."""
    f64 = jnp.zeros((), dtype=jnp.float64)
    f32 = jnp.zeros((), dtype=jnp.float32)

    # The floor must stay a normal number in both, not underflow to zero.
    assert _log_floor(f32) > 0.0
    assert _log_floor(f64) > 0.0
    assert _log_floor(f32) > _log_floor(f64)

    # The relative gate must stay above round-off, not sink beneath it.
    assert _active_tol(f32) > float(jnp.finfo(jnp.float32).eps)
    assert _active_tol(f64) > float(jnp.finfo(jnp.float64).eps)


def test_gauss_legendre_sampling_beats_the_knot_only_rule() -> None:
    """Sampling rho between knots removes the interpolation error entirely.

    On knots alone the solve can only be as good as the rule that
    reconstructs rho across each interval -- fourth order, and that is an
    information limit, not an implementation one. Given the density *at*
    Gauss-Legendre nodes it is doing quadrature instead, and for a smooth
    profile a four-point rule is exact to round-off.

    Both calls use the same knots, the same tails and the same assembly, so
    the gap below is the interior rule and nothing else.
    """
    r = jnp.geomspace(1e-4, 1e4, 128)
    rho = 1.0 / (2.0 * jnp.pi) / (r * (1.0 + r) ** 3)
    l_per_mode, G = jnp.asarray([0.0]), jnp.asarray(1.0)

    knots_only = solve_poisson_profiles(r, rho[:, None], l_per_mode, G)[0][:, 0]

    log_gl, _ = gl_log_nodes(jnp.log(r), 4)
    r_gl = jnp.exp(log_gl)
    rho_gl = (1.0 / (2.0 * jnp.pi) / (r_gl * (1.0 + r_gl) ** 3))[:, :, None]
    sampled = solve_poisson_profiles(r, rho[:, None], l_per_mode, G, rho_gl)[0][:, 0]

    expect = -1.0 / (1.0 + r)
    interior = (r > 1e-2) & (r < 1e2)

    def rel(got):
        return float(
            jnp.max(
                jnp.abs(got[interior] - expect[interior]) / jnp.abs(expect[interior])
            )
        )

    e_knots, e_gl = rel(knots_only), rel(sampled)
    assert e_knots < 1e-5, e_knots  # the Hermite rule, for orientation
    assert e_gl < 1e-9, e_gl
    assert e_gl < e_knots / 1000.0, f"knots {e_knots:.2e} -> gl {e_gl:.2e}"


def test_gl_log_nodes_integrates_a_known_function_exactly() -> None:
    """The nodes and weights are a quadrature rule for d(log r), not for dr.

    A Jacobian slip here would be nearly invisible in the solve -- it would
    look like a slightly wrong density -- so it is pinned directly.

    The integrand is a degree-7 polynomial in ``u = log r``, which a
    four-point Gauss-Legendre rule integrates *exactly*, so this asserts at
    round-off rather than against a tolerance. (``exp(3u)`` would not do:
    Gauss-Legendre is spectrally accurate on it, not exact.)
    """
    log_r = jnp.log(jnp.geomspace(0.1, 10.0, 17))
    nodes, weights = gl_log_nodes(log_r, 4)
    assert nodes.shape == weights.shape == (16, 4)

    a, b = log_r[0], log_r[-1]
    got = jnp.sum(weights * nodes**7)
    want = (b**8 - a**8) / 8.0
    assert jnp.allclose(got, want, rtol=1e-13), (got, want)


@pytest.mark.parametrize(
    ("mangle", "label"),
    [
        (lambda g: g[:1], "interval axis collapsed to 1"),
        (lambda g: g[:-1], "one interval short"),
        (lambda g: g[:, :, :0], "no modes"),
        (lambda g: g[:, :, 0], "2-D"),
        (lambda g: g.reshape(-1), "1-D"),
    ],
)
def test_solve_poisson_rejects_a_mismatched_gauss_legendre_array(mangle, label) -> None:
    """A wrong-shaped ``rho_gl`` must fail loudly, not broadcast.

    This is the dangerous shape error: with the interval axis collapsed to 1,
    ``rho_gl`` broadcasts against the ``(n_r - 1, k)`` weights instead of
    failing, and the solve returns *finite, plausible* numbers computed from
    the wrong integrals. Nothing downstream would notice.

    Shapes are static, so the check is made at trace time and costs nothing
    at runtime -- the same treatment as the ``n_r >= 3`` guard above.
    """
    r = jnp.geomspace(1e-3, 1e3, 64)
    rho = (1.0 / (2.0 * jnp.pi) / (r * (1.0 + r) ** 3))[:, None]
    log_gl, _ = gl_log_nodes(jnp.log(r), 4)
    r_gl = jnp.exp(log_gl)
    rho_gl = (1.0 / (2.0 * jnp.pi) / (r_gl * (1.0 + r_gl) ** 3))[:, :, None]

    args = (r, rho, jnp.asarray([0.0]), jnp.asarray(1.0))
    solve_poisson_profiles(*args, rho_gl)  # the correct shape is accepted

    with pytest.raises(ValueError, match="rho_gl must"):
        solve_poisson_profiles(*args, mangle(rho_gl))


def test_the_knot_only_path_does_not_allocate_a_matrix_per_mode() -> None:
    """The non-GL path must stay O(n_r), not O(n_r^2).

    REGRESSION: scaling each panel endpoint to the end of its own interval
    was written as an ``(n_r - 1, n_r)`` broadcast whose two diagonals were
    then extracted -- ``n`` values taken from an ``n**2`` array, per mode,
    per direction. At ``n_r = 2560`` that is 52 MB a time, and the solve took
    2.5 s where it now takes 47 ms.

    Nothing about the *answer* changed, which is why no accuracy test caught
    it, and `build_expansion` always supplies Gauss-Legendre samples so it
    never reached this branch at all. `solve_poisson_lm` is public and does.

    The compiled size is the durable check: an O(n^2) intermediate shows up
    as a quadratic jump in peak buffer size, while the O(n) form grows
    linearly. Compare the two grids rather than asserting an absolute, so
    this survives XLA changing its mind about fusion.
    """
    p, G = -1.5, 1.0

    def peak_bytes(n_r: int) -> int:
        r = jnp.geomspace(1e-3, 1e3, n_r)
        rho = (r**p)[:, None]
        compiled = (
            jax.jit(solve_poisson_profiles)
            .lower(r, rho, jnp.asarray([8.0]), jnp.asarray(G))
            .compile()
        )
        return compiled.memory_analysis().temp_size_in_bytes

    small, large = peak_bytes(256), peak_bytes(1024)
    # Four times the knots: linear would be ~4x, quadratic ~16x.
    assert large < 8 * small, (small, large)


def test_solve_poisson_rejects_an_empty_gauss_legendre_axis() -> None:
    """``k = 0`` must be named, not left to `leggauss`.

    A ``(n_r - 1, 0, n_modes)`` array has the right rank and the right first
    and last axes, so it passes the shape guard, then reaches `leggauss` and
    dies with "deg must be a positive integer" -- which names neither
    ``rho_gl`` nor the caller that supplied it.
    """
    r = jnp.geomspace(1e-3, 1e3, 64)
    rho = (r**-1.5)[:, None]
    args = (r, rho, jnp.asarray([2.0]), jnp.asarray(1.0))

    with pytest.raises(ValueError, match="at least one Gauss-Legendre node"):
        solve_poisson_profiles(*args, jnp.zeros((63, 0, 1)))


@pytest.mark.parametrize("l", [0.0, 2.0, 4.0, 8.0])
def test_the_inner_tail_is_exact_when_anchored_above_dropped_samples(l) -> None:
    r"""Anchoring above a zeroed band must still integrate ``[0, r_i0]`` exactly.

    A pure power law is the case where the tail model is the truth rather
    than an approximation: for :math:`\rho = A r^\alpha`,

    .. math::

        \Phi_l(r) = \frac{-4\pi G A r^{\alpha+2}}{2l+1}
                    \left(\frac{1}{\alpha+l+3} - \frac{1}{\alpha+2-l}\right)

    exactly, provided both integrals converge (:math:`\alpha > -l-3` and
    :math:`\alpha < l-2`). So zeroing the inner knots and anchoring the tail
    above them must reproduce the same answer as not zeroing at all -- the
    analytic tail covers exactly what was discarded.

    REGRESSION (the margin): `fit_log_spline` is a *global* tridiagonal
    solve, so a zero-to-real step perturbs the fitted derivative for several
    knots around it, decaying by roughly the solve's Green's function (0.27
    per knot). Panel ``i0`` reads that derivative, so anchoring right at the
    step integrates a corrupted slope: measured 6.4e-4 there against 5.0e-8
    eight knots above, on this very case. `_ANCHOR_MARGIN` is what moves the
    anchor clear of it, and without it this test fails by four orders.
    """
    alpha, amp, n_r, drop = -2.5, 1.0, 512, 64
    r = jnp.geomspace(1e-6, 1e6, n_r)
    rho = (amp * r**alpha)[:, None]
    l_arr = jnp.asarray([l])

    zeroed = jnp.where(jnp.arange(n_r)[:, None] < drop, 0.0, rho)
    phi, _, _ = solve_poisson_profiles(
        r, zeroed, l_arr, jnp.asarray(1.0), None, jnp.asarray([float(drop)])
    )

    # Compare well above the anchor, where the answer is the closed form.
    lo = drop + 24
    rr = r[lo:]
    want = (
        -4.0
        * jnp.pi
        * amp
        * rr ** (alpha + 2.0)
        / (2.0 * l + 1.0)
        * (1.0 / (alpha + l + 3.0) - 1.0 / (alpha + 2.0 - l))
    )
    got = phi[lo:, 0]
    err = float(jnp.max(jnp.abs(got - want)) / jnp.max(jnp.abs(want)))
    assert err < 1e-4, (l, err)
