"""Tests for the multipole profile build pipeline."""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

from galax.potential._src.base import default_constants
from galax.potential._src.builtin.multipole_profile import expansion_potential
from galax.potential._src.builtin.multipole_profile.build import (
    build_expansion,
    subtract_inner_cusp,
)
from galax.potential._src.harmonic import (
    default_angular_resolution,
    eval_log_spline,
    lm_keys,
)
from galax.potential._src.utils import safe_vector_norm

_G_GALACTIC = float(default_constants["G"].decompose(u.unitsystem("galactic")).value)


def _hernquist_density(xyz, t):
    """Hernquist with M = a = 1: Phi = -G/(1+r)."""
    r = jnp.sqrt(jnp.sum(xyz**2, axis=-1))
    return 1.0 / (2.0 * jnp.pi) / r / (1.0 + r) ** 3


def test_subtract_inner_cusp_removes_a_pure_power_law() -> None:
    """A pure power law leaves a numerically zero residual.

    The amplitude is `rho_lm` at the innermost knot, so the background is
    `amplitude * (r / r_knots[0]) ** alpha`.
    """
    r = jnp.geomspace(1e-3, 1e3, 128)
    rho_lm = (3.0 * r**-1.5)[:, None]
    residual, alpha, amplitude = subtract_inner_cusp(r, rho_lm)

    assert jnp.isclose(alpha[0], -1.5, rtol=1e-10)
    assert jnp.isclose(amplitude[0], 3.0 * r[0] ** -1.5, rtol=1e-12)
    assert jnp.max(jnp.abs(residual)) < 1e-9 * jnp.max(jnp.abs(rho_lm))


def test_subtract_inner_cusp_zeroes_negligible_modes() -> None:
    """Modes negligible at `r_min` get no background, per the 1e-6 gate."""
    r = jnp.geomspace(1e-2, 1e2, 64)
    big = 1.0 / r
    tiny = jnp.full_like(r, 1e-12) * big[0]
    rho_lm = jnp.stack([big, tiny], axis=-1)

    residual, alpha, amplitude = subtract_inner_cusp(r, rho_lm)
    assert amplitude[1] == 0.0
    assert alpha[1] == 0.0
    assert jnp.allclose(residual[:, 1], tiny, atol=0.0)


def test_subtract_inner_cusp_declines_a_sign_changing_mode() -> None:
    """A zero crossing in the fitting window must not produce a background.

    `alpha` is fitted to log|rho_lm| over the innermost three knots, so a sign
    change there reads as a large positive slope and the background diverges
    outward. With the crossing adjacent to knot 0 the old fit gives
    alpha = +69.9 and an overflowing background, wrecking `residual +
    background` by catastrophic cancellation.

    The magnitude gate on |rho_lm[0]| cannot see this: rho_lm[0] here is
    1.1e-5 of the global scale, comfortably above the 1e-6 threshold. Nor is
    the alpha clip sufficient on its own -- clipped to +3 the background still
    reaches ~1e8 times the mode. Only the sign check rejects it.
    """
    r = jnp.geomspace(1e-2, 1e2, 64)
    clean = 1.0 / r
    # Sign flips between knots 0 and 1, with |rho_lm[0]| small but above the
    # magnitude gate -- the configuration that makes `alpha` blow up.
    crossing = clean.at[0].set(-1e-5 * clean[0])
    rho_lm = jnp.stack([clean, crossing], axis=-1)

    assert jnp.abs(crossing[0]) > 1e-6 * jnp.max(jnp.abs(rho_lm))

    residual, alpha, amplitude = subtract_inner_cusp(r, rho_lm)

    assert amplitude[1] == 0.0
    assert alpha[1] == 0.0
    # No background, so the residual carries the mode exactly.
    assert jnp.allclose(residual[:, 1], crossing, atol=0.0)
    # The well-behaved neighbour still gets its cusp subtracted.
    assert jnp.isclose(alpha[0], -1.0, rtol=1e-10)
    assert jnp.max(jnp.abs(residual[:, 0])) < 1e-9 * jnp.max(jnp.abs(clean))


def test_subtract_inner_cusp_clips_an_extreme_fitted_slope() -> None:
    """Even a same-sign fit is clipped to a physically plausible slope."""
    r = jnp.geomspace(1e-2, 1e2, 64)
    rho_lm = (r**-8.0)[:, None]
    _, alpha, _ = subtract_inner_cusp(r, rho_lm)
    assert alpha[0] == -3.0


def test_build_expansion_reconstructs_a_cuspy_density() -> None:
    """rho_lm from residual + background round-trips the projected rho_lm.

    NFW's r^-1 cusp is exactly the case direct splining handles badly, which
    is why the background is subtracted before fitting.
    """

    def rho(xyz, t):
        r = jnp.linalg.norm(xyz, axis=-1)
        return 1.0 / (r * (1.0 + r) ** 2)

    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)
    r = jnp.geomspace(1e-2, 3e2, 128)
    coeffs = build_expansion(
        rho, r, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )

    log_r = jnp.log(r)
    residual = eval_log_spline(
        log_r, coeffs["rho_residual_lm"], coeffs["drho_residual_lm"], log_r
    )
    background = coeffs["rho_amplitude"] * (r[:, None] / r[0]) ** coeffs["rho_alpha"]
    got = residual + background

    xyz = jnp.stack([r, jnp.zeros_like(r), jnp.zeros_like(r)], axis=-1)
    expect = jnp.sqrt(4.0 * jnp.pi) * rho(xyz, jnp.asarray(0.0))
    assert jnp.allclose(got[:, 0], expect, rtol=1e-10)


def test_build_expansion_returns_consistent_shapes() -> None:
    def rho(xyz, t):
        r = jnp.linalg.norm(xyz, axis=-1)
        return jnp.exp(-r)

    keys = lm_keys(4, "plane_reflection")
    n_theta, n_phi = default_angular_resolution(4)
    r = jnp.geomspace(1e-2, 1e2, 32)
    coeffs = build_expansion(
        rho, r, 4, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )

    n_modes = len(keys)
    expect = {
        "phi_lm": (32, n_modes),
        "dphi_lm": (32, n_modes),
        "d2phi_lm": (32, n_modes),
        "phi_asympt_powers": (2, 2, n_modes),
        "phi_asympt_scales": (2, 1, n_modes),
        "rho_residual_lm": (32, n_modes),
        "drho_residual_lm": (32, n_modes),
        "rho_alpha": (n_modes,),
        "rho_amplitude": (n_modes,),
    }
    # Compare the key *sets* first, so adding a return value without adding it
    # here fails loudly. `d2phi_lm` was added and this test kept passing on
    # the six keys it already knew about, which is how the gap arose.
    assert set(coeffs) == set(expect)
    assert {k: v.shape for k, v in coeffs.items()} == expect


def test_refining_n_r_actually_improves_the_potential() -> None:
    """``n_r`` must be an effective knob, not a knob against a floor.

    REGRESSION: the Poisson solve models the mass outside ``[r_min, r_max]``
    as one power law fitted at the boundary, and that model's error does not
    shrink with ``n_r``. Solving on the caller's grid therefore put a floor
    under everything -- 128 to 2048 knots moved the error only 1.2e-3 to
    6.8e-4, so the one knob a user has did almost nothing. `build_expansion`
    now solves on a padded grid and keeps the requested range, which puts the
    tail model below the interior quadrature.

    What is asserted is that refining still pays, deliberately *not* the
    convergence order: the order is the interior rule's, and it is pinned
    where that rule lives, in ``test_poisson.py``. Restating it here would
    make this test fail whenever the quadrature is improved, which is
    backwards -- the floor is this module's regression, the order is not.
    """
    errs = []
    for n_r in (128, 256, 512):
        r_knots = jnp.geomspace(0.05, 20.0, n_r)
        keys = lm_keys(0, "spherical")
        built = build_expansion(
            _hernquist_density,
            r_knots,
            0,
            keys,
            8,
            8,
            jnp.asarray(0.0),
            jnp.asarray(_G_GALACTIC),
        )
        built = {**built, "r_knots": r_knots}
        rq = jnp.geomspace(0.06, 18.0, 24)
        x = jnp.stack([rq, jnp.zeros_like(rq), jnp.zeros_like(rq)], -1)
        got = expansion_potential(built, x, 0, keys)
        want = -_G_GALACTIC / (1.0 + rq)
        errs.append(float(jnp.max(jnp.abs((got - want) / want))))

    assert errs[0] < 1e-3, errs
    # Each refinement must buy at least 3x. The pre-padding behaviour bought
    # ~1.2x over the same span, and a reinstated floor would show up as
    # ratios collapsing toward 1 well before any tolerance here is hit.
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(q > 3.0 for q in ratios), f"n_r is not paying: {ratios} from {errs}"


@pytest.mark.parametrize("n_r", [1, 2, 3])
def test_build_expansion_rejects_an_undersized_grid(n_r) -> None:
    """`build_expansion` is public, so it must guard its own contract.

    `MultipoleProfilePotential.from_density` enforces ``n_r >= 4``, but this
    is re-exported via `galax.potential.multipole_profile` and can be called
    directly. The check cannot be left to `solve_poisson_lm`'s ``n_r >= 3``
    either: padding runs first, and a one-knot grid pads to three and passes
    it. On top of that, ``r_knots[1]`` on a one-knot grid is out of bounds,
    where JAX clamps instead of raising -- so every padded knot would land on
    the original and the failure would surface far from its cause.
    """
    keys = lm_keys(0, "spherical")
    with pytest.raises(ValueError, match="at least 4 entries"):
        build_expansion(
            _hernquist_density,
            jnp.geomspace(0.05, 20.0, n_r),
            0,
            keys,
            2,
            2,
            jnp.asarray(0.0),
            jnp.asarray(1.0),
        )


def test_the_build_reaches_machine_precision_on_a_closed_form() -> None:
    """The whole pipeline, against a closed form, at the level agama reaches.

    This is the end-to-end guard on the three things that have to hold at
    once, since any one of them alone caps the result:

    - the radial quadrature samples the density *inside* each interval, so it
      is not limited by reconstructing rho between knots;
    - the profiles are interpolated with the quintic basis, using the exact
      derivatives the solve returns -- the cubic basis caps at ~5e-8 however
      good the integrals are;
    - the grid is padded far enough that the boundary power-law model is not
      the floor.

    Reference: ``agama`` reproduces this same closed form to 9.7e-16 inside
    the grid with 128 radial points (measured offline; agama is deliberately
    not a test dependency). At ``n_r=256`` this reaches 4.1e-15 and at 512
    8.7e-16, so the bound below is loose by an order and pins the regime
    rather than the digits.
    """
    keys = lm_keys(0, "spherical")
    r_knots = jnp.geomspace(0.05, 20.0, 256)
    p = {
        **build_expansion(
            _hernquist_density,
            r_knots,
            0,
            keys,
            8,
            8,
            jnp.asarray(0.0),
            jnp.asarray(_G_GALACTIC),
        ),
        "r_knots": r_knots,
    }
    rq = jnp.geomspace(0.06, 18.0, 64)
    x = jnp.stack([rq, jnp.zeros_like(rq), jnp.zeros_like(rq)], -1)
    got = expansion_potential(p, x, 0, keys)
    want = -_G_GALACTIC / (1.0 + rq)

    err = float(jnp.max(jnp.abs((got - want) / want)))
    assert err < 1e-13, f"max rel {err:.2e}"


def test_the_solve_returns_the_derivatives_it_claims() -> None:
    """``dphi_lm`` and ``d2phi_lm`` must be the derivatives, not a fit.

    They come out of the same two radial integrals as ``phi_lm``, which is
    what makes the quintic basis free. If either were quietly replaced by a
    spline fit of ``phi_lm`` the potential would still look right -- the
    quintic would just silently fall back to cubic accuracy -- so this
    compares against the closed form's own log-derivatives instead.

    Hernquist monopole: ``Phi = -G/(1+r)``, so in ``u = log r``,
    ``dPhi/du = G r/(1+r)^2`` and ``d2Phi/du2 = G r (1-r)/(1+r)^3``. The
    stored modes are ``Phi_lm``, which for ``l=0`` is ``Phi * sqrt(4 pi)``.
    """
    keys = lm_keys(0, "spherical")
    r = jnp.geomspace(0.05, 20.0, 256)
    p = build_expansion(
        _hernquist_density,
        r,
        0,
        keys,
        8,
        8,
        jnp.asarray(0.0),
        jnp.asarray(_G_GALACTIC),
    )

    y00 = 1.0 / jnp.sqrt(4.0 * jnp.pi)
    interior = slice(8, -8)
    for key, want in (
        ("dphi_lm", _G_GALACTIC * r / (1.0 + r) ** 2),
        ("d2phi_lm", _G_GALACTIC * r * (1.0 - r) / (1.0 + r) ** 3),
    ):
        got = p[key][:, 0] * y00
        scale = jnp.max(jnp.abs(want))
        err = float(jnp.max(jnp.abs(got[interior] - want[interior])) / scale)
        assert err < 1e-10, f"{key}: {err:.2e}"


@pytest.mark.parametrize("l_max", [4, 6, 8, 12])
def test_gradients_are_finite_in_float32_at_high_l(l_max: int) -> None:
    """Gradients, in the dtype `galax` actually runs, at the l a caller asks for.

    REGRESSION: the radial solve formed ``I_in = cumsum(rho x^(l+3))`` and
    only then divided by ``x^(l+1)``. On the padded grid at ``l=8`` that
    intermediate reaches ``e^104``, past float32's ``e^88.7``, while the
    quotient it feeds sits at ``e^-2.7`` -- the overflow was an artifact of
    the factorization, not of the answer.

    The forward pass hid it: the overflow landed in the padded region and was
    sliced off, so values stayed correct while ``cumsum``'s reverse rule
    turned the ``inf`` into ``nan`` cotangents that reached the interior.
    Gradients were `nan` from ``l_max = 5`` on an ordinary ``[0.05, 20]``
    grid, and the suite could not see it because ``pyproject`` forces x64.

    So this asserts on the *gradient*, not the value, and runs under
    `jax.enable_x64(False)`.
    """
    keys = lm_keys(l_max, "none")
    n_theta, n_phi = default_angular_resolution(l_max)

    def total(mass):
        def rho(xyz, t):
            r = jnp.linalg.norm(xyz, axis=-1)
            safe = jnp.where(r > 0, r, 1e-20)
            return mass / (2.0 * jnp.pi) / (safe * (1.0 + safe) ** 3)

        built = build_expansion(
            rho,
            jnp.geomspace(0.05, 20.0, 64),
            l_max,
            keys,
            n_theta,
            n_phi,
            jnp.asarray(0.0),
            jnp.asarray(1.0),
        )
        return jnp.sum(jnp.abs(built["phi_lm"]))

    with jax.enable_x64(False):  # noqa: FBT003
        value = total(2.0)
        grad = jax.grad(total)(2.0)

    assert jnp.isfinite(value), value
    assert jnp.isfinite(grad), grad


@pytest.mark.parametrize(("r_min", "r_max"), [(1e-3, 1e3), (1e-6, 1e6), (1e-10, 1e10)])
def test_a_very_wide_bracket_stays_finite_in_float32(r_min, r_max) -> None:
    """Padding must not push the grid out of the dtype's range.

    REGRESSION: the pad extended a fixed two spans either side, so a caller
    asking for ``[1e-6, 1e6]`` -- already 27.6 e-folds -- got a 138 e-fold
    padded grid, where float32's 87.4 budget is long gone and every output
    was `nan`.

    The reach is now capped by what the dtype can exponentiate, which
    degrades gracefully: a bracket this wide gets less tail accuracy rather
    than no answer at all. Ordinary brackets are nowhere near the cap and are
    unaffected, which the accuracy tests above pin.

    The density here is written to survive the padded range, which is the
    caller's job: padding evaluates ``rho_fn`` well outside the requested
    bracket, and the unguarded ``1/(2 pi) / r / (1 + r)**3`` used elsewhere in
    this file overflows float32 at ``r ~ 1e19`` on its own, before the solve
    sees it. That is a real trap but a different one, and pinning it here
    would test the density rather than the padding.
    """
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def rho(xyz, t):
        # The same Hernquist profile, formed in logs so it is finite at any
        # radius: `(1 + r) ** 3` alone overflows float32 by `r ~ 1e13`.
        r = jnp.sqrt(jnp.sum(xyz**2, axis=-1))
        safe = jnp.where(r > 0, r, jnp.finfo(r.dtype).tiny)
        return jnp.exp(-jnp.log(2.0 * jnp.pi) - jnp.log(safe) - 3.0 * jnp.log1p(safe))

    with jax.enable_x64(False):  # noqa: FBT003
        # Retrace under this config: `build_expansion` is jitted, and a cache
        # entry left by an earlier x64 test in this file is reused otherwise,
        # so the float32 path never actually runs and the test passes alone
        # while failing in the file.
        built = build_expansion(
            rho,
            jnp.geomspace(r_min, r_max, 64),
            0,
            keys,
            n_theta,
            n_phi,
            jnp.asarray(0.0),
            jnp.asarray(_G_GALACTIC),
        )

    for name, arr in built.items():
        assert jnp.all(jnp.isfinite(arr)), (name, arr)


@pytest.mark.parametrize(
    ("gamma", "mass", "rtol"),
    [
        (1.0, 1.0, 1e-5),
        (1.9, 1.0, 1e-5),
        (2.5, 1.0, 1e-5),
        (2.9, 1.0, 1e-4),
        # The same cusp, rescaled. A cap that bounds `rho_lm` against the
        # dtype passes at `mass = 1` by coincidence -- float32's min normal
        # and max are near-reciprocal, so `1/r**2.9` lands just inside the
        # budget -- and fails for any smaller prefactor, because what
        # overflows is `mass / r**2.9` once `r**2.9` underflows to zero.
        # These three were 0.111 under such a cap.
        (2.9, 1e-2, 1e-4),
        (2.9, 1e-4, 1e-4),
        (2.9, 1e-6, 1e-4),
    ],
)
def test_a_steep_cusp_survives_the_padded_sampling_in_float32(
    gamma, mass, rtol
) -> None:
    """A density the *pad* cannot represent must not zero the whole build.

    REGRESSION: padding evaluates ``rho_fn`` `_PAD_MULTIPLE` spans outside
    the requested bracket, so a cusp ``rho ~ r**-gamma`` is sampled at
    ``r_min * exp(-reach)`` where the density overflows once
    ``gamma * reach`` clears the dtype's exponent. Those samples are zeroed,
    and the radial solve anchors its inner tail above them, so the band they
    should have carried is covered analytically instead of lost.

    One unrepresentable sample took out everything: the projection turns
    ``inf`` into ``nan`` for ``l >= 1``, and `fit_log_spline` solves one
    system per mode over the whole padded grid, so a single bad row reached
    every radius and every mode. Measured at ``gamma = 2.5``, ``l_max = 8``
    over ``[1e-4, 1e4]``: **41472 non-finite out of 41472** in float32, none
    in float64.

    ``gamma`` up to 3 is physical (finite mass), and 2.5 is mainstream --
    steep Dehnen and generalized-NFW models sit here. So this is an ordinary
    density silently returning `nan` in `galax`'s default dtype, not an
    exotic input.

    The values are checked too, not just finiteness. Unrepresentable samples
    *are* zeroed -- the pad is not shortened to dodge them, which was tried
    and measured worse, costing every caller tail accuracy to avoid an
    overflow the anchored tail already absorbs. What makes zeroing cheap is
    that the solve anchors its inner tail above the dropped band, so
    ``[0, r_pad]`` is still integrated analytically.

    Zeroing alone used to cost 11% at ``gamma = 2.9``: the innermost 51 pad
    knots were dropped, and a zeroed ``rho[0]`` *also* failed the gate on
    that tail, so the band went too. For ``rho ~ r**-2.9`` the monopole
    integrand is ``r**-0.9`` and that band carries ~12% of the enclosed
    mass, which is the error that was observed.

    Against a float64 build, over the cases below:

    ======= ======== ==========
    gamma    mass     max rel
    ======= ======== ==========
    1.0      1        1.9e-07
    1.9      1        1.2e-07
    2.5      1        9.9e-08
    2.9      1        4.9e-06
    2.9      1e-2     4.7e-06
    2.9      1e-4     4.7e-06
    2.9      1e-6     4.9e-06
    ======= ======== ==========

    The mass column matters: an earlier fix bounded the projected
    coefficient against the dtype maximum, which is normalisation-dependent
    -- it passed at ``mass = 1`` by coincidence and returned the full 11%
    for anything smaller. Anchoring the tail does not care what the density
    is scaled by.

    """
    keys = lm_keys(8, "none")
    n_theta, n_phi = default_angular_resolution(8)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        return mass / (r**gamma * (1.0 + r) ** (4.0 - gamma))

    def build():
        # Built inside whichever dtype context is active: a grid made under
        # x64 and passed into a float32 build is a different test.
        r_knots = jnp.geomspace(1e-4, 1e4, 256)
        return build_expansion(
            rho, r_knots, 8, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
        )

    with jax.enable_x64(False):  # noqa: FBT003
        got = build()["phi_lm"]
        assert got.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(got)), (
            gamma,
            int(jnp.sum(~jnp.isfinite(got))),
            got.size,
        )
        f32 = jnp.asarray(got, dtype=float)

    # ...and it must be the right answer, not merely a finite one.
    ref = jnp.asarray(build()["phi_lm"], dtype=float)
    scale = jnp.max(jnp.abs(ref))
    assert float(jnp.max(jnp.abs(f32 - ref)) / scale) < rtol


def test_a_cusp_that_overflows_almost_the_whole_pad_keeps_its_interior() -> None:
    """The anchor must never land inside the range the caller asked for.

    REGRESSION: `build_expansion` clamps the anchor to ``lo``, the first
    retained knot, so that an unrepresentable band can never cost interior
    data. But `solve_poisson_profiles` adds `_ANCHOR_MARGIN` *after* that, so
    the effective anchor was ``lo + 8``. Every retained knot below it had its
    panels zeroed and the seed zero, so it came back with no inner-integral
    contribution at all -- representable interior data discarded by the guard
    written to protect it.

    It needs a cusp steep enough to overflow nearly the whole pad while
    staying finite inside the bracket: ``M = 1e25`` over ``[1e-4, 1e4]``
    reaches pad knot 124 of 128, where ``gamma = 2.9`` at unit mass reaches
    only 51. That is why the existing cases never found it.

    Measured float32 against float64: 9.0e-01 before the clamp accounted for
    the margin, 1.1e-03 after. The residual is honest degradation -- with
    almost the whole pad unrepresentable the boundary model carries the
    answer -- not the catastrophic loss of an unseeded recurrence.
    """
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        return 1e25 / (r**2.9 * (1.0 + r) ** 1.1)

    def build():
        r_knots = jnp.geomspace(1e-4, 1e4, 256)
        return build_expansion(
            rho, r_knots, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
        )["phi_lm"][:, 0]

    with jax.enable_x64(False):  # noqa: FBT003
        got = build()
        assert jnp.all(jnp.isfinite(got)), got
        f32 = jnp.asarray(got, dtype=float)

    ref = jnp.asarray(build(), dtype=float)
    scale = jnp.max(jnp.abs(ref))

    # The innermost retained knots are the ones the anchor used to swallow.
    inner = float(jnp.max(jnp.abs(f32[:8] - ref[:8])) / scale)
    assert inner < 1e-2, inner
    assert float(jnp.max(jnp.abs(f32 - ref)) / scale) < 1e-2


def test_a_nonfinite_outer_pad_leaves_the_inner_pad_alone() -> None:
    """A bad sample above the bracket must not anchor the tail below it.

    REGRESSION: `first_ok` scanned the whole padded grid for the deepest
    non-finite sample. "Everything below a bad band is suspect" is an
    argument about the *inner* pad -- the band the inner tail replaces -- and
    it does not reach past `lo`. A single non-finite sample in the *outer*
    pad was read as one in the inner pad, and since `first_ok` is clamped to
    ``lo - _ANCHOR_MARGIN``, it drove the anchor to ``lo`` and discarded the
    entire inner pad of finite, real density: exactly the unpadded
    configuration `_PAD_MULTIPLE` exists to avoid.

    The outer pad reaches ``r ~ 1e10`` on an ordinary bracket, so this needs
    no exotic density -- any ``rho_fn`` whose intermediates overflow out
    there returns ``nan``. Here it is forced explicitly, well above
    ``r_max``, so the mechanism is the only thing under test.

    Measured in float64 against the Hernquist closed form: 6.1e-05 while the
    scan ran over the whole grid, 1.0e-10 once restricted to the inner pad --
    which is what an unanchored solve gives, since nothing in the inner pad
    is bad and there is nothing to anchor.
    """
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        val = _hernquist_density(xyz, t)
        return jnp.where(r > 1e5, jnp.nan, val)

    r_knots = jnp.geomspace(1e-2, 1e2, 128)
    phi = build_expansion(
        rho, r_knots, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )["phi_lm"][:, 0] / jnp.sqrt(4 * jnp.pi)

    exact = -1.0 / (1.0 + r_knots)
    err = float(jnp.max(jnp.abs(phi - exact)) / jnp.max(jnp.abs(exact)))
    assert err < 1e-8, err


def test_a_nonfinite_band_mid_pad_anchors_above_it() -> None:
    """The anchor follows the *deepest* bad sample, not the first good one.

    A density need not be monotonic in log r, so an unrepresentable band can
    sit in the middle of the inner pad with finite samples below it. Reading
    the first good index puts the anchor at 0 -- no anchoring at all -- and
    leaves the zeroed band mid-pad for `fit_log_spline` to interpolate
    across, which it cannot. Everything below such a band is suspect even
    where it happens to sample finite.

    Measured in float64 against the Hernquist closed form: 7.5e-11 anchoring
    above the band, 8.1e-08 anchoring at the first good sample.
    """
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        # Inside the inner pad, which for this grid spans [9.4e-07, 1e-02].
        return jnp.where((r > 1e-5) & (r < 3e-5), jnp.nan, _hernquist_density(xyz, t))

    r_knots = jnp.geomspace(1e-2, 1e2, 128)
    phi = build_expansion(
        rho, r_knots, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )["phi_lm"][:, 0] / jnp.sqrt(4 * jnp.pi)

    exact = -1.0 / (1.0 + r_knots)
    err = float(jnp.max(jnp.abs(phi - exact)) / jnp.max(jnp.abs(exact)))
    assert err < 1e-9, err


_F32_EPS = float(jnp.finfo(jnp.float32).eps)
"""Round-off floor for the never-worse bound; see the assertion that uses it."""


# Measured on `main` (an unanchored solve) at 7260498c, float32, max relative
# error against the Dehnen closed form, keyed by (r_min, r_max, gamma, n_r).
# Anchoring must never do worse than these, and should usually do far better.
#
# The bracket is in the key because it is the axis that discriminates: the
# anchor is `_ANCHOR_MARGIN` knots above the bad band, and how much density
# that skips depends on the pad's step, which is the bracket's log span over
# the pad's knot count. A table pinned to one bracket cannot see it, and the
# first version of this test was pinned to [1e-4, 1e4]. The rows below the
# first block are the brackets where that cost up to 3378x. Adding a bracket
# means adding rows; the table is the case list.
_UNANCHORED = {
    (1e-4, 1e4, 1.0, 4): 1.6e-03,
    (1e-4, 1e4, 1.0, 5): 7.4e-05,
    (1e-4, 1e4, 1.0, 6): 2.6e-05,
    (1e-4, 1e4, 1.0, 7): 8.3e-06,
    (1e-4, 1e4, 1.0, 8): 8.9e-07,
    (1e-4, 1e4, 1.0, 16): 1.5e-07,
    (1e-4, 1e4, 1.0, 64): 1.5e-07,
    (1e-4, 1e4, 2.5, 4): 3.7e-06,
    (1e-4, 1e4, 2.5, 5): 1.5e-06,
    (1e-4, 1e4, 2.5, 6): 1.5e-06,
    (1e-4, 1e4, 2.5, 7): 1.3e-06,
    (1e-4, 1e4, 2.5, 8): 1.3e-06,
    (1e-4, 1e4, 2.5, 16): 1.3e-06,
    (1e-4, 1e4, 2.5, 64): 1.3e-06,
    (1e-4, 1e4, 2.7, 4): 6.9e-04,
    (1e-4, 1e4, 2.7, 5): 7.3e-04,
    (1e-4, 1e4, 2.7, 6): 6.4e-04,
    (1e-4, 1e4, 2.7, 7): 7.0e-04,
    (1e-4, 1e4, 2.7, 8): 7.2e-04,
    (1e-4, 1e4, 2.7, 16): 7.1e-04,
    (1e-4, 1e4, 2.7, 64): 6.8e-04,
    (1e-4, 1e4, 2.9, 4): 1.1e-01,
    (1e-4, 1e4, 2.9, 5): 1.1e-01,
    (1e-4, 1e4, 2.9, 6): 1.1e-01,
    (1e-4, 1e4, 2.9, 7): 1.2e-01,
    (1e-4, 1e4, 2.9, 8): 1.2e-01,
    (1e-4, 1e4, 2.9, 16): 1.2e-01,
    (1e-4, 1e4, 2.9, 64): 1.2e-01,
    (1e-5, 1e5, 2.5, 5): 3.8e-06,
    (1e-5, 1e5, 2.5, 6): 4.8e-06,
    (1e-5, 1e5, 2.5, 7): 4.6e-06,
    (1e-5, 1e5, 2.7, 5): 1.6e-03,
    (1e-5, 1e5, 2.7, 6): 1.5e-03,
    (1e-5, 1e5, 2.7, 7): 1.3e-03,
    (1e-6, 1e6, 2.5, 5): 1.3e-05,
    (1e-6, 1e6, 2.5, 6): 1.6e-05,
    (1e-6, 1e6, 2.5, 7): 1.3e-05,
    (1e-6, 1e6, 2.7, 5): 3.0e-03,
    (1e-6, 1e6, 2.7, 6): 2.9e-03,
    (1e-6, 1e6, 2.7, 7): 2.8e-03,
    (1e-4, 1e2, 2.5, 5): 8.8e-07,
    (1e-4, 1e2, 2.5, 6): 1.7e-06,
    (1e-4, 1e2, 2.5, 7): 1.3e-06,
    (1e-4, 1e2, 2.7, 5): 6.9e-04,
    (1e-4, 1e2, 2.7, 6): 6.5e-04,
    (1e-4, 1e2, 2.7, 7): 7.1e-04,
    (1e-3, 1e3, 2.5, 5): 4.1e-08,
    (1e-3, 1e3, 2.5, 6): 4.0e-08,
    (1e-3, 1e3, 2.5, 7): 2.8e-07,
    (1e-3, 1e3, 2.7, 5): 3.7e-04,
    (1e-3, 1e3, 2.7, 6): 3.3e-04,
    (1e-3, 1e3, 2.7, 7): 3.5e-04,
}


@pytest.mark.parametrize(("r_min", "r_max", "gamma", "n_r"), _UNANCHORED)
def test_anchoring_is_never_worse_than_an_unanchored_solve(
    r_min, r_max, gamma, n_r
) -> None:
    r"""Anchoring must not cost accuracy at any resolution, cusp or bracket.

    REGRESSION, three times over, each one a guard bounding the wrong thing.

    First the tail's activity gate was normalised by ``max|rho_col|`` over
    the whole column while testing ``rho_col[i0]``; before anchoring those
    were the same point for a cusp, so the gate could never fire. With an
    anchor it rejected once the ratio fell under ``sqrt(eps)``, dropping the
    tail -- the failure anchoring exists to prevent, re-created by its own
    margin.

    Then the fix for *that* exposed a bound on the slope fit's
    self-consistency, set first at 1.0 and then at 0.1 by measurement. Both
    bounded the fit's error *relative to the tail*, which does not reach the
    answer on its own: what reaches it is that error times how much larger
    the modelled band is than the band actually lost, and that factor is set
    by the grid, which the ratio never sees. 1.0 cost 99458x at
    ``gamma = 2.5, n_r = 5``; 0.1 cost 3378x at ``gamma = 2.5, n_r = 6`` over
    [1e-5, 1e5]. The gate is now the break-even test between the two, with no
    constant to tune.

    The sweep matters as much as the rule. Each defect lived where the
    previous version of this test did not look: at ``n_r`` 6 and 7 when it
    ran ``[4, 5, 8, 16, ...]``, at ``gamma = 2.5`` when it swept only 2.9,
    and at every bracket but one when it was pinned to [1e-4, 1e4]. A
    parameter sampled around its failure is not swept.

    The bound is `_UNANCHORED` with a 3x allowance for arithmetic
    reordering -- not a flat constant, which would have let the
    1.5e-06 -> 1.5e-01 case through at any threshold loose enough to pass
    ``gamma = 2.9`` at all.
    """
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        return 1.0 / (r**gamma * (1.0 + r) ** (4.0 - gamma))

    mass = 4.0 * jnp.pi / (3.0 - gamma)

    with jax.enable_x64(False):  # noqa: FBT003
        r_knots = jnp.geomspace(r_min, r_max, n_r)
        got = build_expansion(
            rho, r_knots, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
        )["phi_lm"][:, 0]
        assert jnp.all(jnp.isfinite(got)), got
        phi = jnp.asarray(got, dtype=float) / jnp.sqrt(4.0 * jnp.pi)

    r = jnp.asarray(r_knots, dtype=float)
    exact = -(mass / (2.0 - gamma)) * (1.0 - (r / (r + 1.0)) ** (2.0 - gamma))
    err = float(jnp.max(jnp.abs(phi - exact)) / jnp.max(jnp.abs(exact)))

    # 1.5x, not 3x. The worst row runs at 0.998 of its tabulated value, so 3x
    # was pure slack -- it let a gate loosened by a whole nat through at
    # (1e-4, 1e4, 2.5, 7), where the error doubles to 2.7e-06 against a 1.3e-06
    # row. 1.5x still leaves every row its measured headroom.
    #
    # Floored at a few float32 ulps, because a ratio between two round-off
    # numbers measures the platform, not the solve. Five of these rows sit
    # under it -- (1e-3, 1e3, 2.5, 5) is 4.1e-08, a third of an ulp -- and the
    # first CI run after the 3x allowance came down failed there on macOS at
    # 1.193e-07, which is 1.00 ulp exactly. The floor clears every row a
    # mutation has to beat, so it costs no sensitivity: the (2.5, 7) case
    # above stays bounded at 1.9e-06, well above it.
    bound = max(1.5 * _UNANCHORED[(r_min, r_max, gamma, n_r)], 5.0 * _F32_EPS)
    assert err < bound, (
        r_min,
        r_max,
        gamma,
        n_r,
        err,
    )


def test_a_cored_cusp_under_the_overflow_radius_degrades_no_further() -> None:
    r"""Pin the one case anchoring is *worse* on, so it cannot grow.

    The anchoring gate reads the slope fit at the anchor and above it. The
    band the tail extrapolates across lies entirely *below* the anchor and
    holds no sample -- it is by construction the band that was zeroed. So a
    density that is a clean power law above the float32 overflow radius and
    something else below it passes the gate with ``|s1 - s0|`` at round-off:
    maximal confidence drawn from an absence of data.

    This is such a density -- a cusp with its core hidden under the overflow
    radius, put there by a large prefactor. Anchoring extrapolates
    :math:`r^{-\gamma}` to the origin and over-counts; an unanchored solve
    drops the band and under-counts. Measured float32 against float64:
    **1.12 anchored, 2.6e-01 unanchored**, a 4.4x regression.

    No gate on the retained samples can fix this, because the evidence is
    gone. What this test is for is that the 4.4x stays 4.4x. Both answers are
    already useless here, and float64 never reaches it, so this is a bound on
    a known limit rather than a correctness requirement.

    `galax`'s own `safe_vector_norm` does not trigger it: its offset is
    ``finfo(dtype).tiny``, so the float32 core sits inside the overflow band
    where it is unobservable, which is why
    `test_a_steep_cusp_survives_the_padded_sampling_in_float32` reaches 4.9e-06.
    """
    gamma, mass_scale, core, n_r = 2.9, 6.8e29, 1e-4, 32
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        return mass_scale / (
            (r**2 + core**2) ** (gamma / 2.0) * (1.0 + r) ** (4.0 - gamma)
        )

    def build(r_knots):
        return build_expansion(
            rho, r_knots, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
        )["phi_lm"][:, 0]

    ref = jnp.asarray(build(jnp.geomspace(1e-1, 1e2, n_r)), dtype=float)

    with jax.enable_x64(False):  # noqa: FBT003
        got = build(jnp.geomspace(1e-1, 1e2, n_r))
        assert jnp.all(jnp.isfinite(got)), got
        f32 = jnp.asarray(got, dtype=float)

    err = float(jnp.max(jnp.abs(f32 - ref)) / jnp.max(jnp.abs(ref)))
    assert err < 2.0, err


def test_a_second_rescue_pins_the_gate_from_below() -> None:
    r"""A second lower-edge pin, a nat away from the first.

    `_UNANCHORED` structurally cannot catch a gate that has been tightened: it
    bounds from *above* against the unanchored value, which is exactly what a
    tightened gate returns. So the only thing holding the gate open is the
    rescue tests, and with one of them the break-even point could drift by
    2.9 nats before anything failed -- enough to give up most of what
    anchoring buys.

    This case sits at a different `exp_in * span` from the
    :math:`\gamma = 2.99` one, so the two together bracket the gate far more
    tightly than either alone.
    Measured float32 against the Dehnen closed form: 5.8e-02 anchored,
    1.7e-01 unanchored.
    """
    gamma, n_r = 2.9, 6
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        return 1.0 / (r**gamma * (1.0 + r) ** (4.0 - gamma))

    mass = 4.0 * jnp.pi / (3.0 - gamma)

    with jax.enable_x64(False):  # noqa: FBT003
        r_knots = jnp.geomspace(1e-6, 1e6, n_r)
        got = build_expansion(
            rho, r_knots, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
        )["phi_lm"][:, 0]
        assert jnp.all(jnp.isfinite(got)), got
        phi = jnp.asarray(got, dtype=float) / jnp.sqrt(4.0 * jnp.pi)

    r = jnp.asarray(r_knots, dtype=float)
    exact = -(mass / (2.0 - gamma)) * (1.0 - (r / (r + 1.0)) ** (2.0 - gamma))
    err = float(jnp.max(jnp.abs(phi - exact)) / jnp.max(jnp.abs(exact)))
    assert err < 1e-1, err


def test_anchoring_rescues_a_near_divergent_cusp() -> None:
    r"""The anchoring gate must stay loose enough to fire where it matters.

    Every other test here bounds anchoring from *above* -- it must not be
    worse than an unanchored solve. That cannot pin the gate from below,
    because any tightening of it only ever falls back to the unanchored
    answer, which those tests permit. So the gate could drift shut unnoticed,
    silently giving up the cases anchoring exists for.

    This is the case that discriminates. At :math:`\gamma = 2.99` -- just
    inside the finite-mass limit, where almost all the mass is in the cusp --
    an unanchored float32 build is 81% wrong, because the cusp is exactly
    what overflows and gets zeroed. Anchoring rescues it to 2.7%.

    It is also the case every *constant* threshold on the slope fit's
    self-consistency had to trade against: 0.01 was free of regressions
    everywhere else in the sweep and lost this rescue, while 0.1 kept it and
    cost 3378x elsewhere. Nothing in between did both. The break-even gate
    does, which is the point of it.
    """
    gamma, n_r = 2.99, 8
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        return 1.0 / (r**gamma * (1.0 + r) ** (4.0 - gamma))

    mass = 4.0 * jnp.pi / (3.0 - gamma)

    with jax.enable_x64(False):  # noqa: FBT003
        r_knots = jnp.geomspace(1e-4, 1e4, n_r)
        got = build_expansion(
            rho, r_knots, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
        )["phi_lm"][:, 0]
        phi = jnp.asarray(got, dtype=float) / jnp.sqrt(4.0 * jnp.pi)

    r = jnp.asarray(r_knots, dtype=float)
    exact = -(mass / (2.0 - gamma)) * (1.0 - (r / (r + 1.0)) ** (2.0 - gamma))
    err = float(jnp.max(jnp.abs(phi - exact)) / jnp.max(jnp.abs(exact)))

    # Unanchored is 8.1e-01 here; anchoring reaches 2.7e-02.
    assert err < 1e-1, err
