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


def test_subtract_inner_cusp_splits_only_the_monopole() -> None:
    """Every mode but `l = 0` keeps its profile in the residual.

    Mode 1 here is a clean, same-sign power law as large as the monopole, so
    nothing about it is degenerate: under the magnitude-and-sign gate this
    replaced it would have been split like any other well-behaved mode. It
    must not be, because `alpha` and `amplitude` are interpolated in time by
    a grid build and only the monopole is guaranteed not to pass through
    zero -- see `test_a_time_grid_density_survives_a_sign_changing_mode`.
    """
    r = jnp.geomspace(1e-2, 1e2, 64)
    mono, other = 1.0 / r, 2.0 * r**-1.5
    rho_lm = jnp.stack([mono, other], axis=-1)

    residual, alpha, amplitude = subtract_inner_cusp(r, rho_lm)

    # The monopole is split.
    assert jnp.isclose(alpha[0], -1.0, rtol=1e-10)
    assert jnp.isclose(amplitude[0], mono[0], rtol=1e-12)
    assert jnp.max(jnp.abs(residual[:, 0])) < 1e-9 * jnp.max(jnp.abs(mono))

    # The l > 0 mode is not, despite being perfectly splittable.
    assert alpha[1] == 0.0
    assert amplitude[1] == 0.0
    assert jnp.allclose(residual[:, 1], other, atol=0.0)


def test_subtract_inner_cusp_survives_a_sign_changing_mode() -> None:
    """A zero crossing in the fitting window must not produce a background.

    `alpha` is fitted to log|rho_lm| over the innermost three knots, so a sign
    change there reads as a large positive slope and a background that
    diverges outward: with the crossing adjacent to knot 0 the fit gives
    alpha = +69.9, and even clipped to +3 the background reaches ~1e8 times
    the mode, wrecking `residual + background` by catastrophic cancellation.

    Restricting the split to `l = 0` covers this without a sign test, because
    a sign change is only possible in the modes that are no longer split.
    """
    r = jnp.geomspace(1e-2, 1e2, 64)
    clean = 1.0 / r
    crossing = clean.at[0].set(-1e-5 * clean[0])
    rho_lm = jnp.stack([clean, crossing], axis=-1)

    residual, alpha, amplitude = subtract_inner_cusp(r, rho_lm)

    assert amplitude[1] == 0.0
    assert alpha[1] == 0.0
    # No background, so the residual carries the mode exactly.
    assert jnp.allclose(residual[:, 1], crossing, atol=0.0)
    assert jnp.all(jnp.isfinite(residual))
    # The monopole still gets its cusp subtracted.
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


@jax.jit
def _steep_cusp_phi_lm(shape):
    """`build_expansion` of ``mass / (r**g (1 + r)**(4 - g))``, shape traced."""
    gamma, outer, mass = shape
    keys = lm_keys(8, "none")
    n_theta, n_phi = default_angular_resolution(8)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        return mass / (r**gamma * (1.0 + r) ** outer)

    # Built inside whichever dtype context is active: a grid made under x64
    # and passed into a float32 build is a different test.
    r_knots = jnp.geomspace(1e-4, 1e4, 256)
    return build_expansion(
        rho, r_knots, 8, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )["phi_lm"]


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
    # `gamma` and `mass` ride in a traced array so all seven rows share one
    # compile per dtype; a closure per row compiled each. `4 - gamma` is formed
    # here, in double, as the closure formed it.
    shape = [gamma, 4.0 - gamma, mass]

    with jax.enable_x64(False):  # noqa: FBT003
        got = _steep_cusp_phi_lm(jnp.asarray(shape))
        assert got.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(got)), (
            gamma,
            int(jnp.sum(~jnp.isfinite(got))),
            got.size,
        )
        f32 = jnp.asarray(got, dtype=float)

    # ...and it must be the right answer, not merely a finite one.
    ref = jnp.asarray(_steep_cusp_phi_lm(jnp.asarray(shape)), dtype=float)
    assert ref.dtype == jnp.float64, "the reference needs x64; see the pytest env"
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
        # Inside the inner pad, which for this grid spans [1.0e-10, 8.7e-03]:
        # `_pad_grid` reaches by log span, not by knot count.
        return jnp.where((r > 1e-5) & (r < 3e-5), jnp.nan, _hernquist_density(xyz, t))

    r_knots = jnp.geomspace(1e-2, 1e2, 128)
    phi = build_expansion(
        rho, r_knots, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )["phi_lm"][:, 0] / jnp.sqrt(4 * jnp.pi)

    exact = -1.0 / (1.0 + r_knots)
    err = float(jnp.max(jnp.abs(phi - exact)) / jnp.max(jnp.abs(exact)))
    assert err < 1e-9, err


def test_a_second_rescue_pins_the_gate_from_below() -> None:
    r"""A second lower-edge pin, a nat away from the first.

    `_UNANCHORED` (in `test_build_sweeps.py`) structurally cannot catch a gate
    that has been tightened: it bounds from *above* against the unanchored
    value, which is exactly what a tightened gate returns. So the only thing
    holding the gate open is the rescue tests, and with one of them the
    break-even point could drift by 2.9 nats before anything failed -- enough
    to give up most of what anchoring buys.

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


def _grad_build(rho_of, n_r: int = 64):
    """``phi_lm.sum()`` as a function of ``(A, s)``, for ``rho_of(A, s)``."""
    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)

    def total(p):
        return build_expansion(
            rho_of(p[0], p[1]),
            jnp.geomspace(1e-4, 1e4, n_r),
            0,
            keys,
            n_theta,
            n_phi,
            jnp.asarray(0.0),
            jnp.asarray(1.0),
        )["phi_lm"].sum()

    return total


def _log_dehnen(gamma):
    """Dehnen-like cusp ``A / (x**g (1 + x)**(4 - g))``, ``x = r / s``, in log form.

    Written this way its derivative stays representable where the density is
    large, which ``A / (x**g * ...)`` does not (see `from_density`'s Notes).
    """

    def rho_of(A, s):
        def rho(xyz, t):
            x = safe_vector_norm(xyz) / s
            return jnp.exp(
                jnp.log(A) - gamma * jnp.log(x) - (4.0 - gamma) * jnp.log1p(x)
            )

        return rho

    return rho_of


@pytest.mark.parametrize("gamma", [1.0, 2.5, 2.9])
def test_float32_gradients_through_a_build_whose_pad_overflows(gamma) -> None:
    """The amplitude and scale-radius gradients are finite and match float64.

    REGRESSION: every one of these was `nan` in float32. The padded grid
    samples the density where it overflows (``gamma = 2.5`` reaches ``inf``
    near ``1e-16``), `_drop_nonfinite` zeroes those samples, and the zero
    cotangent then met the infinite derivative inside the caller's own density:
    ``0 * inf``. A ``where`` applied afterwards cannot help, and one inside the
    density is not ours to write, so `harmonic_coeffs` now differentiates
    ``rho_fn`` at a representable radius for the radii the build discards.

    The outer pad needed the same: there the density underflows to a finite
    zero, but its derivative through ``x**g`` is ``0 * inf``. Measured against
    float64, over this grid and ``[1e-6, 1e6]``, ``n_r`` of 16 and 64: 4e-8 to
    1.4e-5.
    """
    total = _grad_build(_log_dehnen(gamma))

    ref = jnp.asarray(jax.grad(total)(jnp.asarray([1.0, 1.0])), dtype=float)
    assert ref.dtype == jnp.float64, "the reference needs x64; see the pytest env"

    with jax.enable_x64(False):  # noqa: FBT003
        got = jax.grad(total)(jnp.asarray([1.0, 1.0]))
        assert got.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(got)), got
        f32 = jnp.asarray(got, dtype=float)

    err = float(jnp.max(jnp.abs(f32 - ref) / jnp.abs(ref)))
    assert err < 1e-4, err


def test_float32_forward_mode_agrees_with_reverse_mode_through_the_build() -> None:
    """``jacfwd`` and ``jacrev`` agree, so the rule is a consistent derivative."""
    total = _grad_build(_log_dehnen(2.9))

    with jax.enable_x64(False):  # noqa: FBT003
        p = jnp.asarray([1.0, 1.0])
        fwd, rev = jax.jacfwd(total)(p), jax.jacrev(total)(p)
        assert jnp.all(jnp.isfinite(fwd)), fwd
        assert jnp.allclose(fwd, rev, rtol=1e-4), (fwd, rev)


def test_the_amplitude_gradient_survives_a_density_with_no_scale_gradient() -> None:
    """``A / (x**g (1 + x)**h)`` has an exact ``dA`` even when its ``ds`` is `nan`.

    JAX differentiates the quotient through ``1 / den**2``, which underflows
    for ``gamma = 2.9`` once the density nears the dtype's maximum, so the
    gradient with respect to the scale radius is not representable and stays
    non-finite (the Notes of `from_density` say so). The amplitude enters
    linearly, so its gradient must equal the value itself, and must not be lost
    with it.
    """
    gamma = 2.9

    def rho_of(A, s):
        def rho(xyz, t):
            x = safe_vector_norm(xyz) / s
            return A / (x**gamma * (1.0 + x) ** (4.0 - gamma))

        return rho

    total = _grad_build(rho_of)
    with jax.enable_x64(False):  # noqa: FBT003
        p = jnp.asarray([1.0, 1.0])
        value, grad = total(p), jax.grad(total)(p)
        assert jnp.isfinite(grad[0]), grad
        assert jnp.allclose(grad[0], value, rtol=1e-5), (grad[0], value)
