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


# `galax`'s projection asks for float64 explicitly, so running the builder
# under `enable_x64(False)` emits JAX's truncation warning, which
# `filterwarnings = ["error"]` turns into a failure. That wart is real and
# worth fixing, but it is not what these two tests are about.
@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
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


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
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


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
@pytest.mark.parametrize(
    ("gamma", "rtol"), [(1.0, 1e-5), (1.9, 1e-5), (2.5, 1e-5), (2.9, 0.2)]
)
def test_a_steep_cusp_survives_the_padded_sampling_in_float32(gamma, rtol) -> None:
    """A density the *pad* cannot represent must not zero the whole build.

    REGRESSION: padding evaluates ``rho_fn`` `_PAD_MULTIPLE` spans outside
    the requested bracket, so a cusp ``rho ~ r**-gamma`` is sampled at
    ``r_min * exp(-reach)`` where the density overflows once
    ``gamma * reach`` clears the dtype's exponent. `_pad_grid`'s reach cap
    does not help -- it is sized so the solver's own ``x**2`` stays finite
    and knows nothing about ``rho_fn``'s slope.

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

    The values are checked too, not just finiteness. Dropping a pad sample
    usually costs nothing -- the density there is huge but its ``x**(l+3)``
    weight is negligible -- and float32 tracks float64 to round-off through
    ``gamma = 2.5``:

    ======= ==========
    gamma    max rel
    ======= ==========
    1.0      1.7e-7
    1.9      1.2e-7
    2.5      1.2e-6
    2.9      1.1e-1
    ======= ==========

    It is not free at the steep end: by ``gamma = 2.9`` enough of the inner
    pad is dropped to cost 11%, hence the looser bound there. That is a
    bounded, one-sided loss of *tail* accuracy rather than a `nan`, which is
    the trade this module takes everywhere else -- but sampling `rho_fn`
    many decades out is the real problem, and extrapolating the density into
    the pad analytically instead would avoid it.
    """
    keys = lm_keys(8, "none")
    n_theta, n_phi = default_angular_resolution(8)

    def rho(xyz, t):
        r = safe_vector_norm(xyz)
        return 1.0 / (r**gamma * (1.0 + r) ** (4.0 - gamma))

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
