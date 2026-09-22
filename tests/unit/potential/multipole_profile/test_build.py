"""Tests for the multipole profile build pipeline."""

import quaxed.numpy as jnp

from galax.potential._src.builtin.multipole_profile.build import (
    build_expansion,
    subtract_inner_cusp,
)
from galax.potential._src.harmonic import (
    default_angular_resolution,
    eval_log_spline,
    lm_keys,
)


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
    assert coeffs["phi_lm"].shape == (32, n_modes)
    assert coeffs["dphi_lm"].shape == (32, n_modes)
    assert coeffs["rho_residual_lm"].shape == (32, n_modes)
    assert coeffs["drho_residual_lm"].shape == (32, n_modes)
    assert coeffs["rho_alpha"].shape == (n_modes,)
    assert coeffs["rho_amplitude"].shape == (n_modes,)
