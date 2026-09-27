"""Tests for the multipole profile build pipeline."""

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
    assert coeffs["phi_lm"].shape == (32, n_modes)
    assert coeffs["dphi_lm"].shape == (32, n_modes)
    assert coeffs["rho_residual_lm"].shape == (32, n_modes)
    assert coeffs["drho_residual_lm"].shape == (32, n_modes)
    assert coeffs["rho_alpha"].shape == (n_modes,)
    assert coeffs["rho_amplitude"].shape == (n_modes,)


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
