"""Tests for the radial Poisson solve."""

import pathlib

import numpy as np
import pytest

import quaxed.numpy as jnp

from galax.potential._src.harmonic.poisson import solve_poisson_lm

REFERENCE = (
    pathlib.Path(__file__).parents[3]
    / "functional"
    / "reference"
    / "harmonic"
    / "bfeax_reference.npz"
)


@pytest.mark.parametrize(
    "case",
    ["nfw_sph", "nfw_tri", "hernquist_sph", "plummer_sph", "jaffe_sph"],
)
def test_solve_poisson_lm_matches_bfeax(case) -> None:
    """Phi_lm agrees with `bfeax` on every mode it handles correctly.

    Modes with rho_lm(r_max) < 0 are excluded: `bfeax` drops their outer tail
    and we do not. See https://github.com/jnibauer/bfeax/issues/1
    """
    ref = np.load(REFERENCE)
    r = jnp.asarray(ref["r_knots"])
    rho_lm = jnp.asarray(ref[f"{case}_rho_lm"])
    l_per_mode = jnp.asarray(ref[f"{case}_lm"][:, 0], dtype=float)
    expect = ref[f"{case}_phi_lm"]

    got = np.asarray(solve_poisson_lm(r, rho_lm, l_per_mode, jnp.asarray(1.0)))
    agrees = np.asarray(ref[f"{case}_rho_lm"])[-1, :] > 0.0

    scale = np.max(np.abs(expect))
    assert np.allclose(
        got[:, agrees], expect[:, agrees], atol=1e-13 * scale, rtol=1e-11
    )
    # And the excluded modes really do differ -- otherwise the fix did nothing.
    if not agrees.all():
        assert not np.allclose(
            got[:, ~agrees], expect[:, ~agrees], atol=1e-13 * scale, rtol=1e-11
        )


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

    `bfeax` gates on `rho_lm[-1] > 0.0`, silently dropping the correction for
    negative modes -- routine for l >= 1. A mode and its negation must give
    exactly opposite Phi_lm, since the solve is linear in rho_lm.

    See https://github.com/jnibauer/bfeax/issues/1
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
