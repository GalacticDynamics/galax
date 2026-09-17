"""Tests for the radial Poisson solve."""

import pathlib

import numpy as np
import pytest

import quaxed.numpy as jnp

from galax.potential._src.builtin.multipole_profile.poisson import solve_poisson_lm

REFERENCE = (
    pathlib.Path(__file__).parents[3]
    / "functional"
    / "reference"
    / "multipole_profile"
    / "bfeax_reference.npz"
)


@pytest.mark.parametrize(
    "case",
    ["nfw_sph", "nfw_tri", "hernquist_sph", "plummer_sph", "jaffe_sph"],
)
def test_solve_poisson_lm_matches_bfeax(case) -> None:
    """Phi_lm agrees with the vendored `bfeax` oracle at G=1."""
    ref = np.load(REFERENCE)
    r = jnp.asarray(ref["r_knots"])
    rho_lm = jnp.asarray(ref[f"{case}_rho_lm"])
    l_per_mode = jnp.asarray(ref[f"{case}_lm"][:, 0], dtype=float)
    expect = ref[f"{case}_phi_lm"]

    got = solve_poisson_lm(r, rho_lm, l_per_mode, jnp.asarray(1.0))
    scale = np.max(np.abs(expect))
    assert np.allclose(np.asarray(got), expect, atol=1e-13 * scale, rtol=1e-11)


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
