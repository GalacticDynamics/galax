"""Tests for the angular projection machinery."""

import pathlib

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp

from galax.potential._src.harmonic.project import (
    angular_grid,
    default_angular_resolution,
    harmonic_coeffs,
    lm_keys,
    real_ylm,
)
from galax.potential._src.symmetry import Symmetry


def test_lm_keys_none_is_every_mode() -> None:
    """`Symmetry.NONE` keeps all (l, m) with -l <= m <= l."""
    assert lm_keys(2, Symmetry.NONE) == (
        (0, 0),
        (1, -1),
        (1, 0),
        (1, 1),
        (2, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (2, 2),
    )


def test_lm_keys_accepts_none_as_an_alias_for_symmetry_none() -> None:
    assert lm_keys(2, None) == lm_keys(2, Symmetry.NONE)


def test_lm_keys_spherical_is_monopole_only() -> None:
    assert lm_keys(8, Symmetry.SPHERICAL) == ((0, 0),)


def test_lm_keys_zrotation_is_every_l_with_m_zero() -> None:
    """Rotation about z kills m != 0 and nothing else -- odd l is kept."""
    assert lm_keys(5, Symmetry.ZROTATION) == (
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
    )


def test_lm_keys_zrotation_zreflection_is_even_l_with_m_zero() -> None:
    """Adding z -> -z to the z-rotation drops the odd l."""
    assert lm_keys(5, Symmetry.ZROTATION_ZREFLECTION) == ((0, 0), (2, 0), (4, 0))


def test_zrotation_retains_odd_l_of_a_z_offset_density() -> None:
    """An exactly z-rotation-invariant density offset in z has odd-l power.

    `Symmetry.ZROTATION` constrains rotation about z, not z -> -z. Restricting
    to even l (as `Symmetry.ZROTATION_ZREFLECTION` and upstream ``bfeax``'s
    "axisymmetric" do) silently discards that power: here the (1, 0) mode is
    27% of the monopole.
    """

    def rho(xyz, t):
        R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
        z = xyz[..., 2]
        return jnp.exp(-jnp.sqrt(R2 + 1.0)) * jnp.exp(-(((z - 2.0) / 3.0) ** 2))

    l_max = 4
    keys = lm_keys(l_max, Symmetry.ZROTATION)
    assert keys == ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
    # ZROTATION_ZREFLECTION would drop exactly the odd-l modes measured below.
    assert lm_keys(l_max, Symmetry.ZROTATION_ZREFLECTION) == ((0, 0), (2, 0), (4, 0))

    # Converged quadrature, so this measures the mode set and not the rule.
    got = harmonic_coeffs(
        rho, jnp.asarray([1.0]), l_max, keys, 60, 61, jnp.asarray(0.0)
    )
    rho_lm = dict(zip(keys, got[0], strict=True))

    assert jnp.isclose(rho_lm[(0, 0)], 0.6298936286502026, rtol=1e-10)
    assert jnp.isclose(rho_lm[(1, 0)], 0.17199367509517, rtol=1e-10)
    assert jnp.isclose(rho_lm[(3, 0)], 0.015176113051951181, rtol=1e-10)
    # The dropped modes are not negligible: (1, 0) is 27% of the monopole.
    assert rho_lm[(1, 0)] / rho_lm[(0, 0)] > 0.25
    assert rho_lm[(3, 0)] / rho_lm[(0, 0)] > 0.02


def test_lm_keys_plane_reflection_is_even_l_even_nonneg_m() -> None:
    """Reflection about all three principal planes: both indices even, m >= 0."""
    assert lm_keys(4, Symmetry.PLANE_REFLECTION) == (
        (0, 0),
        (2, 0),
        (2, 2),
        (4, 0),
        (4, 2),
        (4, 4),
    )


def test_lm_keys_rejects_unknown_symmetry() -> None:
    with pytest.raises(ValueError, match="is not a valid Symmetry"):
        lm_keys(4, "octahedral")


@pytest.mark.parametrize("sym", list(Symmetry))
def test_plain_string_and_member_behave_identically(sym) -> None:
    """`Symmetry` is a `StrEnum`: the value is accepted wherever the member is."""
    assert sym == sym.value
    assert lm_keys(5, sym.value) == lm_keys(5, sym)


def test_real_ylm_is_orthonormal() -> None:
    """<Y_lm, Y_l'm'> = delta over a Gauss-Legendre x uniform-phi grid.

    Exercises the sqrt(2) real-harmonic convention: get it wrong and the
    diagonal comes out at 2 or 1/2 rather than 1.
    """
    l_max = 4
    keys = lm_keys(l_max, None)

    # Quadrature exact enough for degree-2*l_max integrands.
    n_theta, n_phi = 3 * l_max + 2, 4 * l_max + 3
    x, w = np.polynomial.legendre.leggauss(n_theta)
    cos_t = jnp.asarray(x)
    sin_t = jnp.sqrt(1.0 - cos_t**2)
    phi = jnp.arange(n_phi) * (2.0 * jnp.pi / n_phi)
    uvec = jnp.stack(
        [
            sin_t[:, None] * jnp.cos(phi)[None, :],
            sin_t[:, None] * jnp.sin(phi)[None, :],
            jnp.broadcast_to(cos_t[:, None], (n_theta, n_phi)),
        ],
        axis=-1,
    )
    weights = jnp.asarray(w)[:, None] * (2.0 * jnp.pi / n_phi)

    Y = real_ylm(l_max, keys, uvec)  # (n_modes, n_theta, n_phi)
    gram = jnp.einsum("aij,bij,ij->ab", Y, Y, weights)
    assert jnp.allclose(gram, jnp.eye(len(keys)), atol=1e-12)


def test_real_ylm_is_finite_and_differentiable_on_the_z_axis() -> None:
    """The reason `iter_Ylm` is used instead of theta/phi harmonics.

    A (theta, phi) formulation has a 0/0 gradient from `atan2` on the whole
    z-axis for every m >= 1.
    """
    keys = lm_keys(3, None)

    def f(z):
        uvec = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z])
        return jnp.sum(real_ylm(3, keys, uvec))

    val = f(jnp.asarray(1.0))
    grad = jax.grad(f)(jnp.asarray(1.0))
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad)


REFERENCE = (
    pathlib.Path(__file__).parents[3]
    / "functional"
    / "reference"
    / "harmonic"
    / "bfeax_reference.npz"
)


def _spheroid_density(alpha, beta, gamma, q_y=1.0, q_z=1.0):
    """Build the profile used for the vendored reference values."""

    def rho(xyz, t):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        rt = jnp.sqrt(x**2 + (y / q_y) ** 2 + (z / q_z) ** 2)
        return rt ** (-gamma) * (1.0 + rt**alpha) ** ((gamma - beta) / alpha)

    return rho


def test_default_angular_resolution_oversamples() -> None:
    """`n_theta = 2 l_max + 2`, `n_phi = 4 l_max + 2` -- ~2x Nyquist.

    Deliberately above ``bfeax``'s minimal (l_max + 2, 2 l_max + 1), which is
    exact only for band-limited densities and aliases otherwise.
    """
    assert default_angular_resolution(8) == (18, 34)
    assert default_angular_resolution(0) == (2, 2)


def test_angular_grid_vectors_are_unit_and_weights_sum_to_4pi() -> None:
    uvec, w = angular_grid(10, 17)
    assert uvec.shape == (10, 17, 3)
    assert w.shape == (10, 17)
    assert jnp.allclose(jnp.linalg.norm(uvec, axis=-1), 1.0, atol=1e-14)
    assert jnp.isclose(jnp.sum(w), 4.0 * jnp.pi, atol=1e-12)


def test_harmonic_coeffs_monopole_of_a_spherical_profile() -> None:
    """For spherical rho, rho_00(r) = sqrt(4 pi) rho(r), since Y_00 = 1/sqrt(4 pi)."""
    rho = _spheroid_density(1.0, 3.0, 1.0)
    r = jnp.asarray([0.1, 1.0, 10.0])
    got = harmonic_coeffs(rho, r, 0, ((0, 0),), 2, 1, jnp.asarray(0.0))
    xyz = jnp.stack([r, jnp.zeros_like(r), jnp.zeros_like(r)], axis=-1)
    expect = jnp.sqrt(4.0 * jnp.pi) * rho(xyz, jnp.asarray(0.0))
    assert jnp.allclose(got[:, 0], expect, rtol=1e-12)


@pytest.mark.parametrize(
    ("case", "alpha", "beta", "gamma", "q_y", "q_z", "symmetry"),
    [
        ("nfw_sph", 1.0, 3.0, 1.0, 1.0, 1.0, "spherical"),
        ("nfw_tri", 1.0, 3.0, 1.0, 0.8, 0.5, "plane_reflection"),
        ("hernquist_sph", 1.0, 4.0, 1.0, 1.0, 1.0, "spherical"),
        ("plummer_sph", 2.0, 5.0, 0.0, 1.0, 1.0, "spherical"),
        ("jaffe_sph", 1.0, 4.0, 2.0, 1.0, 1.0, "spherical"),
    ],
)
def test_harmonic_coeffs_matches_bfeax(
    case, alpha, beta, gamma, q_y, q_z, symmetry
) -> None:
    """rho_lm agrees with the vendored `bfeax` oracle to float64 tolerance."""
    ref = np.load(REFERENCE)
    r = jnp.asarray(ref["r_knots"])
    expect = ref[f"{case}_rho_lm"]
    keys = lm_keys(8, symmetry)
    assert [tuple(k) for k in ref[f"{case}_lm"]] == list(keys)

    # Pinned to the literal rule `bfeax` used to generate the reference, not
    # `default_angular_resolution(8)`: the library default has deliberately
    # moved to an oversampled rule to control aliasing, and this test is an
    # oracle comparison that must reproduce the upstream quadrature exactly.
    n_theta, n_phi = 10, 17
    got = harmonic_coeffs(
        _spheroid_density(alpha, beta, gamma, q_y, q_z),
        r,
        8,
        keys,
        n_theta,
        n_phi,
        jnp.asarray(0.0),
    )
    scale = np.max(np.abs(expect))
    assert np.allclose(np.asarray(got), expect, atol=1e-13 * scale, rtol=1e-11)
