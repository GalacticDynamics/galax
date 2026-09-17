"""Tests for the angular projection machinery."""

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp

from galax.potential._src.builtin.multipole_profile.project import (
    lm_keys,
    real_ylm,
)


def test_lm_keys_none_is_every_mode() -> None:
    """`symmetry=None` keeps all (l, m) with -l <= m <= l."""
    assert lm_keys(2, None) == (
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


def test_lm_keys_spherical_is_monopole_only() -> None:
    assert lm_keys(8, "spherical") == ((0, 0),)


def test_lm_keys_axisymmetric_is_even_l_m_zero() -> None:
    """Azimuthal symmetry kills m != 0; equatorial symmetry kills odd l."""
    assert lm_keys(5, "axisymmetric") == ((0, 0), (2, 0), (4, 0))


def test_lm_keys_triaxial_is_even_l_even_nonneg_m() -> None:
    """Octant symmetry: cosine terms only, both indices even."""
    assert lm_keys(4, "triaxial") == (
        (0, 0),
        (2, 0),
        (2, 2),
        (4, 0),
        (4, 2),
        (4, 4),
    )


def test_lm_keys_rejects_unknown_symmetry() -> None:
    with pytest.raises(ValueError, match="Unknown symmetry"):
        lm_keys(4, "octahedral")


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
