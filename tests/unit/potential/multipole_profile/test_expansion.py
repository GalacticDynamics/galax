"""Tests for expansion evaluation."""

import jax

import quaxed.numpy as jnp

from galax.potential._src.builtin.multipole_profile.expansion import (
    expansion_density,
    expansion_potential,
)
from galax.potential._src.builtin.multipole_profile.funcs import (
    build_expansion,
    radial_grid,
)
from galax.potential._src.builtin.multipole_profile.project import (
    default_angular_resolution,
    lm_keys,
)


def _hernquist_params(n_r: int = 256):
    """M=1, a=1 Hernquist: rho = 1/(2 pi) / (r (1+r)^3), Phi = -1/(1+r), G=1."""

    def rho(xyz, t):
        r = jnp.linalg.norm(xyz, axis=-1)
        return 1.0 / (2.0 * jnp.pi) / (r * (1.0 + r) ** 3)

    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)
    r = radial_grid(n_r, jnp.asarray(1e-3), jnp.asarray(1e3))
    p = build_expansion(
        rho, r, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )
    return {**p, "r_knots": r}, keys


def test_expansion_potential_matches_the_hernquist_closed_form() -> None:
    p, keys = _hernquist_params()
    xyz = jnp.asarray([[1.0, 2.0, 3.0], [0.5, 0.0, 0.0], [10.0, 5.0, 2.0]])
    r = jnp.linalg.norm(xyz, axis=-1)
    got = expansion_potential(p, xyz, 0, keys)
    assert jnp.allclose(got, -1.0 / (1.0 + r), rtol=1e-3)


def test_expansion_density_matches_the_hernquist_closed_form() -> None:
    """The rho_lm splines are far more accurate than laplacian(Phi)."""
    p, keys = _hernquist_params()
    xyz = jnp.asarray([[1.0, 2.0, 3.0], [0.5, 0.0, 0.0], [10.0, 5.0, 2.0]])
    r = jnp.linalg.norm(xyz, axis=-1)
    expect = 1.0 / (2.0 * jnp.pi) / (r * (1.0 + r) ** 3)
    assert jnp.allclose(expansion_density(p, xyz, 0, keys), expect, rtol=1e-4)


def test_expansion_potential_is_batched_consistently() -> None:
    p, keys = _hernquist_params()
    batch = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = expansion_potential(p, batch, 0, keys)
    assert got.shape == (2,)
    assert jnp.allclose(got[0], expansion_potential(p, batch[0], 0, keys))


def test_gradient_is_finite_on_the_z_axis() -> None:
    """`bfeax` returns NaN here; this is the regression guard."""

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    keys = lm_keys(4, None)
    n_theta, n_phi = default_angular_resolution(4)
    r = radial_grid(64, jnp.asarray(1e-2), jnp.asarray(1e2))
    p = {
        **build_expansion(
            rho, r, 4, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
        ),
        "r_knots": r,
    }

    grad = jax.grad(lambda xyz: expansion_potential(p, xyz, 4, keys))(
        jnp.asarray([0.0, 0.0, 2.0])
    )
    assert jnp.all(jnp.isfinite(grad))


def test_expansion_potential_is_jittable() -> None:
    """`l_max` and `keys` are static, so shapes are compile-time constants."""
    p, keys = _hernquist_params(n_r=32)
    jitted = jax.jit(expansion_potential, static_argnums=(2, 3))
    assert jnp.isfinite(jitted(p, jnp.asarray([1.0, 2.0, 3.0]), 0, keys))
