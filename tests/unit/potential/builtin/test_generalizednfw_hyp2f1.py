"""Regression tests for `gNFWPotential`'s native hyp2f1 fallback.

Targeted at three bugs in `tfp.math.hypergeometric.hyp2f1_small_argument`
that `gNFWPotential`'s native implementation fixes:

- GalacticDynamics/galax#817: returns `NaN` for `Bz_from_hyp2f1`'s `b == 0`
  call pattern once `z >= 0.9` (i.e. `r >= 9 * r_s`).
- GalacticDynamics/galax#819: its custom gradient supports only `z`, so
  `jax.grad` with respect to `gamma` was silently wrong.
- GalacticDynamics/galax#820: its custom gradient is implemented via
  `jax.custom_vjp`, which can never be forward-differentiated, so
  `.hessian()`/`.tidal_tensor()` raised unconditionally (any `r`, any
  `gamma`).

This module is standalone (not integrated into the
`AbstractSinglePotential_Test` fixture suite used by sibling potentials)
since no such suite exists yet for `gNFWPotential`; adding one is a
separate, pre-existing gap.
"""

import jax
import numpy as np
import pytest
from scipy import integrate

import quaxed.numpy as jnp

import galax.potential as gp
from galax.potential._src.builtin.nfw.generalized import mass_enclosed
from galax.potential._src.builtin.nfw.hyp2f1 import (
    Bz_from_hyp2f1,
    _Bz_from_hyp2f1_impl,
)

GAMMAS = [0.0, 0.5, 1.0, 1.5, 1.99]
LARGE_RADII = [9.0, 9.1, 20.0, 100.0, 1000.0]  # z = r/(r+r_s) >= 0.9


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("r", LARGE_RADII)
def test_potential_finite_at_large_radius(gamma: float, r: float) -> None:
    """`potential`/`density` must not be NaN for r >= 9*r_s (was #817)."""
    pot = gp.gNFWPotential(m=1e12, r_s=1, gamma=gamma, units="galactic")
    x = jnp.array([r, 0.0, 0.0])
    assert jnp.isfinite(pot.potential(x, t=0))
    assert jnp.isfinite(pot.density(x, t=0))
    assert jnp.all(jnp.isfinite(pot.gradient(x, t=0)))


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("r", [0.1, 0.5, 1.0, 8.0, *LARGE_RADII])
def test_mass_enclosed_matches_direct_quadrature(gamma: float, r: float) -> None:
    """`mass_enclosed` must match direct numerical integration of the density."""
    m, r_s = 1e12, 1.0

    def rho(rr: float) -> float:
        x = rr / r_s
        rho0 = m / (4 * np.pi * r_s**3)
        return rho0 * x ** (-gamma) * (1 + x) ** (gamma - 3)

    expect, _ = integrate.quad(lambda rr: 4 * np.pi * rr**2 * rho(rr), 0, r, limit=400)

    got = float(mass_enclosed({"m": m, "r_s": r_s, "gamma": gamma}, jnp.array(r)))
    assert got == pytest.approx(expect, rel=1e-8)


def test_gamma1_matches_nfw_at_large_radius() -> None:
    """`gamma=1` gNFW must still match plain NFW beyond the old z=0.9 cutoff."""
    gnfw = gp.gNFWPotential(m=1e12, r_s=1, gamma=1, units="galactic")
    nfw = gp.NFWPotential(m=1e12, r_s=1, units="galactic")
    for r in LARGE_RADII:
        x = jnp.array([r, 0.0, 0.0])
        assert jnp.isclose(gnfw.potential(x, t=0), nfw.potential(x, t=0), atol=1e-8)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("r", [1.0, 5.0, *LARGE_RADII])
def test_hessian_and_tidal_tensor_finite(gamma: float, r: float) -> None:
    """`.hessian()`/`.tidal_tensor()` must not raise, for any r/gamma (was #820)."""
    pot = gp.gNFWPotential(m=1e12, r_s=1, gamma=gamma, units="galactic")
    x = jnp.array([r, 0.3, 0.1])
    assert jnp.all(jnp.isfinite(pot.hessian(x, t=0)))
    assert jnp.all(jnp.isfinite(pot.tidal_tensor(x, t=0)))


@pytest.mark.parametrize("gamma", GAMMAS)
def test_hessian_matches_finite_difference_of_gradient(gamma: float) -> None:
    """`.hessian()` must match a finite-difference Jacobian of `.gradient()`."""
    pot = gp.gNFWPotential(m=1e12, r_s=1, gamma=gamma, units="galactic")
    x = jnp.array([20.0, 0.3, 0.1])
    h = np.asarray(pot.hessian(x, t=0))

    eps = 1e-4
    fd = np.zeros((3, 3))
    for i in range(3):
        gp_ = np.asarray(pot.gradient(x.at[i].add(eps), t=0))
        gm_ = np.asarray(pot.gradient(x.at[i].add(-eps), t=0))
        fd[i] = (gp_ - gm_) / (2 * eps)
    fd = (fd + fd.T) / 2  # symmetrize finite-difference noise

    assert np.allclose(h, fd, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("gamma", [0.3, 0.8, 1.5, 1.95])
def test_gradient_wrt_gamma_matches_finite_difference(gamma: float) -> None:
    """`jax.grad` w.r.t. `gamma` must match finite differences (was #819)."""

    def pot_fn(g: float) -> float:
        pot = gp.gNFWPotential(m=1e12, r_s=1, gamma=g, units="galactic")
        return pot.potential(jnp.array([5.0, 0.0, 0.0]), t=0)

    grad = float(jax.grad(pot_fn)(jnp.asarray(gamma)))
    eps = 1e-6
    fd = float((pot_fn(gamma + eps) - pot_fn(gamma - eps)) / (2 * eps))
    assert grad == pytest.approx(fd, rel=1e-4)


@pytest.mark.parametrize(
    ("a", "b", "z"),
    [(1.7, 0.0, 0.3), (1.7, 0.0, 0.95), (1.0, 1.3, 0.7), (1.0, 0.6, 0.95)],
)
def test_bz_from_hyp2f1_custom_jvp_matches_plain_autodiff(
    a: float, b: float, z: float
) -> None:
    """`Bz_from_hyp2f1`'s custom_jvp must agree with autodiff of the raw impl."""
    a_, b_, z_ = jnp.asarray(a), jnp.asarray(b), jnp.asarray(z)
    plain = jax.jit(_Bz_from_hyp2f1_impl)

    val_custom = Bz_from_hyp2f1(a_, b_, z_)
    val_plain = plain(a_, b_, z_)
    assert val_custom == pytest.approx(float(val_plain), rel=1e-10)

    grad_custom = jax.grad(Bz_from_hyp2f1, argnums=(0, 1, 2))(a_, b_, z_)
    grad_plain = jax.grad(plain, argnums=(0, 1, 2))(a_, b_, z_)
    for gc, gp_ in zip(grad_custom, grad_plain, strict=True):
        assert float(gc) == pytest.approx(float(gp_), abs=1e-6)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_bz_from_hyp2f1_batched_z_matches_looped(gamma: float) -> None:
    """`Bz_from_hyp2f1` must handle batched `z`.

    `_Bz0_taylor_series` summed over *all* axes instead of just the series
    index, and `_Bz0_log_series` summed `jax.lax.scan`'s stacked output over
    the wrong axis -- both silently assumed `z` was a scalar. This crashed
    (or, worse, could silently miscompute) any batched evaluation, e.g.
    `gNFWPotential.potential()` on more than one position at once.
    """
    a = jnp.asarray(3.0 - gamma)  # matches mass_enclosed's b=0 call pattern
    z = jnp.array([0.1, 0.3, 0.5, 0.6, 0.8, 0.95, 0.999])  # spans both series

    batched = Bz_from_hyp2f1(a, jnp.asarray(0.0), z)
    looped = jnp.array([Bz_from_hyp2f1(a, jnp.asarray(0.0), zi) for zi in z])
    assert jnp.allclose(batched, looped, atol=1e-10)


def test_potential_batched_positions_matches_looped() -> None:
    """`gNFWPotential.potential()` must handle multiple positions at once."""
    pot = gp.gNFWPotential(m=1e12, r_s=1, gamma=1.3, units="galactic")
    xs = jnp.array([[3.0, 0, 0], [5.0, 0, 0], [8.0, 0, 0], [20.0, 0, 0], [100.0, 0, 0]])

    batched = pot.potential(xs, t=0)
    looped = jnp.array([pot.potential(x, t=0) for x in xs])
    assert jnp.allclose(batched, looped, atol=1e-10)
