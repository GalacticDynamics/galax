"""Tests for the softening kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from galax.nbody._src.kernels import (
    DehnenK1Kernel,
    DehnenK2Kernel,
    NewtonianKernel,
    PlummerKernel,
    SplineKernel,
)


def test_force_is_minus_grad_potential() -> None:
    """The pair acceleration -K x equals minus the gradient of the potential."""
    k = PlummerKernel()
    x = jnp.array([0.3, -0.2, 0.5])
    h = jnp.asarray(0.1)

    def phi(x):
        return k.potential(jnp.sum(x**2), h)

    np.testing.assert_allclose(
        -jax.grad(phi)(x), -k.force_factor(jnp.sum(x**2), h) * x, rtol=1e-12
    )


def test_plummer_values() -> None:
    """Check the Plummer kernel at r^2 = 3, h = 1."""
    k = PlummerKernel()
    assert k.force_factor(jnp.array(3.0), jnp.array(1.0)) == pytest.approx(4.0**-1.5)
    assert k.potential(jnp.array(3.0), jnp.array(1.0)) == pytest.approx(-0.5)


def test_combine_is_max() -> None:
    """Pair softening is the larger of the two."""
    assert PlummerKernel().combine(jnp.array(0.1), jnp.array(0.3)) == 0.3


ALL = [
    NewtonianKernel(),
    PlummerKernel(),
    DehnenK1Kernel(),
    DehnenK2Kernel(),
    SplineKernel(),
]
# r^2 values spanning the spline's q <= 0.5, 0.5 < q < 1 and q >= 1 branches at h = 1.
R2 = jnp.array([0.01, 0.09, 0.2, 0.36, 0.64, 0.9, 1.44, 4.0])


@pytest.mark.parametrize("k", ALL, ids=lambda k: type(k).__name__)
def test_force_factor_is_2_dphi_dr2(k) -> None:
    h = jnp.ones_like(R2)
    dphi = jax.vmap(jax.grad(k.potential))(R2, h)
    np.testing.assert_allclose(k.force_factor(R2, h), 2 * dphi, rtol=1e-12)


@pytest.mark.parametrize("k", ALL[1:], ids=lambda k: type(k).__name__)
def test_newtonian_far_away(k) -> None:
    r2, h = jnp.asarray(1e6), jnp.asarray(1e-2)
    np.testing.assert_allclose(k.force_factor(r2, h), r2**-1.5, rtol=1e-6)
    np.testing.assert_allclose(k.potential(r2, h), -(r2**-0.5), rtol=1e-6)


@pytest.mark.parametrize("q", [0.5, 1.0])
def test_spline_continuous(q) -> None:
    k, h, e = SplineKernel(), jnp.asarray(1.0), 1e-9
    for fn in (k.force_factor, k.potential):
        lo, hi = fn(jnp.asarray((q - e) ** 2), h), fn(jnp.asarray((q + e) ** 2), h)
        np.testing.assert_allclose(lo, hi, rtol=1e-7)


def test_spline_center_and_zero_softening() -> None:
    k, h = SplineKernel(), jnp.asarray(2.0)
    assert k.potential(jnp.asarray(0.0), h) == pytest.approx(-2.8 / 2.0)
    assert k.force_factor(jnp.asarray(0.0), h) == pytest.approx(32 / 3 / 8.0)
    assert jnp.isfinite(jax.grad(k.force_factor)(jnp.asarray(0.0), h))
    r2 = jnp.asarray(0.25)
    assert k.force_factor(r2, jnp.asarray(0.0)) == pytest.approx(8.0)
