"""Tests for the softening kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from galax.nbody._src.kernels import PlummerKernel


def test_force_is_minus_grad_potential() -> None:
    """The pair acceleration -K x equals minus the gradient of the potential."""
    k = PlummerKernel()
    x = jnp.array([0.3, -0.2, 0.5])
    h = 0.1

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
