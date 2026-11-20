"""Tests for the experimental spray release schedule."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from galax.dynamics._src.experimental.schedule import Uniform

EPS = np.finfo(np.float64).eps


def test_lattice_ends() -> None:
    """The lattice ends exactly at ``t1`` and starts at ``t1 - t_age``."""
    s = Uniform(jnp.asarray(1234.5), 7, 3, jnp.asarray(2.0))
    lat = s.lattice()
    assert lat.shape == (s.n_steps() + 1,)
    assert lat[-1] == 2.0
    np.testing.assert_allclose(lat[0], 2.0 - 1234.5, rtol=EPS, atol=0)


@pytest.mark.parametrize("steps", [1, 4])
def test_releases_on_lattice(steps: int) -> None:
    """Release times are lattice points, evenly spaced from ``t1 - t_age``."""
    s = Uniform(jnp.asarray(1000.0), 6, steps, jnp.asarray(10.0))
    rel = s.release_times()
    assert rel.shape == (6,)
    assert np.array_equal(rel, s.lattice()[s.release_index()])
    np.testing.assert_allclose(
        rel, jnp.linspace(10.0 - 1000.0, 10.0, 6), rtol=0, atol=2 * EPS * 1000.0
    )


def test_grad_t_age_smooth() -> None:
    """The lattice's gradient in ``t_age`` has no jumps."""
    g = jax.grad(lambda a: Uniform(a, 5, 4).lattice().sum())
    a = jnp.asarray(300.0)
    assert abs(g(a + 1e-9) - g(a)) < 1e-9 * abs(g(a))


def test_one_and_two_releases() -> None:
    """One release is one step; two releases are ``steps_per_release`` steps."""
    one = Uniform(jnp.asarray(50.0), 1, 4, jnp.asarray(0.0))
    assert one.lattice().shape == (2,)
    np.testing.assert_array_equal(one.lattice(), [-50.0, 0.0])
    np.testing.assert_array_equal(one.release_times(), [-50.0])

    two = Uniform(jnp.asarray(50.0), 2, 4, jnp.asarray(0.0))
    assert two.lattice().shape == (5,)
    np.testing.assert_array_equal(two.release_index(), [0, 4])


def test_traced_t_age_vmap() -> None:
    """A traced ``t_age`` vmaps."""
    ages = jnp.array([100.0, 200.0])
    batched = jax.vmap(lambda a: Uniform(a, 4, 3).lattice())(ages)
    for i, a in enumerate(ages):
        np.testing.assert_array_equal(batched[i], Uniform(a, 4, 3).lattice())
