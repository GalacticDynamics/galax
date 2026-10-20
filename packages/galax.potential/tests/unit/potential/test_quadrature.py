"""Tests for the Gauss-Legendre quadrature helper."""

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.utils import gauss_legendre, gauss_legendre_nodes

ORDER = 50


@pytest.mark.parametrize("interval", [(-1.0, 1.0), (-1, 1)])
def test_reference_interval_is_exact(interval: tuple[float, float]) -> None:
    """``[-1, 1]`` must skip the affine map, which is not value-neutral.

    Parametrized over the ``float`` and ``int`` spellings: the guard compares
    tuples, and Python compares those element-wise and numerically, so
    ``(-1, 1)`` takes the same branch and hashes to the same cache entry.
    """
    x, w = gauss_legendre_nodes(ORDER, interval)
    x_ref, w_ref = np.polynomial.legendre.leggauss(ORDER)

    assert np.array_equal(np.asarray(x), x_ref)
    assert np.array_equal(np.asarray(w), w_ref)
    assert gauss_legendre_nodes(ORDER, interval) is gauss_legendre_nodes(
        ORDER, (-1.0, 1.0)
    )


def test_unit_interval_is_pinned() -> None:
    """``[0, 1]`` (the default) is bitwise the historical ``0.5 * (x + 1)``."""
    x, w = gauss_legendre_nodes(ORDER)
    x_ref, w_ref = np.polynomial.legendre.leggauss(ORDER)

    assert np.array_equal(np.asarray(x), 0.5 * (x_ref + 1))
    assert np.array_equal(np.asarray(w), 0.5 * w_ref)


def test_nodes_are_cached() -> None:
    """The nodes are constants: the same arrays come back."""
    assert gauss_legendre_nodes(ORDER) is gauss_legendre_nodes(ORDER)


def test_batched_quantity_integrand() -> None:
    """``f: (N,) -> (N, *batch)`` sums to ``(*batch,)``, bitwise as before."""
    batch = u.Q(np.array([1.0, 2.0, 3.0]), "m")

    def f(x):
        return batch * jnp.exp(-(x[:, None] ** 2))

    got = gauss_legendre(f, ORDER)

    x_ref, w_ref = np.polynomial.legendre.leggauss(ORDER)
    x, w = jnp.asarray(0.5 * (x_ref + 1)), jnp.asarray(0.5 * w_ref)
    expect = jnp.sum(f(x) * w[:, None], axis=0)

    assert got.shape == (3,)
    assert np.array_equal(np.asarray(got.ustrip("m")), np.asarray(expect.ustrip("m")))


@pytest.mark.parametrize(
    "pot",
    [
        gp.AxisymmetricGaussianPotential(
            m_tot=u.Q(1e12, "Msun"),
            r_s=u.Q(5.0, "kpc"),
            q2=u.Q(0.7, ""),
            units="galactic",
        ),
        gp.TriaxialGaussianPotential(
            m_tot=u.Q(1e12, "Msun"),
            r_s=u.Q(5.0, "kpc"),
            q1=u.Q(0.9, ""),
            q2=u.Q(0.7, ""),
            units="galactic",
        ),
        gp.TriaxialNFWPotential(
            m=u.Q(1e12, "Msun"),
            r_s=u.Q(5.0, "kpc"),
            q1=u.Q(0.9, ""),
            q2=u.Q(0.7, ""),
            units="galactic",
        ),
    ],
)
def test_no_quadrature_arrays_in_pytree(pot: gp.AbstractPotential) -> None:
    """The nodes & weights are constants, not pytree leaves."""
    assert all(np.ndim(leaf) == 0 for leaf in jax.tree.leaves(pot))
