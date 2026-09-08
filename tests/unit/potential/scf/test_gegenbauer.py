"""Test the Gegenbauer polynomial recurrence."""

import numpy as np
import scipy.special as sps

import quaxed.numpy as jnp

from galax.potential.scf import gegenbauer_all


def _scipy_reference(nmax: int, alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate ``C_n^alpha(x)`` with scipy, for every ``n`` and ``alpha``."""
    return np.array(
        [[sps.eval_gegenbauer(n, a, x) for a in alpha] for n in range(nmax + 1)]
    )


def test_matches_scipy() -> None:
    """`gegenbauer_all` agrees with `scipy.special.eval_gegenbauer`."""
    nmax = 8
    alpha = np.array([1.5, 3.5, 5.5, 7.5])
    x = np.array([-0.95, -0.3, 0.0, 0.4, 0.99])

    got = gegenbauer_all(nmax, jnp.asarray(alpha), jnp.asarray(x))

    assert got.shape == (nmax + 1, len(alpha), len(x))
    assert jnp.allclose(got, _scipy_reference(nmax, alpha, x), atol=1e-10)


def test_nmax_zero() -> None:
    """``C_0^alpha(x) == 1`` for every alpha and x."""
    got = gegenbauer_all(0, jnp.asarray([1.5, 4.5]), jnp.asarray([-0.5, 0.0, 0.7]))

    assert got.shape == (1, 2, 3)
    assert jnp.allclose(got, 1.0)


def test_nmax_one() -> None:
    """``C_1^alpha(x) == 2 * alpha * x``, the recurrence seed."""
    alpha, x = jnp.asarray([1.5, 4.5]), jnp.asarray([-0.5, 0.0, 0.7])

    got = gegenbauer_all(1, alpha, x)

    assert jnp.allclose(got[1], 2 * alpha[:, None] * x[None, :])


def test_scalar_x() -> None:
    """A scalar ``x`` yields no trailing batch axis."""
    nmax, alpha = 5, np.array([1.5, 3.5])

    got = gegenbauer_all(nmax, jnp.asarray(alpha), jnp.asarray(0.25))

    assert got.shape == (nmax + 1, len(alpha))
    expected = _scipy_reference(nmax, alpha, np.array(0.25))
    assert jnp.allclose(got, expected, atol=1e-10)
