"""Test the SCF radial basis functions against a scalar reference."""

import numpy as np
import scipy.special as sps

import quaxed.numpy as jnp

from galax.potential.scf import phi_nl, rho_nl

SQRT_FOURPI = 3.544907701811031


def _ref_phi_nl(n: int, l: int, s: float) -> float:
    """Scalar transcription of ``phi_nl`` from gala's bfe_helper.cpp."""
    xi = (s - 1) / (s + 1)
    return (
        -SQRT_FOURPI
        * s**l
        * (1 + s) ** (-2 * l - 1)
        * sps.eval_gegenbauer(n, 2 * l + 1.5, xi)
    )


def _ref_rho_nl(n: int, l: int, s: float) -> float:
    """Scalar transcription of ``rho_nl`` from gala's bfe_helper.cpp."""
    xi = (s - 1) / (s + 1)
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    return (
        SQRT_FOURPI
        * (knl / (2 * np.pi))
        * s**l
        / (s * (1 + s) ** (2 * l + 3))
        * sps.eval_gegenbauer(n, 2 * l + 1.5, xi)
    )


def test_phi_nl_matches_scalar_reference() -> None:
    """Vectorized `phi_nl` equals the per-(n,l) scalar transcription."""
    nmax, lmax = 6, 4
    s = np.array([0.05, 0.5, 1.0, 3.0, 40.0])

    got = phi_nl(nmax, lmax, jnp.asarray(s))

    assert got.shape == (nmax + 1, lmax + 1, len(s))
    expected = np.array(
        [
            [[_ref_phi_nl(n, l, si) for si in s] for l in range(lmax + 1)]
            for n in range(nmax + 1)
        ]
    )
    assert jnp.allclose(got, expected, rtol=1e-10)


def test_rho_nl_matches_scalar_reference() -> None:
    """Vectorized `rho_nl` equals the per-(n,l) scalar transcription."""
    nmax, lmax = 6, 4
    s = np.array([0.05, 0.5, 1.0, 3.0, 40.0])

    got = rho_nl(nmax, lmax, jnp.asarray(s))

    assert got.shape == (nmax + 1, lmax + 1, len(s))
    expected = np.array(
        [
            [[_ref_rho_nl(n, l, si) for si in s] for l in range(lmax + 1)]
            for n in range(nmax + 1)
        ]
    )
    assert jnp.allclose(got, expected, rtol=1e-10)


def test_phi_00_is_hernquist_shape() -> None:
    """``phi_00(s) == -sqrt(4 pi) / (1 + s)``, the Hernquist anchor."""
    s = jnp.asarray([0.25, 1.0, 7.0])

    got = phi_nl(0, 0, s)

    assert jnp.allclose(got[0, 0], -SQRT_FOURPI / (1 + s), rtol=1e-12)
