"""Test the `SCFPotential` class."""

import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp


def _monopole(m_tot: float = 1e12, r_s: float = 10.0) -> gp.SCFPotential:
    """Build an SCF potential whose only term is the n=l=m=0 monopole."""
    snlm = jnp.zeros((1, 1, 1)).at[0, 0, 0].set(1.0)
    return gp.SCFPotential(
        m_tot=u.Q(m_tot, "Msun"),
        r_s=u.Q(r_s, "kpc"),
        Snlm=snlm,
        Tnlm=jnp.zeros_like(snlm),
        units="galactic",
    )


def test_nmax_lmax_derived_from_coefficients() -> None:
    """`nmax` and `lmax` come from the coefficient array shape."""
    snlm = jnp.zeros((4, 3, 3))
    pot = gp.SCFPotential(
        m_tot=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        Snlm=snlm,
        Tnlm=jnp.zeros_like(snlm),
        units="galactic",
    )

    assert pot.nmax == 3
    assert pot.lmax == 2


def test_mismatched_coefficient_shapes_raise() -> None:
    """`Snlm` and `Tnlm` must share a shape."""
    with pytest.raises(Exception, match="same shape"):
        gp.SCFPotential(
            m_tot=u.Q(1e12, "Msun"),
            r_s=u.Q(10.0, "kpc"),
            Snlm=jnp.zeros((2, 2, 2)),
            Tnlm=jnp.zeros((3, 2, 2)),
            units="galactic",
        )


def test_monopole_is_hernquist_potential() -> None:
    """nmax=lmax=0 with S000=1 reproduces `HernquistPotential` exactly."""
    scf = _monopole()
    hern = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10.0, "kpc"), units="galactic"
    )
    xyz = u.Q(np.array([[1.0, 2.0, 3.0], [-8.0, 0.5, 4.0], [0.1, 0.0, 0.0]]), "kpc")
    t = u.Q(0.0, "Gyr")

    got = scf.potential(xyz, t)
    expect = hern.potential(xyz, t)
    assert jnp.allclose(got, expect, rtol=1e-12, atol=u.Q(1e-8, expect.unit))


def test_scalar_input_gives_scalar_output() -> None:
    """A single position returns a scalar, not a length-1 array."""
    got = _monopole().potential(u.Q(np.array([1.0, 2.0, 3.0]), "kpc"), u.Q(0.0, "Gyr"))

    assert got.shape == ()
