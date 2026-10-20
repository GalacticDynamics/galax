"""Integration tests of `SCFPotential` against gala's SCF implementation."""

import astropy.units as apyu
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ..unit.potential.optional_deps import GSL_ENABLED, OptDeps


def _quadrupole() -> gp.SCFPotential:
    """Build an SCF potential with a non-zero m>0 term, exercising the phi path."""
    snlm = jnp.zeros((2, 3, 3)).at[0, 0, 0].set(1.0).at[0, 2, 2].set(0.1)
    tnlm = jnp.zeros((2, 3, 3)).at[0, 2, 1].set(0.05)
    return gp.SCFPotential(
        m_tot=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        Snlm=snlm,
        Tnlm=tnlm,
        units="galactic",
    )


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
@pytest.mark.skipif(not GSL_ENABLED, reason="requires gala built with GSL")
def test_roundtrip_through_gala() -> None:
    """Galax -> gala -> galax preserves the potential."""
    pot = _quadrupole()
    gala_pot = gp.io.convert_potential(gp.io.GalaLibrary, pot)
    back = gp.io.convert_potential(gp.io.GalaxLibrary, gala_pot)

    xyz = u.Q(np.array([[8.0, 1.0, 2.0], [-3.0, 4.0, 5.0]]), "kpc")
    t = u.Q(0.0, "Gyr")
    got = back.potential(xyz, t)
    expect = pot.potential(xyz, t)
    assert jnp.allclose(got, expect, rtol=1e-12, atol=u.Q(1e-14, expect.unit))


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
@pytest.mark.skipif(not GSL_ENABLED, reason="requires gala built with GSL")
def test_matches_gala_potential_and_density() -> None:
    """Galax and gala agree on the potential and the density."""
    pot = _quadrupole()
    gala_pot = gp.io.convert_potential(gp.io.GalaLibrary, pot)
    xyz = np.array([[8.0, 1.0, 2.0], [-3.0, 4.0, 5.0]]).T * apyu.kpc

    assert np.allclose(
        np.asarray(pot.potential(u.Q(xyz.T.value, "kpc"), u.Q(0.0, "Gyr")).value),
        gala_pot.energy(xyz).to_value("kpc2 / Myr2"),
        rtol=1e-10,
    )
    assert np.allclose(
        np.asarray(pot.density(u.Q(xyz.T.value, "kpc"), u.Q(0.0, "Gyr")).value),
        gala_pot.density(xyz).to_value("Msun / kpc3"),
        rtol=1e-10,
    )
