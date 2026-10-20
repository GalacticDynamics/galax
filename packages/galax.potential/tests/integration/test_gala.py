"""Integration tests of the gala potential I/O: gala -> galax conversion."""

import pytest

import coordinax as cx

import galax.potential as gp
from ..unit.potential.optional_deps import OptDeps


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
def test_offset_hernquist() -> None:
    """Test gala potential with an offset Hernquist potential."""
    # Deferred: gala is optional, and this module is collected by the CI job
    # that installs no extras.
    from gala.potential import HernquistPotential as GalaHernquistPotential
    from gala.units import galactic

    gpot = GalaHernquistPotential(m=1e12, c=5, units=galactic, origin=[1.0, 2, 3])
    gxpot = gp.io.convert_potential(gp.io.GalaxLibrary, gpot)

    assert isinstance(gxpot, gp.TransformedPotential)
    assert gxpot.xop.delta_q == cx.CartesianPos3D.from_([1.0, 2, 3], "kpc")

    assert isinstance(gxpot.base_potential, gp.HernquistPotential)
    assert set(gxpot.units.base_units) == set(galactic._core_units)
