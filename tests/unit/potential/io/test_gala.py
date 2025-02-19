"""Testing the gala potential I/O module."""

from typing import ClassVar

import astropy.units as apyu
import pytest
from plum import convert

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from galax._interop.optional_deps import OptDeps

parametrize_test_method_gala = pytest.mark.parametrize(
    ("method0", "method1", "atol"),
    [
        ("potential", "energy", 1e-8),
        ("gradient", "gradient", 1e-8),
        ("density", "density", 1e-8),
        ("hessian", "hessian", 1e-8),
    ],
)


class GalaIOMixin:
    """Mixin for testing gala potential I/O.

    This is mixed into the ``TestAbstractPotential`` class.
    """

    HAS_GALA_COUNTERPART: ClassVar[bool] = True

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractPotential, x: gt.QuSz3
    ) -> None:
        """Test roundtripping ``gala_to_galax(galax_to_gala())``."""
        # First we need to check that the potential is gala-compatible
        if not self.HAS_GALA_COUNTERPART:
            pytest.skip("potential does not have a gala counterpart")

        gala_pot = gp.io.convert_potential(gp.io.GalaLibrary, pot)
        rpot = gp.io.convert_potential(gp.io.GalaxLibrary, gala_pot)

        # quick test that the potential energies are the same
        got = rpot(x, 0)
        exp = pot(x, 0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-14, exp.unit))

        # TODO: add more robust tests

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    @parametrize_test_method_gala
    def test_method_gala(
        self,
        pot: gp.AbstractPotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        # First we need to check that the potential is gala-compatible
        if not self.HAS_GALA_COUNTERPART:
            pytest.skip("potential does not have a gala counterpart")

        galax = convert(getattr(pot, method0)(x, t=0), u.Quantity)
        galap = gp.io.convert_potential(gp.io.GalaLibrary, pot)
        gala = getattr(galap, method1)(convert(x, apyu.Quantity), t=0 * apyu.Myr)
        assert jnp.allclose(
            jnp.ravel(galax),
            jnp.ravel(convert(gala, u.Quantity)),
            atol=u.Quantity(atol, galax.unit),
        )


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
def test_offset_hernquist() -> None:
    """Test gala potential with an offset Hernquist potential."""
    from gala.potential import HernquistPotential as GalaHernquistPotential
    from gala.units import galactic

    gpot = GalaHernquistPotential(m=1e12, c=5, units=galactic, origin=[1.0, 2, 3])
    gxpot = gp.io.convert_potential(gp.io.GalaxLibrary, gpot)

    assert isinstance(gxpot, gp.TransformedPotential)
    assert gxpot.xop.translation == cx.CartesianPos3D.from_([1.0, 2, 3], "kpc")

    assert isinstance(gxpot.original_potential, gp.HernquistPotential)
    assert set(gxpot.units.base_units) == set(galactic._core_units)
