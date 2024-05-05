"""Testing the gala potential I/O module."""

from inspect import get_annotations
from typing import ClassVar

import astropy.units as u
import pytest
from plum import convert

import coordinax as cx
import quaxed.numpy as qnp
from unxt import Quantity

import galax.potential as gp
import galax.typing as gt
from galax.utils._optional_deps import HAS_GALA

if HAS_GALA:
    from galax.potential._potential.io._gala import _GALA_TO_GALAX_REGISTRY
else:
    _GALA_TO_GALAX_REGISTRY = {}


parametrize_test_method_gala = pytest.mark.parametrize(
    ("method0", "method1", "atol"),
    [
        ("potential_energy", "energy", 1e-8),
        ("gradient", "gradient", 1e-8),
        ("density", "density", 1e-8),
        ("hessian", "hessian", 1e-8),
    ],
)


class GalaIOMixin:
    """Mixin for testing gala potential I/O.

    This is mixed into the ``TestAbstractPotentialBase`` class.
    """

    # All the Gala-mapped potentials
    _GALA_CAN_MAP_TO: ClassVar = set(
        [  # get from GALA_TO_GALAX_REGISTRY or the single-dispatch registry
            _GALA_TO_GALAX_REGISTRY.get(pot, get_annotations(func)["return"])
            for pot, func in gp.io.gala_to_galax.registry.items()
        ]
        if HAS_GALA
        else []
    )

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractPotentialBase, x: gt.QVec3
    ) -> None:
        """Test roundtripping ``gala_to_galax(galax_to_gala())``."""
        from .gala_helper import galax_to_gala

        # First we need to check that the potential is gala-compatible
        if type(pot) not in self._GALA_CAN_MAP_TO:
            pytest.skip(f"potential {pot} cannot be mapped to from gala")

        # TODO: a more robust test
        rpot = gp.io.gala_to_galax(galax_to_gala(pot))

        # quick test that the potential energies are the same
        assert qnp.array_equal(pot(x, t=0), rpot(x, t=0))

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    @parametrize_test_method_gala
    def test_method_gala(
        self,
        pot: gp.AbstractPotentialBase,
        method0: str,
        method1: str,
        x: gt.QVec3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        from ..io.gala_helper import galax_to_gala

        if type(pot) not in self._GALA_CAN_MAP_TO:
            pytest.skip(f"potential {pot} cannot be mapped to gala")

        galax = getattr(pot, method0)(x, t=0)
        gala = getattr(galax_to_gala(pot), method1)(convert(x, u.Quantity), t=0 * u.Myr)
        assert qnp.allclose(
            qnp.ravel(galax),
            qnp.ravel(convert(gala, Quantity)),
            atol=Quantity(atol, galax.unit),
        )


@pytest.mark.skipif(not HAS_GALA, reason="requires gala")
def test_offset_hernquist() -> None:
    """Test gala potential with an offset Hernquist potential."""
    from gala.potential import HernquistPotential as GalaHernquistPotential
    from gala.units import galactic

    gpot = GalaHernquistPotential(m=1e12, c=5, units=galactic, origin=[1.0, 2, 3])
    gxpot = gp.io.gala_to_galax(gpot)

    assert isinstance(gxpot, gp.PotentialFrame)
    assert gxpot.operator[0].translation == cx.Cartesian3DVector.constructor(
        [1.0, 2, 3] * u.kpc
    )

    assert isinstance(gxpot.potential, gp.HernquistPotential)
    assert gxpot.units._core_units == galactic._core_units
