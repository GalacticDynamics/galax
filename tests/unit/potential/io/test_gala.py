"""Testing the gala potential I/O module."""

from inspect import get_annotations
from typing import ClassVar

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from coordinax import Cartesian3DVector

import galax.potential as gp
from galax.potential._potential.frame import PotentialFrame
from galax.typing import QVec3
from galax.utils._optional_deps import HAS_GALA

if HAS_GALA:
    from galax.potential._potential.io._gala import _GALA_TO_GALAX_REGISTRY
else:
    from galax.potential._potential.io._gala_noop import _GALA_TO_GALAX_REGISTRY


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
        self, pot: gp.AbstractPotentialBase, x: QVec3
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


@pytest.mark.skipif(not HAS_GALA, reason="requires gala")
def test_offset_hernquist() -> None:
    """Test gala potential with an offset Hernquist potential."""
    from gala.potential import HernquistPotential as GalaHernquistPotential
    from gala.units import galactic

    gpot = GalaHernquistPotential(m=1e12, c=5, units=galactic, origin=[1.0, 2, 3])
    gxpot = gp.io.gala_to_galax(gpot)

    assert isinstance(gxpot, PotentialFrame)
    assert gxpot.operator[0].translation == Cartesian3DVector.constructor(
        [1.0, 2, 3] * u.kpc
    )

    assert isinstance(gxpot.potential, gp.HernquistPotential)
    assert gxpot.units._core_units == galactic._core_units
