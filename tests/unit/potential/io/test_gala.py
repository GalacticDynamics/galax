"""Testing the gala potential I/O module."""

from inspect import get_annotations
from typing import ClassVar

import jax.numpy as jnp
import pytest
from quax import quaxify

import galax.potential as gp
from galax.potential._potential.io.gala import _GALA_TO_GALAX_REGISTRY
from galax.typing import Vec3
from galax.utils._optional_deps import HAS_GALA

array_equal = quaxify(jnp.array_equal)


class GalaIOMixin:
    """Mixin for testing gala potential I/O.

    This is mixed into the ``TestAbstractPotentialBase`` class.
    """

    # All the Gala-mapped potentials
    _GALA_CAN_MAP_TO: ClassVar = set(
        [
            _GALA_TO_GALAX_REGISTRY.get(pot, get_annotations(func)["return"])
            for pot, func in gp.io.gala_to_galax.registry.items()
        ]
        if HAS_GALA
        else []
    )

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractPotentialBase, x: Vec3
    ) -> None:
        """Test roundtripping ``gala_to_galax(galax_to_gala())``."""
        from .gala_helper import galax_to_gala

        # First we need to check that the potential is gala-compatible
        if type(pot) not in self._GALA_CAN_MAP_TO:
            pytest.skip(f"potential {pot} cannot be mapped to from gala")

        # TODO: a more robust test
        rpot = gp.io.gala_to_galax(galax_to_gala(pot))

        # quick test that the potential energies are the same
        assert array_equal(pot(x, t=0), rpot(x, t=0))
