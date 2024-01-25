from dataclasses import field
from typing import Any

import equinox as eqx
import jax.experimental.array_api as xp
import pytest

import galax.potential as gp
from galax.potential._potential.utils import converter_to_usys
from galax.typing import BatchableFloatOrIntScalarLike, BatchFloatScalar, BatchVec3
from galax.units import UnitSystem, dimensionless, galactic
from galax.utils import partial_jit, vectorize_method

from .test_base import TestAbstractPotentialBase as AbstractPotentialBase_Test
from .test_utils import FieldUnitSystemMixin


class TestAbstractPotential(AbstractPotentialBase_Test, FieldUnitSystemMixin):
    """Test the `galax.potential.AbstractPotentialBase` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.AbstractPotentialBase]:
        class TestPotential(gp.AbstractPotentialBase):
            units: UnitSystem = eqx.field(
                default=None, converter=converter_to_usys, static=True
            )
            _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

            def __post_init__(self):
                object.__setattr__(self, "_G", 1.0)

            @partial_jit()
            @vectorize_method(signature="(3),()->()")
            def _potential_energy(
                self, q: BatchVec3, t: BatchableFloatOrIntScalarLike
            ) -> BatchFloatScalar:
                return xp.sum(q, axis=-1)

        return TestPotential

    @pytest.fixture(scope="class")
    def field_units(self) -> dict[str, Any]:
        return galactic

    @pytest.fixture(scope="class")
    def fields_(self, field_units) -> dict[str, Any]:
        return {"units": field_units}

    ###########################################################################

    def test_init(self):
        """Test the initialization of `AbstractPotentialBase`."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            gp.AbstractPotentialBase()

        # Test that the concrete class can be instantiated
        class TestPotential(gp.AbstractPotentialBase):
            units: UnitSystem = field(default_factory=lambda: dimensionless)

            def _potential_energy(self, q, t):
                return xp.sum(q, axis=-1)

        pot = TestPotential()
        assert isinstance(pot, gp.AbstractPotentialBase)
