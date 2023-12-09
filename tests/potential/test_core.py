from typing import Any

import equinox as eqx
import jax.numpy as xp
import pytest

import galdynamix.potential as gp
from galdynamix.potential._potential.utils import converter_to_usys
from galdynamix.typing import BatchableFloatOrIntScalarLike, BatchFloatScalar, BatchVec3
from galdynamix.units import UnitSystem, galactic
from galdynamix.utils import partial_jit, vectorize_method

from .test_base import TestAbstractPotentialBase


class TestAbstractPotential(TestAbstractPotentialBase):
    """Test the `galdynamix.potential.AbstractPotentialBase` class."""

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
