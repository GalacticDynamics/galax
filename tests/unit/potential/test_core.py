from functools import partial
from typing import Any

import equinox as eqx
import jax
import pytest

import quaxed.array_api as xp
from unxt import Quantity

import galax.potential as gp
import galax.typing as gt
from .test_base import TestAbstractPotentialBase as AbstractPotentialBase_Test
from .test_utils import FieldUnitSystemMixin
from galax.potential._potential.base import default_constants
from galax.units import UnitSystem, galactic, unitsystem
from galax.utils import ImmutableDict


class TestAbstractPotential(AbstractPotentialBase_Test, FieldUnitSystemMixin):
    """Test the `galax.potential.AbstractPotentialBase` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.AbstractPotentialBase]:
        class TestPotential(gp.AbstractPotentialBase):
            m: gp.AbstractParameter = gp.ParameterField(
                dimensions="mass", default=Quantity(1e12, "Msun")
            )
            units: UnitSystem = eqx.field(
                default=galactic, converter=unitsystem, static=True
            )
            constants: ImmutableDict[Quantity] = eqx.field(
                default=default_constants, converter=ImmutableDict
            )

            @partial(jax.jit)
            def _potential_energy(  # TODO: inputs w/ units
                self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
            ) -> gt.BatchFloatQScalar:
                return (
                    self.constants["G"] * self.m(t) / xp.linalg.vector_norm(q, axis=-1)
                )

        return TestPotential

    @pytest.fixture(scope="class")
    def units(self) -> UnitSystem:
        return galactic

    @pytest.fixture(scope="class")
    def fields_(self, field_units: UnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    ###########################################################################

    def test_init(self) -> None:
        """Test the initialization of `AbstractPotentialBase`."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            gp.AbstractPotentialBase()

        # Test that the concrete class can be instantiated
        class TestPotential(gp.AbstractPotentialBase):
            m: gp.AbstractParameter = gp.ParameterField(
                dimensions="mass", default=Quantity(1e12, "Msun")
            )
            units: UnitSystem = eqx.field(
                default=galactic, static=True, converter=unitsystem
            )
            constants: ImmutableDict[Quantity] = eqx.field(
                default=default_constants, converter=ImmutableDict
            )

            @partial(jax.jit)
            def _potential_energy(  # TODO: inputs w/ units
                self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
            ) -> gt.BatchFloatQScalar:
                return (
                    self.constants["G"] * self.m(t) / xp.linalg.vector_norm(q, axis=-1)
                )

        pot = TestPotential()
        assert isinstance(pot, gp.AbstractPotentialBase)
