import re
from typing import Any

import pytest
from jaxtyping import Array
from plum import NotFoundLookupError
from typing_extensions import override

import quaxed.numpy as qnp
from unxt import Quantity

import galax.potential as gp
import galax.typing as gt
import galax.units as gu
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from galax.units import UnitSystem, dimensionless


class TestNullPotential(AbstractPotential_Test):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.NullPotential]:
        return gp.NullPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_units: UnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    # ==========================================================================

    def test_init_units_from_args(
        self, pot_cls: type[gp.AbstractPotentialBase], fields_unitless: dict[str, Array]
    ) -> None:
        """Test unit system from None."""
        # strip the units from the fields otherwise the test will fail
        # because the units are not equal and we just want to check that
        # when the units aren't specified, the default is dimensionless
        # and a numeric value works.
        fields_unitless.pop("units", None)
        pot = pot_cls(**fields_unitless, units=None)
        assert pot.units == dimensionless

    @override
    def test_init_units_from_name(
        self, pot_cls: type[gp.AbstractPotentialBase], fields_unitless: dict[str, Array]
    ) -> None:
        """Test unit system from named string."""
        fields_unitless.pop("units")

        pot = pot_cls(**fields_unitless, units="dimensionless")
        assert pot.units == gu.dimensionless

        pot = pot_cls(**fields_unitless, units="solarsystem")
        assert pot.units == gu.solarsystem

        pot = pot_cls(**fields_unitless, units="galactic")
        assert pot.units == gu.galactic

        msg = "`unitsystem('invalid_value')` could not be resolved."
        with pytest.raises(NotFoundLookupError, match=re.escape(msg)):
            pot_cls(**fields_unitless, units="invalid_value")

    # ==========================================================================

    def test_potential_energy(self, pot: gp.NullPotential, x: gt.Vec3) -> None:
        """Test :meth:`NullPotential.potential_energy`."""
        expected = Quantity(0.0, pot.units["specific energy"])
        assert qnp.isclose(  # TODO: .value & use pytest-arraydiff
            pot.potential_energy(x, t=0).decompose(pot.units).value, expected.value
        )

    def test_gradient(self, pot: gp.NullPotential, x: gt.Vec3) -> None:
        """Test :meth:`NullPotential.gradient`."""
        expected = Quantity([0.0, 0.0, 0.0], pot.units["acceleration"])
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.gradient(x, t=0).decompose(pot.units).value, expected.value
        )

    def test_density(self, pot: gp.NullPotential, x: gt.Vec3) -> None:
        """Test :meth:`NullPotential.density`."""
        expected = Quantity(0.0, pot.units["mass density"])
        assert qnp.isclose(  # TODO: .value & use pytest-arraydiff
            pot.density(x, t=0).decompose(pot.units).value, expected.value
        )

    def test_hessian(self, pot: gp.NullPotential, x: gt.Vec3) -> None:
        """Test :meth:`NullPotential.hessian`."""
        expected = Quantity(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "1/Myr2"
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.hessian(x, t=0).decompose(pot.units).value, expected.value
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotentialBase, x: gt.Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expected = Quantity(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "1/Myr2"
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.tidal_tensor(x, t=0).decompose(pot.units).value, expected.value
        )
