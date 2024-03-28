"""Tests for `galax.potential._potential.utils` package."""

import re
from dataclasses import replace
from typing import Any

import astropy.units as u
import pytest
from jax import Array
from plum import NotFoundLookupError

from unxt import UnitSystem, unitsystem
from unxt.unitsystems import dimensionless, galactic, solarsystem

from galax.potential import AbstractPotentialBase
from galax.utils._optional_deps import HAS_GALA


class TestConverterToUtils:
    """Tests for `galax.potential._potential.utils.unitsystem`."""

    def test_invalid(self) -> None:
        """Test conversion from unsupported value."""
        msg = "`unitsystem(1234567890)` could not be resolved."
        with pytest.raises(NotFoundLookupError, match=re.escape(msg)):
            unitsystem(1234567890)

    def test_from_usys(self) -> None:
        """Test conversion from UnitSystem."""
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        assert unitsystem(usys) == usys

    def test_from_none(self) -> None:
        """Test conversion from None."""
        assert unitsystem(None) == dimensionless

    def test_from_args(self) -> None:
        """Test conversion from tuple."""
        value = UnitSystem(u.km, u.s, u.Msun, u.radian)
        assert unitsystem(value) == value

    def test_from_name(self) -> None:
        """Test conversion from named string."""
        assert unitsystem("dimensionless") == dimensionless
        assert unitsystem("solarsystem") == solarsystem
        assert unitsystem("galactic") == galactic

        with pytest.raises(KeyError, match="invalid_value"):
            unitsystem("invalid_value")

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    def test_from_gala(self) -> None:
        """Test conversion from gala."""
        # -------------------------------
        # UnitSystem
        from gala.units import UnitSystem as GalaUnitSystem

        value = GalaUnitSystem(u.km, u.s, u.Msun, u.radian)
        assert unitsystem(value) == UnitSystem(*value._core_units)

        # -------------------------------
        # DimensionlessUnitSystem
        from gala.units import DimensionlessUnitSystem as GalaDimensionlessUnitSystem

        value = GalaDimensionlessUnitSystem()
        assert unitsystem(value) == dimensionless


# ============================================================================


class FieldUnitSystemMixin:
    """Mixin for testing the ``units`` field on a ``Potential``."""

    @pytest.fixture()
    def fields_unitless(self, fields: dict[str, Any]) -> dict[str, Array]:
        """Fields with no units."""
        return {
            k: (v.value if isinstance(v, u.Quantity) else v) for k, v in fields.items()
        }

    # ===========================================

    def test_init_units_invalid(self, pot: AbstractPotentialBase) -> None:
        """Test invalid unit system."""
        msg = "`unitsystem(1234567890)` could not be resolved."
        with pytest.raises(NotFoundLookupError, match=re.escape(msg)):
            replace(pot, units=1234567890)

    def test_init_units_from_usys(self, pot: AbstractPotentialBase) -> None:
        """Test unit system from UnitSystem."""
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        assert replace(pot, units=usys).units == usys

    # TODO: sort this out
    # def test_init_units_from_args(
    #     self, pot_cls: type[AbstractPotentialBase], fields_unitless: dict[str, Array]
    # ) -> None:
    #     """Test unit system from None."""
    #     # strip the units from the fields otherwise the test will fail
    #     # because the units are not equal and we just want to check that
    #     # when the units aren't specified, the default is dimensionless
    #     # and a numeric value works.
    #     fields_unitless.pop("units", None)
    #     pot = pot_cls(**fields_unitless, units=None)
    #     assert pot.units == dimensionless

    def test_init_units_from_tuple(self, pot: AbstractPotentialBase) -> None:
        """Test unit system from tuple."""
        units = (u.km, u.s, u.Msun, u.radian)
        assert replace(pot, units=units).units == UnitSystem(*units)

    def test_init_units_from_name(
        self, pot_cls: type[AbstractPotentialBase], fields_unitless: dict[str, Array]
    ) -> None:
        """Test unit system from named string."""
        fields_unitless.pop("units")

        # TODO: sort this out
        # pot = pot_cls(**fields_unitless, units="dimensionless")
        # # assert pot.units == dimensionless

        pot = pot_cls(**fields_unitless, units="solarsystem")
        assert pot.units == solarsystem

        pot = pot_cls(**fields_unitless, units="galactic")
        assert pot.units == galactic

        with pytest.raises(KeyError, match="invalid_value"):
            pot_cls(**fields_unitless, units="invalid_value")
