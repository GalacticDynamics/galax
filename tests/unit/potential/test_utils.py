"""Tests for `galax.potential._potential.utils` package."""

from dataclasses import replace

import astropy.units as u
import pytest

from galax.potential._potential.utils import (
    UnitSystem,
    converter_to_usys,
    dimensionless,
    galactic,
    solarsystem,
)
from galax.utils._optional_deps import HAS_GALA


class TestConverterToUtils:
    """Tests for `galax.potential._potential.utils.converter_to_usys`."""

    def test_invalid(self):
        """Test conversion from unsupported value."""
        with pytest.raises(NotImplementedError):
            converter_to_usys(1234567890)

    def test_from_usys(self):
        """Test conversion from UnitSystem."""
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        assert converter_to_usys(usys) == usys

    def test_from_none(self):
        """Test conversion from None."""
        assert converter_to_usys(None) == dimensionless

    def test_from_args(self):
        """Test conversion from tuple."""
        value = UnitSystem(u.km, u.s, u.Msun, u.radian)
        assert converter_to_usys(value) == value

    def test_from_name(self):
        """Test conversion from named string."""
        assert converter_to_usys("dimensionless") == dimensionless
        assert converter_to_usys("solarsystem") == solarsystem
        assert converter_to_usys("galactic") == galactic

        with pytest.raises(NotImplementedError):
            converter_to_usys("invalid_value")

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    def test_from_gala(self):
        """Test conversion from gala."""
        # -------------------------------
        # UnitSystem
        from gala.units import UnitSystem as GalaUnitSystem

        value = GalaUnitSystem(u.km, u.s, u.Msun, u.radian)
        assert converter_to_usys(value) == UnitSystem(*value._core_units)

        # -------------------------------
        # DimensionlessUnitSystem
        from gala.units import DimensionlessUnitSystem as GalaDimensionlessUnitSystem

        value = GalaDimensionlessUnitSystem()
        assert converter_to_usys(value) == dimensionless


# ============================================================================


class FieldUnitSystemMixin:
    """Mixin for testing the ``units`` field on a ``Potential``."""

    @pytest.fixture()
    def fields_unitless(self, fields):
        """Fields with no units."""
        return {
            k: (v.value if isinstance(v, u.Quantity) else v) for k, v in fields.items()
        }

    # ===========================================

    def test_init_units_invalid(self, pot):
        """Test invalid unit system."""
        msg = "cannot convert 1234567890 to a UnitSystem"
        with pytest.raises(NotImplementedError, match=msg):
            replace(pot, units=1234567890)

    def test_init_units_from_usys(self, pot):
        """Test unit system from UnitSystem."""
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        assert replace(pot, units=usys).units == usys

    def test_init_units_from_args(self, pot_cls, fields_unitless):
        """Test unit system from None."""
        # strip the units from the fields otherwise the test will fail
        # because the units are not equal and we just want to check that
        # when the units aren't specified, the default is dimensionless
        # and a numeric value works.
        fields_unitless.pop("units")
        pot = pot_cls(**fields_unitless, units=None)
        assert pot.units == dimensionless

    def test_init_units_from_tuple(self, pot):
        """Test unit system from tuple."""
        units = (u.km, u.s, u.Msun, u.radian)
        assert replace(pot, units=units).units == UnitSystem(*units)

    def test_init_units_from_name(self, pot_cls, fields_unitless):
        """Test unit system from named string."""
        fields_unitless.pop("units")

        pot = pot_cls(**fields_unitless, units="dimensionless")
        assert pot.units == dimensionless

        pot = pot_cls(**fields_unitless, units="solarsystem")
        assert pot.units == solarsystem

        pot = pot_cls(**fields_unitless, units="galactic")
        assert pot.units == galactic

        msg = "cannot convert invalid_value to a UnitSystem"
        with pytest.raises(NotImplementedError, match=msg):
            pot_cls(**fields_unitless, units="invalid_value")
