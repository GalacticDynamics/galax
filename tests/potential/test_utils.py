"""Tests for `galdynamix.potential._potential.utils` package."""

import astropy.units as u
import pytest

from galdynamix.potential._potential.utils import (
    UnitSystem,
    converter_to_usys,
    dimensionless,
    galactic,
    solarsystem,
)


class TestConverterToUtils:
    """Tests for `galdynamix.potential._potential.utils.converter_to_usys`."""

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


# ============================================================================


class FieldUnitSystemMixin:
    """Mixin for testing the ``units`` field on a ``Potential``."""

    def test_init_units_invalid(self, pot_cls, fields):
        """Test invalid unit system."""
        fields.pop("units")
        msg = "cannot convert 1234567890 to a UnitSystem"
        with pytest.raises(NotImplementedError, match=msg):
            pot_cls(**fields, units=1234567890)

    def test_init_units_from_usys(self, pot_cls, fields):
        """Test unit system from UnitSystem."""
        fields.pop("units")
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        pot = pot_cls(**fields, units=usys)
        assert pot.units == usys

    def test_init_units_from_args(self, pot_cls, fields):
        """Test unit system from None."""
        fields.pop("units")
        pot = pot_cls(**fields, units=None)
        assert pot.units == dimensionless

    def test_init_units_from_tuple(self, pot_cls, fields):
        """Test unit system from tuple."""
        fields.pop("units")
        pot = pot_cls(**fields, units=(u.km, u.s, u.Msun, u.radian))
        assert pot.units == UnitSystem(u.km, u.s, u.Msun, u.radian)

    def test_init_units_from_name(self, pot_cls, fields):
        """Test unit system from named string."""
        fields.pop("units")

        pot = pot_cls(**fields, units="dimensionless")
        assert pot.units == dimensionless

        pot = pot_cls(**fields, units="solarsystem")
        assert pot.units == solarsystem

        pot = pot_cls(**fields, units="galactic")
        assert pot.units == galactic

        msg = "cannot convert invalid_value to a UnitSystem"
        with pytest.raises(NotImplementedError, match=msg):
            pot_cls(**fields, units="invalid_value")
