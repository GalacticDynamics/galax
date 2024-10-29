"""Tests for `galax.potential._src.utils` package."""

from dataclasses import replace
from typing import Any

import astropy.units as u
import pytest
from jax import Array

from unxt import unitsystem
from unxt.unitsystems import galactic, solarsystem

from galax.potential import AbstractBasePotential


class FieldUnitSystemMixin:
    """Mixin for testing the ``units`` field on a ``Potential``."""

    @pytest.fixture
    def fields_unitless(self, fields: dict[str, Any]) -> dict[str, Array]:
        """Fields with no units."""
        return {
            k: (v.value if isinstance(v, u.Quantity) else v) for k, v in fields.items()
        }

    # ===========================================

    def test_init_units_from_usys(self, pot: AbstractBasePotential) -> None:
        """Test unit system from UnitSystem."""
        usys = unitsystem(u.km, u.s, u.Msun, u.radian)
        assert replace(pot, units=usys).units == usys

    # TODO: sort this out
    # def test_init_units_from_args(
    #     self, pot_cls: type[AbstractBasePotential], fields_unitless: dict[str, Array]
    # ) -> None:
    #     """Test unit system from None."""
    #     # strip the units from the fields otherwise the test will fail
    #     # because the units are not equal and we just want to check that
    #     # when the units aren't specified, the default is dimensionless
    #     # and a numeric value works.
    #     fields_unitless.pop("units", None)
    #     pot = pot_cls(**fields_unitless, units=None)
    #     assert pot.units == dimensionless

    def test_init_units_from_tuple(self, pot: AbstractBasePotential) -> None:
        """Test unit system from tuple."""
        units = (u.km, u.s, u.Msun, u.radian)
        assert replace(pot, units=units).units == unitsystem(*units)

    def test_init_units_from_name(
        self, pot_cls: type[AbstractBasePotential], fields_unitless: dict[str, Array]
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
