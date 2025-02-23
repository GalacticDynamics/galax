"""Tests for `galax.potential._src.utils` package."""

from dataclasses import replace
from typing import Any

import pytest
from jax import Array

import unxt as u

from galax.potential import AbstractPotential


class FieldUnitSystemMixin:
    """Mixin for testing the ``units`` field on a ``Potential``."""

    @pytest.fixture
    def fields_unitless(self, fields: dict[str, Any]) -> dict[str, Array]:
        """Fields with no units."""
        return {
            k: (v.value if isinstance(v, u.Quantity) else v) for k, v in fields.items()
        }

    # ===========================================

    def test_init_units_from_usys(self, pot: AbstractPotential) -> None:
        """Test unit system from UnitSystem."""
        usys = u.unitsystem("km", "s", "Msun", "radian")
        assert replace(pot, units=usys).units == usys

    def test_init_units_from_tuple(self, pot: AbstractPotential) -> None:
        """Test unit system from tuple."""
        units = ("km", "s", "Msun", "radian")
        assert replace(pot, units=units).units == u.unitsystem(*units)

    def test_init_units_from_name(
        self, pot_cls: type[AbstractPotential], fields_unitless: dict[str, Array]
    ) -> None:
        """Test unit system from named string."""
        fields_unitless.pop("units")

        # TODO: sort this out
        # pot = pot_cls(**fields_unitless, units="dimensionless")
        # # assert pot.units == dimensionless

        pot = pot_cls(**fields_unitless, units="solarsystem")
        assert pot.units == u.unitsystems.solarsystem

        pot = pot_cls(**fields_unitless, units="galactic")
        assert pot.units == u.unitsystems.galactic

        with pytest.raises(KeyError, match="invalid_value"):
            pot_cls(**fields_unitless, units="invalid_value")
