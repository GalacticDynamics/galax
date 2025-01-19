"""Tests for the `galax.potential.CompositePotential` class."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from typing_extensions import override

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u
from unxt.unitsystems import galactic, solarsystem
from zeroth import zeroth

import galax.potential as gp
from .test_base import AbstractPotential_Test
from .test_utils import FieldUnitSystemMixin
from galax.typing import Sz3


# TODO: write the base-class test
class AbstractCompositePotential_Test(AbstractPotential_Test, FieldUnitSystemMixin):
    """Test the `galax.potential.AbstractCompositePotential` class."""

    @pytest.fixture(scope="class")
    def pot(
        self,
        pot_cls: type[gp.AbstractCompositePotential],
        pot_map: Mapping[str, Any],
    ) -> gp.AbstractCompositePotential:
        """Composite potential."""
        return pot_cls(**pot_map)

    # ==========================================================================
    # TODO: use a universal `replace` function then don't need to override
    #       these tests.

    @override
    def test_init_units_invalid(
        self,
        pot_cls: type[gp.AbstractCompositePotential],
        pot_map: Mapping[str, Any],
    ) -> None:
        """Test invalid unit system."""
        # TODO: raise a specific error. The type depends on whether beartype is
        # turned on.
        with pytest.raises(Exception):  # noqa: B017, PT011
            pot_cls(**pot_map, units=1234567890)

    @override
    def test_init_units_from_usys(
        self,
        pot_cls: type[gp.AbstractCompositePotential],
        pot_map: Mapping[str, Any],
    ) -> None:
        """Test unit system from unitsystem."""
        usys = u.unitsystem("km", "s", "Msun", "radian")
        assert pot_cls(**pot_map, units=usys).units == usys

    @override
    def test_init_units_from_tuple(
        self,
        pot_cls: type[gp.AbstractCompositePotential],
        pot_map: Mapping[str, Any],
    ) -> None:
        """Test unit system from tuple."""
        units = ("km", "s", "Msun", "radian")
        assert pot_cls(**pot_map, units=units).units == u.unitsystem(*units)

    @override
    def test_init_units_from_name(
        self,
        pot_cls: type[gp.MilkyWayPotential],
        pot_map: Mapping[str, gp.AbstractPotential],
    ) -> None:
        """Test unit system from named string."""
        # TODO: sort this out
        # units = "dimensionless"
        # pot = pot_cls(**pot_map, units=units)
        # assert pot.units == dimensionless

        pot = pot_cls(**pot_map, units="solarsystem")
        assert pot.units == solarsystem

        pot = pot_cls(**pot_map, units="galactic")
        assert pot.units == galactic

        with pytest.raises(KeyError, match="invalid_value"):
            pot_cls(**pot_map, units="invalid_value")

    # ==========================================================================

    # --------------------------
    # `__or__`

    def test_or_incorrect(self, pot: gp.AbstractCompositePotential) -> None:
        """Test the `__or__` method with incorrect inputs."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = pot | 1

    def test_or_pot(self, pot: gp.AbstractCompositePotential) -> None:
        """Test the `__or__` method with a single potential."""
        single_pot = gp.KeplerPotential(
            m_tot=u.Quantity(1e12, "solMass"), units=galactic
        )
        newpot = pot | single_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-1]
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_or_compot(self, pot: gp.AbstractCompositePotential) -> None:
        """Test the `__or__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m_tot=u.Quantity(1e12, "solMass"), units=galactic),
            kep2=gp.KeplerPotential(m_tot=u.Quantity(1e12, "solMass"), units=galactic),
        )
        newpot = pot | comp_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-2]
        assert newkey == "kep1"
        assert newvalue is newpot["kep1"]

        newkey, newvalue = tuple(newpot.items())[-1]
        assert newkey == "kep2"
        assert newvalue is newpot["kep2"]

    # --------------------------
    # `__ror__`

    def test_ror_incorrect(self, pot: gp.CompositePotential) -> None:
        """Test the `__or__` method with incorrect inputs."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = 1 | pot

    def test_ror_pot(self, pot: gp.CompositePotential) -> None:
        """Test the `__ror__` method with a single potential."""
        single_pot = gp.KeplerPotential(
            m_tot=u.Quantity(1e12, "solMass"), units=galactic
        )
        newpot = single_pot | pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = zeroth(newpot.items())
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_ror_compot(self, pot: gp.CompositePotential) -> None:
        """Test the `__ror__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m_tot=u.Quantity(1e12, "solMass"), units=galactic),
            kep2=gp.KeplerPotential(m_tot=u.Quantity(1e12, "solMass"), units=galactic),
        )
        newpot = comp_pot | pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = zeroth(newpot.items())
        assert newkey == "kep1"
        assert newvalue is newpot["kep1"]

        newkey, newvalue = tuple(newpot.items())[1]
        assert newkey == "kep2"
        assert newvalue is newpot["kep2"]

    # --------------------------
    # `__add__`

    def test_add_incorrect(self, pot: gp.CompositePotential) -> None:
        """Test the `__add__` method with incorrect inputs."""
        # TODO: specific error
        with pytest.raises(Exception):  # noqa: B017, PT011
            _ = pot + 1

    def test_add_pot(self, pot: gp.CompositePotential) -> None:
        """Test the `__add__` method with a single potential."""
        single_pot = gp.KeplerPotential(
            m_tot=u.Quantity(1e12, "solMass"), units=galactic
        )
        newpot = pot + single_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-1]
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_add_compot(self, pot: gp.CompositePotential) -> None:
        """Test the `__add__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m_tot=u.Quantity(1e12, "solMass"), units=galactic),
            kep2=gp.KeplerPotential(m_tot=u.Quantity(1e12, "solMass"), units=galactic),
        )
        newpot = pot + comp_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-2]
        assert newkey == "kep1"
        assert newvalue is newpot["kep1"]

        newkey, newvalue = tuple(newpot.items())[-1]
        assert newkey == "kep2"
        assert newvalue is newpot["kep2"]


class TestCompositePotential(AbstractCompositePotential_Test):
    """Test the `galax.potential.CompositePotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.CompositePotential]:
        """Composite potential class."""
        return gp.CompositePotential

    @pytest.fixture(scope="class")
    def pot_map(self) -> Mapping[str, gp.AbstractPotential]:
        """Composite potential."""
        return {
            "disk": gp.MiyamotoNagaiPotential(
                m_tot=u.Quantity(1e10, "solMass"),
                a=u.Quantity(6.5, "kpc"),
                b=u.Quantity(4.5, "kpc"),
                units=galactic,
            ),
            "halo": gp.NFWPotential(
                m=u.Quantity(1e12, "solMass"), r_s=u.Quantity(5, "kpc"), units=galactic
            ),
        }

    # ==========================================================================
    # TODO: use a universal `replace` function then don't need to override
    #       these tests.

    @override
    def test_init_units_from_usys(
        self,
        pot_cls: type[gp.AbstractCompositePotential],
        pot_map: Mapping[str, gp.AbstractPotential],
    ) -> None:
        """Test unit system from UnitSystem."""
        usys = u.unitsystem("km", "s", "Msun", "radian")
        pot_map_ = {k: replace(v, units=usys) for k, v in pot_map.items()}
        assert pot_cls(**pot_map_, units=usys).units == usys

    @override
    def test_init_units_from_tuple(
        self,
        pot_cls: type[gp.AbstractCompositePotential],
        pot_map: Mapping[str, gp.AbstractPotential],
    ) -> None:
        """Test unit system from tuple."""
        units = ("km", "s", "Msun", "radian")
        pot_map = {k: replace(v, units=units) for k, v in pot_map.items()}
        assert pot_cls(**pot_map, units=units).units == u.unitsystem(*units)

    @override
    def test_init_units_from_name(
        self,
        pot_cls: type[gp.CompositePotential],
        pot_map: Mapping[str, gp.AbstractPotential],
        # pot_map_unitless: Mapping[str, AbstractPotential],
    ) -> None:
        """Test unit system from named string."""
        # TODO: address directly in a followup PR.
        # This should fail if there's a good __check_init__ happening.
        # units = "dimensionless"
        # with pytest.raises((u.UnitConversionError, ValueError)):
        #     potmap = {k: replace(v, units=units) for k, v in pot_map.items()}
        #     # pot = pot_cls(**potmap, units=units)
        #     # assert pot.units == dimensionless

        units = "solarsystem"
        potmap = {k: replace(v, units=units) for k, v in pot_map.items()}
        pot = pot_cls(**potmap, units=units)
        assert pot.units == solarsystem

        units = "galactic"
        potmap = {k: replace(v, units=units) for k, v in pot_map.items()}
        pot = pot_cls(**potmap, units=units)
        assert pot.units == galactic

        with pytest.raises(KeyError, match="invalid_value"):
            pot_cls(**pot_map, units="invalid_value")

    # ==========================================================================

    def test_potential(self, pot: gp.CompositePotential, x: Sz3) -> None:
        expect = u.Quantity(jnp.asarray(-0.6753781), "kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.CompositePotential, x: Sz3) -> None:
        expect = u.Quantity(
            [0.01124388, 0.02248775, 0.03382281], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.CompositePotential, x: Sz3) -> None:
        expect = u.Quantity(2.7958598e08, "Msun / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.CompositePotential, x: Sz3) -> None:
        expect = u.Quantity(
            jnp.asarray(
                [
                    [0.00996317, -0.0025614, -0.00384397],
                    [-0.0025614, 0.00612107, -0.00768793],
                    [-0.00384397, -0.00768793, -0.00027929],
                ]
            ),
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: Sz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.00469486, -0.0025614, -0.00384397],
                [-0.0025614, 0.00085275, -0.00768793],
                [-0.00384397, -0.00768793, -0.00554761],
            ],
            pot.units["frequency drift"],
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
