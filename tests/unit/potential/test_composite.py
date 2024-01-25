from collections.abc import Mapping
from dataclasses import replace

import astropy.units as u
import jax.experimental.array_api as xp
import jax.numpy as jnp
import pytest
from typing_extensions import override

import galax.potential as gp
from galax.units import UnitSystem, dimensionless, galactic, solarsystem
from galax.utils._misc import first

from .test_base import TestAbstractPotentialBase as AbstractPotentialBase_Test
from .test_utils import FieldUnitSystemMixin


# TODO: write the base-class test
class AbstractCompositePotential_Test(AbstractPotentialBase_Test, FieldUnitSystemMixin):
    """Test the `galax.potential.AbstractCompositePotential` class."""


class TestCompositePotential(AbstractCompositePotential_Test):
    """Test the `galax.potential.CompositePotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.CompositePotential]:
        """Composite potential class."""
        return gp.CompositePotential

    @pytest.fixture(scope="class")
    def pot_map(self) -> Mapping[str, gp.AbstractPotentialBase]:
        """Composite potential."""
        return {
            "disk": gp.MiyamotoNagaiPotential(
                m=1e10 * u.solMass, a=6.5 * u.kpc, b=4.5 * u.kpc, units=galactic
            ),
            "halo": gp.NFWPotential(
                m=1e12 * u.solMass, r_s=5 * u.kpc, softening_length=0, units=galactic
            ),
        }

    @pytest.fixture(scope="class")
    def pot(
        self,
        pot_cls: type[gp.CompositePotential],
        pot_map: Mapping[str, gp.AbstractPotentialBase],
    ) -> gp.CompositePotential:
        """Composite potential."""
        return pot_cls(**pot_map)

    @pytest.fixture(scope="class")
    def pot_map_unitless(self) -> Mapping[str, gp.AbstractPotentialBase]:
        """Composite potential."""
        return {
            "disk": gp.MiyamotoNagaiPotential(m=1e10, a=6.5, b=4.5, units=None),
            "halo": gp.NFWPotential(m=1e12, r_s=5, softening_length=0, units=None),
        }

    # ==========================================================================
    # TODO: use a universal `replace` function then don't need to override
    #       these tests.

    @override
    def test_init_units_invalid(
        self,
        pot_cls: type[gp.CompositePotential],
        pot_map: Mapping[str, gp.AbstractPotentialBase],
    ) -> None:
        """Test invalid unit system."""
        # TODO: raise a specific error. The type depends on whether beartype is
        # turned on.
        with pytest.raises(Exception):  # noqa: B017, PT011
            pot_cls(**pot_map, units=1234567890)

    @override
    def test_init_units_from_usys(
        self,
        pot_cls: type[gp.CompositePotential],
        pot_map: Mapping[str, gp.AbstractPotentialBase],
    ) -> None:
        """Test unit system from UnitSystem."""
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        pot_map_ = {k: replace(v, units=usys) for k, v in pot_map.items()}
        assert pot_cls(**pot_map_, units=usys).units == usys

    @override
    def test_init_units_from_args(
        self,
        pot_cls: type[gp.CompositePotential],
        pot_map_unitless: Mapping[str, gp.AbstractPotentialBase],
    ) -> None:
        """Test unit system from None."""
        pot = pot_cls(**pot_map_unitless, units=None)
        assert pot.units == dimensionless

    @override
    def test_init_units_from_tuple(
        self,
        pot_cls: type[gp.CompositePotential],
        pot_map: Mapping[str, gp.AbstractPotentialBase],
    ) -> None:
        """Test unit system from tuple."""
        units = (u.km, u.s, u.Msun, u.radian)
        pot_map = {k: replace(v, units=units) for k, v in pot_map.items()}
        assert pot_cls(**pot_map, units=units).units == UnitSystem(*units)

    @override
    def test_init_units_from_name(
        self,
        pot_cls: type[gp.CompositePotential],
        pot_map: Mapping[str, gp.AbstractPotentialBase],
        pot_map_unitless: Mapping[str, gp.AbstractPotentialBase],
    ) -> None:
        """Test unit system from named string."""
        units = "dimensionless"
        potmap = {k: replace(v, units=units) for k, v in pot_map_unitless.items()}
        pot = pot_cls(**potmap, units=units)
        assert pot.units == dimensionless

        units = "solarsystem"
        potmap = {k: replace(v, units=units) for k, v in pot_map.items()}
        pot = pot_cls(**potmap, units=units)
        assert pot.units == solarsystem

        units = "galactic"
        potmap = {k: replace(v, units=units) for k, v in pot_map.items()}
        pot = pot_cls(**potmap, units=units)
        assert pot.units == galactic

        msg = "cannot convert invalid_value to a UnitSystem"
        with pytest.raises(NotImplementedError, match=msg):
            pot_cls(**pot_map_unitless, units="invalid_value")

    # ==========================================================================

    # --------------------------
    # `__or__`

    def test_or_incorrect(self, pot):
        """Test the `__or__` method with incorrect inputs."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = pot | 1

    def test_or_pot(self, pot: gp.CompositePotential) -> None:
        """Test the `__or__` method with a single potential."""
        single_pot = gp.KeplerPotential(m=1e12 * u.solMass, units=galactic)
        newpot = pot | single_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-1]
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_or_compot(self, pot: gp.CompositePotential) -> None:
        """Test the `__or__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m=1e12 * u.solMass, units=galactic),
            kep2=gp.KeplerPotential(m=1e12 * u.solMass, units=galactic),
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

    def test_ror_incorrect(self, pot):
        """Test the `__or__` method with incorrect inputs."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = 1 | pot

    def test_ror_pot(self, pot: gp.CompositePotential) -> None:
        """Test the `__ror__` method with a single potential."""
        single_pot = gp.KeplerPotential(m=1e12 * u.solMass, units=galactic)
        newpot = single_pot | pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = first(newpot.items())
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_ror_compot(self, pot: gp.CompositePotential) -> None:
        """Test the `__ror__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m=1e12 * u.solMass, units=galactic),
            kep2=gp.KeplerPotential(m=1e12 * u.solMass, units=galactic),
        )
        newpot = comp_pot | pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = first(newpot.items())
        assert newkey == "kep1"
        assert newvalue is newpot["kep1"]

        newkey, newvalue = tuple(newpot.items())[1]
        assert newkey == "kep2"
        assert newvalue is newpot["kep2"]

    # --------------------------
    # `__add__`

    def test_add_incorrect(self, pot):
        """Test the `__add__` method with incorrect inputs."""
        # TODO: specific error
        with pytest.raises(Exception):  # noqa: B017, PT011
            _ = pot + 1

    def test_add_pot(self, pot: gp.CompositePotential) -> None:
        """Test the `__add__` method with a single potential."""
        single_pot = gp.KeplerPotential(m=1e12 * u.solMass, units=galactic)
        newpot = pot + single_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-1]
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_add_compot(self, pot: gp.CompositePotential) -> None:
        """Test the `__add__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m=1e12 * u.solMass, units=galactic),
            kep2=gp.KeplerPotential(m=1e12 * u.solMass, units=galactic),
        )
        newpot = pot + comp_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-2]
        assert newkey == "kep1"
        assert newvalue is newpot["kep1"]

        newkey, newvalue = tuple(newpot.items())[-1]
        assert newkey == "kep2"
        assert newvalue is newpot["kep2"]

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert jnp.isclose(pot.potential_energy(x, t=0), xp.asarray(-0.6753781))

    def test_gradient(self, pot, x):
        assert jnp.allclose(
            pot.gradient(x, t=0), xp.asarray([0.01124388, 0.02248775, 0.03382281])
        )

    def test_density(self, pot, x):
        assert jnp.isclose(pot.density(x, t=0), 2.7958598e08)

    def test_hessian(self, pot, x):
        assert jnp.allclose(
            pot.hessian(x, t=0),
            xp.asarray(
                [
                    [0.00996317, -0.0025614, -0.00384397],
                    [-0.0025614, 0.00612107, -0.00768793],
                    [-0.00384397, -0.00768793, -0.00027929],
                ]
            ),
        )
