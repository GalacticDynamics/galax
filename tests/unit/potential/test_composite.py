from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import astropy.units as u
import pytest
from typing_extensions import override

import quaxed.array_api as xp
import quaxed.numpy as qnp
from unxt import Quantity
from unxt.unitsystems import UnitSystem, dimensionless, galactic, solarsystem

import galax.potential as gp
from .test_base import AbstractPotentialBase_Test
from .test_utils import FieldUnitSystemMixin
from galax.typing import Vec3
from galax.utils._misc import zeroth

if TYPE_CHECKING:
    from galax.potential import (
        AbstractCompositePotential,
        AbstractPotentialBase,
        CompositePotential,
    )


# TODO: write the base-class test
class AbstractCompositePotential_Test(AbstractPotentialBase_Test, FieldUnitSystemMixin):
    """Test the `galax.potential.AbstractCompositePotential` class."""

    @pytest.fixture(scope="class")
    def pot(
        self,
        pot_cls: type[AbstractCompositePotential],
        pot_map: Mapping[str, Any],
    ) -> AbstractCompositePotential:
        """Composite potential."""
        return pot_cls(**pot_map)

    # ==========================================================================
    # TODO: use a universal `replace` function then don't need to override
    #       these tests.

    @override
    def test_init_units_invalid(
        self,
        pot_cls: type[AbstractCompositePotential],
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
        pot_cls: type[AbstractCompositePotential],
        pot_map: Mapping[str, Any],
    ) -> None:
        """Test unit system from UnitSystem."""
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        assert pot_cls(**pot_map, units=usys).units == usys

    @override
    def test_init_units_from_tuple(
        self,
        pot_cls: type[AbstractCompositePotential],
        pot_map: Mapping[str, Any],
    ) -> None:
        """Test unit system from tuple."""
        units = (u.km, u.s, u.Msun, u.radian)
        assert pot_cls(**pot_map, units=units).units == UnitSystem(*units)

    @override
    def test_init_units_from_name(
        self,
        pot_cls: type[MilkyWayPotential],
        pot_map: Mapping[str, AbstractPotentialBase],
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

    def test_or_incorrect(self, pot: AbstractCompositePotential) -> None:
        """Test the `__or__` method with incorrect inputs."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = pot | 1

    def test_or_pot(self, pot: AbstractCompositePotential) -> None:
        """Test the `__or__` method with a single potential."""
        single_pot = gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic)
        newpot = pot | single_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-1]
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_or_compot(self, pot: AbstractCompositePotential) -> None:
        """Test the `__or__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic),
            kep2=gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic),
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

    def test_ror_incorrect(self, pot: CompositePotential) -> None:
        """Test the `__or__` method with incorrect inputs."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = 1 | pot

    def test_ror_pot(self, pot: CompositePotential) -> None:
        """Test the `__ror__` method with a single potential."""
        single_pot = gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic)
        newpot = single_pot | pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = zeroth(newpot.items())
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_ror_compot(self, pot: CompositePotential) -> None:
        """Test the `__ror__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic),
            kep2=gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic),
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

    def test_add_incorrect(self, pot: CompositePotential) -> None:
        """Test the `__add__` method with incorrect inputs."""
        # TODO: specific error
        with pytest.raises(Exception):  # noqa: B017, PT011
            _ = pot + 1

    def test_add_pot(self, pot: CompositePotential) -> None:
        """Test the `__add__` method with a single potential."""
        single_pot = gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic)
        newpot = pot + single_pot

        assert isinstance(newpot, gp.CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-1]
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_add_compot(self, pot: CompositePotential) -> None:
        """Test the `__add__` method with a composite potential."""
        comp_pot = gp.CompositePotential(
            kep1=gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic),
            kep2=gp.KeplerPotential(m_tot=1e12 * u.solMass, units=galactic),
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
    def pot_cls(self) -> type[CompositePotential]:
        """Composite potential class."""
        return gp.CompositePotential

    @pytest.fixture(scope="class")
    def pot_map(self) -> Mapping[str, AbstractPotentialBase]:
        """Composite potential."""
        return {
            "disk": gp.MiyamotoNagaiPotential(
                m_tot=1e10 * u.solMass, a=6.5 * u.kpc, b=4.5 * u.kpc, units=galactic
            ),
            "halo": gp.NFWPotential(m=1e12 * u.solMass, r_s=5 * u.kpc, units=galactic),
        }

    # TODO(@nstarman): figure out what to do with unitless potentials. I think
    #                 they NOT be allowed.
    # @pytest.fixture(scope="class")
    # def pot_map_unitless(self) -> Mapping[str, AbstractPotentialBase]:
    #     """Composite potential."""
    #     return {
    #         "disk": gp.MiyamotoNagaiPotential(
    #             m=Quantity(1e10, "Msun"),
    #             a=Quantity(6.5, "kpc"),
    #             b=Quantity(4.5, "kpc"),
    #             units=None,
    #         ),
    #         "halo": gp.NFWPotential(
    #             m=Quantity(1e12, "Msun"), r_s=Quantity(5, "kpc"), units=None
    #         ),
    #     }

    # ==========================================================================
    # TODO: use a universal `replace` function then don't need to override
    #       these tests.

    @override
    def test_init_units_from_usys(
        self,
        pot_cls: type[AbstractCompositePotential],
        pot_map: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from UnitSystem."""
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        pot_map_ = {k: replace(v, units=usys) for k, v in pot_map.items()}
        assert pot_cls(**pot_map_, units=usys).units == usys

    @pytest.mark.xfail(reason="TODO: unitless potentials are not allowed.")
    @override
    def test_init_units_from_args(
        self,
        pot_cls: type[CompositePotential],
        pot_map: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from None."""
        pot = pot_cls(**pot_map, units=None)
        assert pot.units == dimensionless

    @override
    def test_init_units_from_tuple(
        self,
        pot_cls: type[CompositePotential],
        pot_map: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from tuple."""
        units = (u.km, u.s, u.Msun, u.radian)
        pot_map = {k: replace(v, units=units) for k, v in pot_map.items()}
        assert pot_cls(**pot_map, units=units).units == UnitSystem(*units)

    @override
    def test_init_units_from_name(
        self,
        pot_cls: type[CompositePotential],
        pot_map: Mapping[str, AbstractPotentialBase],
        # pot_map_unitless: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from named string."""
        units = "dimensionless"
        with pytest.raises(  # TODO: address directly
            (u.UnitConversionError, ValueError)
        ):
            potmap = {k: replace(v, units=units) for k, v in pot_map.items()}
            # pot = pot_cls(**potmap, units=units)
            # assert pot.units == dimensionless

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

    def test_potential(self, pot: CompositePotential, x: Vec3) -> None:
        expect = Quantity(xp.asarray(-0.6753781), "kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: CompositePotential, x: Vec3) -> None:
        expect = Quantity(
            [0.01124388, 0.02248775, 0.03382281], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: CompositePotential, x: Vec3) -> None:
        expect = Quantity(2.7958598e08, "Msun / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: CompositePotential, x: Vec3) -> None:
        expect = Quantity(
            xp.asarray(
                [
                    [0.00996317, -0.0025614, -0.00384397],
                    [-0.0025614, 0.00612107, -0.00768793],
                    [-0.00384397, -0.00768793, -0.00027929],
                ]
            ),
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.00469486, -0.0025614, -0.00384397],
                [-0.0025614, 0.00085275, -0.00768793],
                [-0.00384397, -0.00768793, -0.00554761],
            ],
            pot.units["frequency drift"],
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
