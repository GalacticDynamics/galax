import re
from collections.abc import Mapping

import array_api_jax_compat as xp
import astropy.units as u
import jax.numpy as jnp
import pytest
from plum import NotFoundLookupError
from quax import quaxify
from typing_extensions import override

from jax_quantity import Quantity

from .test_composite import AbstractCompositePotential_Test
from galax.potential import (
    AbstractPotentialBase,
    CompositePotential,
    KeplerPotential,
    MilkyWayPotential,
)
from galax.typing import Vec3
from galax.units import UnitSystem, dimensionless, galactic, solarsystem
from galax.utils._misc import first

allclose = quaxify(jnp.allclose)


##############################################################################


class TestMilkyWayPotential(AbstractCompositePotential_Test):
    """Test the `galax.potential.MilkyWayPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[MilkyWayPotential]:
        return MilkyWayPotential

    @pytest.fixture(scope="class")
    def pot_map(
        self, pot_cls: type[MilkyWayPotential]
    ) -> dict[str, dict[str, Quantity]]:
        """Composite potential."""
        return {
            "disk": pot_cls._default_disk,
            "halo": pot_cls._default_halo,
            "bulge": pot_cls._default_bulge,
            "nucleus": pot_cls._default_nucleus,
        }

    @pytest.fixture(scope="class")
    def pot(
        self,
        pot_cls: type[MilkyWayPotential],
        pot_map: Mapping[str, AbstractPotentialBase],
    ) -> MilkyWayPotential:
        """Composite potential."""
        return pot_cls(**pot_map)

    @pytest.fixture(scope="class")
    def pot_map_unitless(self, pot_map) -> Mapping[str, AbstractPotentialBase]:
        """Composite potential."""
        return {k: {kk: vv.value for kk, vv in v.items()} for k, v in pot_map.items()}

    # ==========================================================================
    # TODO: use a universal `replace` function then don't need to override
    #       these tests.

    @override
    def test_init_units_invalid(
        self,
        pot_cls: type[MilkyWayPotential],
        pot_map: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test invalid unit system."""
        # TODO: raise a specific error. The type depends on whether beartype is
        # turned on.
        with pytest.raises(Exception):  # noqa: B017, PT011
            pot_cls(**pot_map, units=1234567890)

    @override
    def test_init_units_from_usys(
        self,
        pot_cls: type[MilkyWayPotential],
        pot_map: MilkyWayPotential,
    ) -> None:
        """Test unit system from UnitSystem."""
        usys = UnitSystem(u.km, u.s, u.Msun, u.radian)
        assert pot_cls(**pot_map, units=usys).units == usys

    @override
    def test_init_units_from_args(
        self,
        pot_cls: type[MilkyWayPotential],
        pot_map_unitless: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from None."""
        pot = pot_cls(**pot_map_unitless, units=None)
        assert pot.units == galactic

    @override
    def test_init_units_from_tuple(
        self,
        pot_cls: type[MilkyWayPotential],
        pot_map: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from tuple."""
        units = (u.km, u.s, u.Msun, u.radian)
        assert pot_cls(**pot_map, units=units).units == UnitSystem(*units)

    @override
    def test_init_units_from_name(
        self,
        pot_cls: type[MilkyWayPotential],
        pot_map: Mapping[str, AbstractPotentialBase],
        pot_map_unitless: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from named string."""
        units = "dimensionless"
        pot = pot_cls(**pot_map_unitless, units=units)
        assert pot.units == dimensionless

        units = "solarsystem"
        pot = pot_cls(**pot_map, units=units)
        assert pot.units == solarsystem

        units = "galactic"
        pot = pot_cls(**pot_map, units=units)
        assert pot.units == galactic

        msg = "`unitsystem('invalid_value')` could not be resolved."
        with pytest.raises(NotFoundLookupError, match=re.escape(msg)):
            pot_cls(**pot_map, units="invalid_value")

    # ==========================================================================

    # --------------------------
    # `__or__`

    def test_or_incorrect(self, pot: MilkyWayPotential) -> None:
        """Test the `__or__` method with incorrect inputs."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = pot | 1

    def test_or_pot(self, pot: MilkyWayPotential) -> None:
        """Test the `__or__` method with a single potential."""
        single_pot = KeplerPotential(m=1e12 * u.solMass, units=galactic)
        newpot = pot | single_pot

        assert isinstance(newpot, CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-1]
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_or_compot(self, pot: MilkyWayPotential) -> None:
        """Test the `__or__` method with a composite potential."""
        comp_pot = CompositePotential(
            kep1=KeplerPotential(m=1e12 * u.solMass, units=galactic),
            kep2=KeplerPotential(m=1e12 * u.solMass, units=galactic),
        )
        newpot = pot | comp_pot

        assert isinstance(newpot, CompositePotential)

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
        single_pot = KeplerPotential(m=1e12 * u.solMass, units=galactic)
        newpot = single_pot | pot

        assert isinstance(newpot, CompositePotential)

        newkey, newvalue = first(newpot.items())
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_ror_compot(self, pot: CompositePotential) -> None:
        """Test the `__ror__` method with a composite potential."""
        comp_pot = CompositePotential(
            kep1=KeplerPotential(m=1e12 * u.solMass, units=galactic),
            kep2=KeplerPotential(m=1e12 * u.solMass, units=galactic),
        )
        newpot = comp_pot | pot

        assert isinstance(newpot, CompositePotential)

        newkey, newvalue = first(newpot.items())
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
        single_pot = KeplerPotential(m=1e12 * u.solMass, units=galactic)
        newpot = pot + single_pot

        assert isinstance(newpot, CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-1]
        assert isinstance(newkey, str)
        assert newvalue is single_pot

    def test_add_compot(self, pot: CompositePotential) -> None:
        """Test the `__add__` method with a composite potential."""
        comp_pot = CompositePotential(
            kep1=KeplerPotential(m=1e12 * u.solMass, units=galactic),
            kep2=KeplerPotential(m=1e12 * u.solMass, units=galactic),
        )
        newpot = pot + comp_pot

        assert isinstance(newpot, CompositePotential)

        newkey, newvalue = tuple(newpot.items())[-2]
        assert newkey == "kep1"
        assert newvalue is newpot["kep1"]

        newkey, newvalue = tuple(newpot.items())[-1]
        assert newkey == "kep2"
        assert newvalue is newpot["kep2"]

    # ==========================================================================

    def test_potential_energy(self, pot: MilkyWayPotential, x: Vec3) -> None:
        """Test the :meth:`MilkyWayPotential.potential_energy` method."""
        assert jnp.isclose(pot.potential_energy(x, t=0).value, xp.asarray(-0.19386052))

    def test_gradient(self, pot: MilkyWayPotential, x: Vec3) -> None:
        """Test the :meth:`MilkyWayPotential.gradient` method."""
        expected = Quantity(
            [0.00256403, 0.00512806, 0.01115272], pot.units["acceleration"]
        )
        assert allclose(pot.gradient(x, t=0).value, expected.value)  # TODO: not .value

    def test_density(self, pot: MilkyWayPotential, x: Vec3) -> None:
        """Test the :meth:`MilkyWayPotential.density` method."""
        assert jnp.isclose(pot.density(x, t=0).value, 33_365_858.46361218)

    def test_hessian(self, pot: MilkyWayPotential, x: Vec3) -> None:
        """Test the :meth:`MilkyWayPotential.hessian` method."""
        assert allclose(
            pot.hessian(x, t=0),
            xp.asarray(
                [
                    [0.00231054, -0.00050698, -0.00101273],
                    [-0.00050698, 0.00155006, -0.00202546],
                    [-0.00101273, -0.00202546, -0.00197444],
                ]
            ),
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = [
            [0.00168182, -0.00050698, -0.00101273],
            [-0.00050698, 0.00092134, -0.00202546],
            [-0.00101273, -0.00202546, -0.00260316],
        ]
        assert allclose(pot.tidal_tensor(x, t=0), xp.asarray(expect))
