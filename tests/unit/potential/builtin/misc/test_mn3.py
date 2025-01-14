from typing import Any

import pytest
from plum import convert

import quaxed.numpy as qnp
import unxt as u

from ...test_core import AbstractPotential_Test
from ..test_common import (
    ParameterMTotMixin,
    ParameterShapeHRMixin,
    ParameterShapeHZMixin,
)
from galax.potential import (
    AbstractBasePotential,
    MN3ExponentialPotential,
    MN3Sech2Potential,
)
from galax.typing import Sz3


class TestMN3ExponentialPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeHRMixin,
    ParameterShapeHZMixin,
):
    """Test the `galax.potential.MN3ExponentialPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[MN3ExponentialPotential]:
        return MN3ExponentialPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_h_R: u.Quantity,
        field_h_z: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "h_R": field_h_R,
            "h_z": field_h_z,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: MN3ExponentialPotential, x: Sz3) -> None:
        expect = u.Quantity(-1.15401718, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: MN3ExponentialPotential, x: Sz3) -> None:
        expect = u.Quantity(
            [0.0689723071793, 0.1379446143587, 0.2013372530559],
            pot.units["acceleration"],
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert qnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: MN3ExponentialPotential, x: Sz3) -> None:
        expect = u.Quantity(731_782_542.3781165, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: MN3ExponentialPotential, x: Sz3) -> None:
        expect = u.Quantity(
            [
                [0.05679591, -0.02435279, -0.03538017],
                [-0.02435279, 0.02026672, -0.07076034],
                [-0.03538017, -0.07076034, -0.03569508],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractBasePotential, x: Sz3) -> None:
        """Test the `AbstractBasePotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.04300673, -0.02435279, -0.03538017],
                [-0.02435279, 0.00647754, -0.07076034],
                [-0.03538017, -0.07076034, -0.04948426],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )


class TestMN3Sech2Potential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeHRMixin,
    ParameterShapeHZMixin,
):
    """Test the `galax.potential.MN3Sech2Potential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[MN3Sech2Potential]:
        return MN3Sech2Potential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_h_R: u.Quantity,
        field_h_z: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "h_R": field_h_R,
            "h_z": field_h_z,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: MN3Sech2Potential, x: Sz3) -> None:
        expect = u.Quantity(-1.13545211, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: MN3Sech2Potential, x: Sz3) -> None:
        expect = u.Quantity(
            [0.059397338333485615, 0.11879467666697123, 0.21959289808834268],
            pot.units["acceleration"],
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert qnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: MN3Sech2Potential, x: Sz3) -> None:
        expect = u.Quantity(211_769_063.98948175, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: MN3Sech2Potential, x: Sz3) -> None:
        expect = u.Quantity(
            [
                [0.05071981, -0.01735505, -0.03287182],
                [-0.01735505, 0.02468723, -0.06574365],
                [-0.03287182, -0.06574365, -0.06343577],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractBasePotential, x: Sz3) -> None:
        """Test the `AbstractBasePotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.04672939, -0.01735505, -0.03287182],
                [-0.01735505, 0.02069681, -0.06574365],
                [-0.03287182, -0.06574365, -0.0674262],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
