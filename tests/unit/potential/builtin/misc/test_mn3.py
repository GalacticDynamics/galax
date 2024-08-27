from typing import Any

import astropy.units as u
import pytest
from plum import convert

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

from ...test_core import AbstractPotential_Test
from ..test_common import (
    ParameterMTotMixin,
    ParameterShapeHRMixin,
    ParameterShapeHZMixin,
)
from galax.potential import AbstractPotentialBase, MN3ExponentialPotential
from galax.typing import Vec3


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
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "h_R": field_h_R,
            "h_z": field_h_z,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: MN3ExponentialPotential, x: Vec3) -> None:
        expect = Quantity(-1.15401718, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: MN3ExponentialPotential, x: Vec3) -> None:
        expect = Quantity(
            [0.0689723071793, 0.1379446143587, 0.2013372530559],
            pot.units["acceleration"],
        )
        got = convert(pot.gradient(x, t=0), Quantity)
        assert qnp.allclose(got, expect, atol=Quantity(1e-8, expect.unit))

    def test_density(self, pot: MN3ExponentialPotential, x: Vec3) -> None:
        expect = Quantity(731_782_542.3781165, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: MN3ExponentialPotential, x: Vec3) -> None:
        expect = Quantity(
            [
                [0.05679591, -0.02435279, -0.03538017],
                [-0.02435279, 0.02026672, -0.07076034],
                [-0.03538017, -0.07076034, -0.03569508],
            ],
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
                [0.04300673, -0.02435279, -0.03538017],
                [-0.02435279, 0.00647754, -0.07076034],
                [-0.03538017, -0.07076034, -0.04948426],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
