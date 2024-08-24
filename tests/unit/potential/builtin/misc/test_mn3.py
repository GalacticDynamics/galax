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
        # TODO: need to update expected value
        expect = Quantity(-0.95208676, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: MN3ExponentialPotential, x: Vec3) -> None:
        # TODO: need to update expected value
        expect = Quantity(
            [0.04264751, 0.08529503, 0.16840152], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), Quantity)
        assert qnp.allclose(got, expect, atol=Quantity(1e-8, expect.unit))

    def test_density(self, pot: MN3ExponentialPotential, x: Vec3) -> None:
        # TODO: need to update expected value
        expect = Quantity(1.9949418e08, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: MN3ExponentialPotential, x: Vec3) -> None:
        # TODO: need to update expected value
        expect = Quantity(
            [
                [0.03691649, -0.01146205, -0.02262999],
                [-0.01146205, 0.01972342, -0.04525999],
                [-0.02262999, -0.04525999, -0.04536254],
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
        # TODO: need to update expected value
        expect = Quantity(
            [
                [0.03315736, -0.01146205, -0.02262999],
                [-0.01146205, 0.0159643, -0.04525999],
                [-0.02262999, -0.04525999, -0.04912166],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
