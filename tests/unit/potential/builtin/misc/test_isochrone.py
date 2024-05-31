from typing import Any

import pytest

import quaxed.numpy as qnp
from unxt import Quantity

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin, ParameterShapeBMixin
from galax.potential import AbstractPotentialBase, IsochronePotential


class TestIsochronePotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeBMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.IsochronePotential]:
        return gp.IsochronePotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m_tot, field_b, field_units) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: IsochronePotential, x: gt.QVec3) -> None:
        expect = Quantity(-0.9231515, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: IsochronePotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [0.04891392, 0.09782784, 0.14674175], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: IsochronePotential, x: gt.QVec3) -> None:
        expect = Quantity(5.04511665e08, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: IsochronePotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.0404695, -0.01688883, -0.02533324],
                [-0.01688883, 0.01513626, -0.05066648],
                [-0.02533324, -0.05066648, -0.0270858],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.03096285, -0.01688883, -0.02533324],
                [-0.01688883, 0.00562961, -0.05066648],
                [-0.02533324, -0.05066648, -0.03659246],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
