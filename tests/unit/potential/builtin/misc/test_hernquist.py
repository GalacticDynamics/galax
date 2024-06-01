from typing import Any

import pytest

import quaxed.numpy as qnp
from unxt import Quantity

import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin, ParameterScaleRadiusMixin
from galax.potential import AbstractPotentialBase, HernquistPotential


class TestHernquistPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterScaleRadiusMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[HernquistPotential]:
        return HernquistPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m_tot, field_r_s, field_units) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: HernquistPotential, x: gt.QVec3) -> None:
        expect = Quantity(-0.94871936, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: HernquistPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [0.05347411, 0.10694822, 0.16042233], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: HernquistPotential, x: gt.QVec3) -> None:
        expect = Quantity(3.989933e08, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: HernquistPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.04362645, -0.01969533, -0.02954299],
                [-0.01969533, 0.01408345, -0.05908599],
                [-0.02954299, -0.05908599, -0.03515487],
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
                [0.0361081, -0.01969533, -0.02954299],
                [-0.01969533, 0.00656511, -0.05908599],
                [-0.02954299, -0.05908599, -0.04267321],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
