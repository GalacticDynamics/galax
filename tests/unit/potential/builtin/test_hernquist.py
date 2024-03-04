from typing import Any

import pytest

import quaxed.numpy as qnp
from unxt import Quantity

import galax.typing as gt
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import MassParameterMixin, ShapeCParameterMixin
from galax.potential import HernquistPotential
from galax.potential._potential.base import AbstractPotentialBase


class TestHernquistPotential(
    AbstractPotential_Test,
    # Parameters
    MassParameterMixin,
    ShapeCParameterMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[HernquistPotential]:
        return HernquistPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m, field_c, field_units) -> dict[str, Any]:
        return {"m": field_m, "c": field_c, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot: HernquistPotential, x: gt.Vec3) -> None:
        expected = Quantity(-0.94871936, pot.units["specific energy"])
        assert qnp.isclose(  # TODO: .value & use pytest-arraydiff
            pot.potential_energy(x, t=0).decompose(pot.units).value,
            expected.value,
        )

    def test_gradient(self, pot: HernquistPotential, x: gt.Vec3) -> None:
        expected = Quantity(
            [0.05347411, 0.10694822, 0.16042233], pot.units["acceleration"]
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.gradient(x, t=0).decompose(pot.units).value,
            expected.value,
        )

    def test_density(self, pot: HernquistPotential, x: gt.Vec3) -> None:
        expected = Quantity(3.989933e08, pot.units["mass density"])
        assert qnp.isclose(  # TODO: .value & use pytest-arraydiff
            pot.density(x, t=0).decompose(pot.units).value,
            expected.value,
        )

    def test_hessian(self, pot: HernquistPotential, x: gt.Vec3) -> None:
        expected = Quantity(
            [
                [0.04362645, -0.01969533, -0.02954299],
                [-0.01969533, 0.01408345, -0.05908599],
                [-0.02954299, -0.05908599, -0.03515487],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.hessian(x, t=0).decompose(pot.units).value, expected.value
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expected = Quantity(
            [
                [0.0361081, -0.01969533, -0.02954299],
                [-0.01969533, 0.00656511, -0.05908599],
                [-0.02954299, -0.05908599, -0.04267321],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.tidal_tensor(x, t=0).decompose(pot.units).value, expected.value
        )
