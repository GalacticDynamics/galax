from typing import Any

import jax.numpy as xp
import pytest

from galax.potential import HernquistPotential

from ..test_core import TestAbstractPotential
from .test_common import MassParameterMixin, ShapeAParameterMixin


class TestHernquistPotential(
    TestAbstractPotential,
    # Parameters
    MassParameterMixin,
    ShapeAParameterMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[HernquistPotential]:
        return HernquistPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m, field_a, field_units) -> dict[str, Any]:
        return {"m": field_m, "a": field_a, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-0.94871936))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.05347411, 0.10694822, 0.16042233])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 3.989933e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.04362645, -0.01969533, -0.02954299],
                    [-0.01969533, 0.01408345, -0.05908599],
                    [-0.02954299, -0.05908599, -0.03515487],
                ]
            ),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))
