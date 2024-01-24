from typing import Any

import jax.numpy as xp
import pytest

import galax.potential as gp

from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import MassParameterMixin, ShapeBParameterMixin


class TestIsochronePotential(
    AbstractPotential_Test,
    # Parameters
    MassParameterMixin,
    ShapeBParameterMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.IsochronePotential]:
        return gp.IsochronePotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m, field_b, field_units) -> dict[str, Any]:
        return {"m": field_m, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-0.9231515))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.04891392, 0.09782784, 0.14674175])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 5.04511665e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.0404695, -0.01688883, -0.02533324],
                    [-0.01688883, 0.01513626, -0.05066648],
                    [-0.02533324, -0.05066648, -0.0270858],
                ]
            ),
        )
