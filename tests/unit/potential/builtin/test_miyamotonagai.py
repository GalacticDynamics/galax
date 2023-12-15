from typing import Any

import jax.numpy as xp
import pytest

import galax.potential as gp

from ..test_core import TestAbstractPotential
from .test_common import (
    MassParameterMixin,
    ShapeAParameterMixin,
    ShapeBParameterMixin,
)


class TestMiyamotoNagaiPotential(
    TestAbstractPotential,
    # Parameters
    MassParameterMixin,
    ShapeAParameterMixin,
    ShapeBParameterMixin,
):
    """Test the `galax.potential.MiyamotoNagaiPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MiyamotoNagaiPotential]:
        return gp.MiyamotoNagaiPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m, field_a, field_b, field_units) -> dict[str, Any]:
        return {"m": field_m, "a": field_a, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-0.95208676))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.04264751, 0.08529503, 0.16840152])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 1.9949418e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.03691649, -0.01146205, -0.02262999],
                    [-0.01146205, 0.01972342, -0.04525999],
                    [-0.02262999, -0.04525999, -0.04536254],
                ]
            ),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))
