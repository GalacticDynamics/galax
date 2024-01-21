from typing import Any

import jax.numpy as xp
import pytest

import galax.potential as gp

from ..test_core import TestAbstractPotential


class TestBarPotential(TestAbstractPotential):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.NullPotential]:
        return gp.NullPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_units) -> dict[str, Any]:
        return {"units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(0.0))

    def test_gradient(self, pot, x):
        assert xp.allclose(pot.gradient(x, t=0), xp.array([0.0, 0.0, 0.0]))

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 0.0)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))
