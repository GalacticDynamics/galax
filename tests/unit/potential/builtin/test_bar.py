from typing import Any

import jax.numpy as xp
import pytest

import galax.potential as gp

from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import (
    MassParameterMixin,
    ShapeAParameterMixin,
    ShapeBParameterMixin,
    ShapeCParameterMixin,
)


class TestBarPotential(
    AbstractPotential_Test,
    # Parameters
    MassParameterMixin,
    ShapeAParameterMixin,
    ShapeBParameterMixin,
    ShapeCParameterMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.BarPotential]:
        return gp.BarPotential

    @pytest.fixture(scope="class")
    def field_Omega(self) -> dict[str, Any]:
        return 0

    @pytest.fixture(scope="class")
    def fields_(
        self, field_m, field_a, field_b, field_c, field_Omega, field_units
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "a": field_a,
            "b": field_b,
            "c": field_c,
            "Omega": field_Omega,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-0.94601574))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.04011905, 0.08383918, 0.16552719])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 1.94669274e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.03529841, -0.01038389, -0.02050134],
                    [-0.01038389, 0.0195721, -0.04412159],
                    [-0.02050134, -0.04412159, -0.04386589],
                ]
            ),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))
