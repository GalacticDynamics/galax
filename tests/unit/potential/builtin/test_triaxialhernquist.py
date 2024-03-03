from typing import Any

import jax.numpy as jnp
import pytest
from quax import quaxify

import array_api_jax_compat as xp
from jax_quantity import Quantity

from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import (
    MassParameterMixin,
    ShapeCParameterMixin,
    ShapeQ1ParameterMixin,
    ShapeQ2ParameterMixin,
)
from galax.potential import TriaxialHernquistPotential
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import Vec3

allclose = quaxify(jnp.allclose)


class TestTriaxialHernquistPotential(
    AbstractPotential_Test,
    # Parameters
    MassParameterMixin,
    ShapeCParameterMixin,
    ShapeQ1ParameterMixin,
    ShapeQ2ParameterMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[TriaxialHernquistPotential]:
        return TriaxialHernquistPotential

    @pytest.fixture(scope="class")
    def fields_(
        self, field_m, field_c, field_q1, field_q2, field_units
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "c": field_c,
            "q1": field_q1,
            "q2": field_q2,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot: TriaxialHernquistPotential, x: Vec3) -> None:
        assert jnp.isclose(pot.potential_energy(x, t=0).value, xp.asarray(-0.61215074))

    def test_gradient(self, pot: TriaxialHernquistPotential, x: Vec3) -> None:
        expected = Quantity(
            [0.01312095, 0.02168751, 0.15745134], pot.units["acceleration"]
        )
        assert allclose(pot.gradient(x, t=0).value, expected.value)  # TODO: not .value

    @pytest.mark.xfail(reason="WFF?")
    def test_density(self, pot: TriaxialHernquistPotential, x: Vec3) -> None:
        assert pot.density(x, t=0).value >= 0

    def test_hessian(self, pot: TriaxialHernquistPotential, x: Vec3) -> None:
        assert jnp.allclose(
            pot.hessian(x, t=0),
            xp.asarray(
                [
                    [0.01223294, -0.00146778, -0.0106561],
                    [-0.00146778, 0.00841767, -0.01761339],
                    [-0.0106561, -0.01761339, -0.07538941],
                ]
            ),
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = [
            [0.03047921, -0.00146778, -0.0106561],
            [-0.00146778, 0.02666394, -0.01761339],
            [-0.0106561, -0.01761339, -0.05714314],
        ]
        assert allclose(pot.tidal_tensor(x, t=0), xp.asarray(expect))
