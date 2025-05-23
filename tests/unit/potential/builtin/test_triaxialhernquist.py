"""Test the `TriaxialHernquistPotential` class."""

from typing import Any, ClassVar

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMTotMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
)


class TestTriaxialHernquistPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
):
    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.TriaxialHernquistPotential]:
        return gp.TriaxialHernquistPotential

    @pytest.fixture(scope="class")
    def fields_(
        self, field_m_tot, field_r_s, field_q1, field_q2, field_units
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_s": field_r_s,
            "q1": field_q1,
            "q2": field_q2,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.TriaxialHernquistPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-0.61215074, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.TriaxialHernquistPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [0.01312095, 0.02168751, 0.15745134], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    @pytest.mark.xfail(reason="WFF?")
    def test_density(self, pot: gp.TriaxialHernquistPotential, x: gt.QuSz3) -> None:
        assert pot.density(x, t=0).decompose(pot.units).value >= 0

    def test_hessian(self, pot: gp.TriaxialHernquistPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.01223294, -0.00146778, -0.0106561],
                [-0.00146778, 0.00841767, -0.01761339],
                [-0.0106561, -0.01761339, -0.07538941],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, "1/Myr2")
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.03047921, -0.00146778, -0.0106561],
                [-0.00146778, 0.02666394, -0.01761339],
                [-0.0106561, -0.01761339, -0.05714314],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
