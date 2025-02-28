"""Test the `galax.potential.TriaxialNFWPotential` class."""

from typing import Any, ClassVar

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ...test_core import AbstractSinglePotential_Test
from ..test_common import (
    ParameterMMixin,
    ParameterScaleRadiusMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
)


class TestTriaxialNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterScaleRadiusMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
):
    """Test the `galax.potential.TriaxialNFWPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.TriaxialNFWPotential]:
        return gp.TriaxialNFWPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_q1: u.Quantity,
        field_q2: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "q1": field_q1,
            "q2": field_q2,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.TriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-1.06475915, unit="kpc2 / Myr2")
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_gradient(self, pot: gp.TriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.03189139, 0.0604938, 0.13157674], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.TriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(2.32106514e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.TriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.02774251, -0.00788965, -0.0165603],
                [-0.00788965, 0.01521376, -0.03105306],
                [-0.0165603, -0.03105306, -0.02983532],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.02336886, -0.00788965, -0.0165603],
                [-0.00788965, 0.01084011, -0.03105306],
                [-0.0165603, -0.03105306, -0.03420897],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
