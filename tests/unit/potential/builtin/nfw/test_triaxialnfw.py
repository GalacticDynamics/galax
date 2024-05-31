"""Test the `galax.potential.TriaxialNFWPotential` class."""

from typing import Any, ClassVar

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import (
    ParameterMMixin,
    ParameterScaleRadiusMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
)
from galax.potential import AbstractPotentialBase, TriaxialNFWPotential


class TestTriaxialNFWPotential(
    AbstractPotential_Test,
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
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "q1": field_q1,
            "q2": field_q2,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: TriaxialNFWPotential, x: gt.QVec3) -> None:
        expect = Quantity(-1.06475915, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: TriaxialNFWPotential, x: gt.QVec3) -> None:
        expect = Quantity([0.03189139, 0.0604938, 0.13157674], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: TriaxialNFWPotential, x: gt.QVec3) -> None:
        expect = Quantity(2.32106514e08, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: TriaxialNFWPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.02774251, -0.00788965, -0.0165603],
                [-0.00788965, 0.01521376, -0.03105306],
                [-0.0165603, -0.03105306, -0.02983532],
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
                [0.02336886, -0.00788965, -0.0165603],
                [-0.00788965, 0.01084011, -0.03105306],
                [-0.0165603, -0.03105306, -0.03420897],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
