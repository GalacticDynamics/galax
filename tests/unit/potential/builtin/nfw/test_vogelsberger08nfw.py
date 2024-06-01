"""Test the `galax.potential.Vogelsberger08TriaxialNFWPotential` class."""

from typing import Any, ClassVar

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...param.test_field import ParameterFieldMixin
from ...test_core import AbstractPotential_Test
from ..test_common import (
    ParameterMMixin,
    ParameterScaleRadiusMixin,
    ParameterShapeQ1Mixin,
)
from galax.potential import AbstractPotentialBase, Vogelsberger08TriaxialNFWPotential


class ShapeTransitionRadiusParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a_r(self) -> Quantity["dimensionless"]:
        return Quantity(1.0, "")

    # =====================================================

    def test_a_r_constant(self, pot_cls, fields):
        """Test the `a_r` parameter."""
        fields["a_r"] = Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a_r(t=0) == Quantity(1.0, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_a_r_userfunc(self, pot_cls, fields):
        """Test the `a_r` parameter."""
        fields["a_r"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a1(t=0) == 2


class TestVogelsberger08TriaxialNFWPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterScaleRadiusMixin,
    ParameterShapeQ1Mixin,
    ShapeTransitionRadiusParameterMixin,
):
    """Test the `galax.potential.Vogelsberger08TriaxialNFWPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.Vogelsberger08TriaxialNFWPotential]:
        return gp.Vogelsberger08TriaxialNFWPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_q1: u.Quantity,
        field_a_r: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "q1": field_q1,
            "a_r": field_a_r,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(
        self, pot: Vogelsberger08TriaxialNFWPotential, x: gt.QVec3
    ) -> None:
        expect = Quantity(-1.91410199, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(
        self, pot: Vogelsberger08TriaxialNFWPotential, x: gt.QVec3
    ) -> None:
        expect = Quantity([0.07701115, 0.14549116, 0.19849185], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(
        self, pot: Vogelsberger08TriaxialNFWPotential, x: gt.QVec3
    ) -> None:
        expect = Quantity(1.10157433e09, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(
        self, pot: Vogelsberger08TriaxialNFWPotential, x: gt.QVec3
    ) -> None:
        expect = Quantity(
            [
                [0.06195284, -0.0274773, -0.0351074],
                [-0.0274773, 0.02218247, -0.06568078],
                [-0.0351074, -0.06568078, -0.02186349],
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
                [0.04119557, -0.0274773, -0.0351074],
                [-0.0274773, 0.00142519, -0.06568078],
                [-0.0351074, -0.06568078, -0.04262076],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
