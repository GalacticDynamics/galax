"""Test the `galax.potential.Vogelsberger08TriaxialNFWPotential` class."""

from typing import Any, ClassVar

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ...param.test_field import ParameterFieldMixin
from ...test_core import AbstractSinglePotential_Test
from ..test_common import (
    ParameterMMixin,
    ParameterScaleRadiusMixin,
    ParameterShapeQ1Mixin,
)


class ShapeTransitionRadiusParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a_r(self) -> u.Quantity["dimensionless"]:
        return u.Quantity(1.0, "")

    # =====================================================

    def test_a_r_constant(self, pot_cls, fields):
        """Test the `a_r` parameter."""
        fields["a_r"] = u.Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a_r(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "")

    def test_a_r_userfunc(self, pot_cls, fields):
        """Test the `a_r` parameter."""

        def cos_a_r(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "")

        fields["a_r"] = cos_a_r
        pot = pot_cls(**fields)
        assert pot.a_r(t=u.Quantity(0, "Myr")) == u.Quantity(10, "")


class TestVogelsberger08TriaxialNFWPotential(
    AbstractSinglePotential_Test,
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
        field_units: u.AbstractUnitSystem,
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
        self, pot: gp.Vogelsberger08TriaxialNFWPotential, x: gt.QuSz3
    ) -> None:
        expect = u.Quantity(-1.91410199, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(
        self, pot: gp.Vogelsberger08TriaxialNFWPotential, x: gt.QuSz3
    ) -> None:
        expect = u.Quantity([0.07701115, 0.14549116, 0.19849185], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(
        self, pot: gp.Vogelsberger08TriaxialNFWPotential, x: gt.QuSz3
    ) -> None:
        expect = u.Quantity(1.10157433e09, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(
        self, pot: gp.Vogelsberger08TriaxialNFWPotential, x: gt.QuSz3
    ) -> None:
        expect = u.Quantity(
            [
                [0.06195284, -0.0274773, -0.0351074],
                [-0.0274773, 0.02218247, -0.06568078],
                [-0.0351074, -0.06568078, -0.02186349],
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
                [0.04119557, -0.0274773, -0.0351074],
                [-0.0274773, 0.00142519, -0.06568078],
                [-0.0351074, -0.06568078, -0.04262076],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
