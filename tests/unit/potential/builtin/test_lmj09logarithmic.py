from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
    ParameterShapeQ3Mixin,
    ParameterVCMixin,
)
from galax._interop.optional_deps import OptDeps


class ParameterPhiMixin(ParameterFieldMixin):
    """Test the phi parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_phi(self) -> u.Quantity["angle"]:
        return u.Quantity(220, "deg")

    # =====================================================

    def test_phi_units(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["phi"] = u.Quantity(1.0, u.unit(220 * u.unit("deg")))
        pot = pot_cls(**fields)
        assert isinstance(pot.phi, gp.params.ConstantParameter)
        assert pot.phi.value == u.Quantity(220, "deg")

    def test_phi_constant(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["phi"] = u.Quantity(1.0, "deg")
        pot = pot_cls(**fields)
        assert pot.phi(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "deg")

    def test_phi_userfunc(self, pot_cls, fields):
        """Test the phi parameter."""

        def cos_phi(t: u.Quantity["time"]) -> u.Quantity["angle"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "deg")

        fields["phi"] = cos_phi
        pot = pot_cls(**fields)
        assert pot.phi(t=u.Quantity(0, "Myr")) == u.Quantity(10, "deg")


class TestLMJ09LogarithmicPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterVCMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
    ParameterShapeQ3Mixin,
    ParameterPhiMixin,
):
    """Test the `galax.potential.LMJ09LogarithmicPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.LMJ09LogarithmicPotential]:
        return gp.LMJ09LogarithmicPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_v_c: u.Quantity,
        field_r_s: u.Quantity,
        field_q1: u.Quantity,
        field_q2: u.Quantity,
        field_q3: u.Quantity,
        field_phi: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "v_c": field_v_c,
            "r_s": field_r_s,
            "q1": field_q1,
            "q2": field_q2,
            "q3": field_q3,
            "phi": field_phi,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.LMJ09LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(0.09557772, unit="kpc2 / Myr2")
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_gradient(self, pot: gp.LMJ09LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity([-0.00114565, 0.00442512, 0.01391965], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_density(self, pot: gp.LMJ09LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(31101011.36872738, "solMass / kpc3")
        got = pot.density(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_hessian(self, pot: gp.LMJ09LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(
            [
                [0.00242779, -0.00161236, 0.00063003],
                [-0.00161236, 0.00234527, -0.0024335],
                [0.00063003, -0.0024335, -0.00301492],
            ],
            "1/Myr2",
        )
        got = pot.hessian(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        exp = u.Quantity(
            [
                [0.00184175, -0.00161236, 0.00063003],
                [-0.00161236, 0.00175922, -0.0024335],
                [0.00063003, -0.0024335, -0.00360097],
            ],
            "1/Myr2",
        )
        got = pot.tidal_tensor(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    # ==========================================================================
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            # ("density", "density", 1e-8),  # TODO: get gala and galax to agree
            # ("hessian", "hessian", 1e-8),  # TODO: get gala and galax to agree
        ],
    )
    def test_method_gala(
        self,
        pot: gp.AbstractPotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        super().test_method_gala(pot, method0, method1, x, atol)
