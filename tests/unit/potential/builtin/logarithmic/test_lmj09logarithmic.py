from typing import Any

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...param.test_field import ParameterFieldMixin
from ...test_core import AbstractPotential_Test
from ..test_common import (
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
    ParameterShapeQ3Mixin,
)
from .test_common import ParameterRSMixin, ParameterVCMixin
from galax.potential import (
    AbstractPotentialBase,
    ConstantParameter,
    LMJ09LogarithmicPotential,
)
from galax.utils._optional_deps import HAS_GALA


class ParameterPhiMixin(ParameterFieldMixin):
    """Test the phi parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_phi(self) -> Quantity["angle"]:
        return Quantity(220, "deg")

    # =====================================================

    def test_phi_units(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["phi"] = Quantity(1.0, u.Unit(220 * u.deg))
        pot = pot_cls(**fields)
        assert isinstance(pot.phi, ConstantParameter)
        assert pot.phi.value == Quantity(220, "deg")

    def test_phi_constant(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["phi"] = Quantity(1.0, "deg")
        pot = pot_cls(**fields)
        assert pot.phi(t=0) == Quantity(1.0, "deg")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_phi_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["phi"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.phi(t=0) == 2


class TestLMJ09LogarithmicPotential(
    AbstractPotential_Test,
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
        field_units: AbstractUnitSystem,
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

    def test_potential(self, pot: LMJ09LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(0.11819267, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: LMJ09LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity([-0.00046885, 0.00181093, 0.00569646], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: LMJ09LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(48995543.34035844, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: LMJ09LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.00100608, -0.00070826, 0.00010551],
                [-0.00070826, 0.00114681, -0.00040755],
                [0.00010551, -0.00040755, 0.00061682],
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
                [8.28469691e-05, -7.08263497e-04, 1.05514716e-04],
                [-7.08263497e-04, 2.23569293e-04, -4.07553647e-04],
                [1.05514716e-04, -4.07553647e-04, -3.06416262e-04],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ==========================================================================
    # Interoperability

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
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
        pot: gp.AbstractPotentialBase,
        method0: str,
        method1: str,
        x: gt.QVec3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        super().test_method_gala(pot, method0, method1, x, atol)
