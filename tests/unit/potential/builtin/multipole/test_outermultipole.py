"""Test the `MultipoleOuterPotential` class."""

from typing import Any

import astropy.units as u
import pytest
from jaxtyping import Array, Shaped
from plum import convert
from typing_extensions import override

import quaxed.numpy as qnp
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin, ParameterScaleRadiusMixin
from .test_abstractmultipole import ParameterSlmMixin, ParameterTlmMixin
from galax.utils._optional_deps import HAS_GALA

###############################################################################


class TestMultipoleOuterPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterScaleRadiusMixin,
    ParameterSlmMixin,
    ParameterTlmMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.MultipoleOuterPotential]:
        return gp.MultipoleOuterPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_s: u.Quantity,
        field_l_max: int,
        field_Slm: Shaped[Array, "3 3"],
        field_Tlm: Shaped[Array, "3 3"],
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_s": field_r_s,
            "l_max": field_l_max,
            "Slm": field_Slm,
            "Tlm": field_Tlm,
            "units": field_units,
        }

    # ==========================================================================

    def test_check_init(
        self, pot_cls: type[gp.MultipoleInnerPotential], fields_: dict[str, Any]
    ) -> None:
        """Test the `MultipoleInnerPotential.__check_init__` method."""
        fields_["Slm"] = fields_["Slm"][::2]  # make it the wrong shape
        with pytest.raises(ValueError, match="Slm and Tlm must have the shape"):
            pot_cls(**fields_)

    # ==========================================================================

    def test_potential(self, pot: gp.MultipoleOuterPotential, x: gt.QVec3) -> None:
        expect = Quantity(0.62939434, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.MultipoleOuterPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [-0.13487022, -0.26974043, -0.19481253], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), Quantity)
        assert qnp.allclose(got, expect, atol=Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.MultipoleOuterPotential, x: gt.QVec3) -> None:
        expect = Quantity(4.90989768e-07, unit="solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.MultipoleOuterPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [-0.08670228, 0.09633587, 0.09954706],
                [0.09633587, 0.05780152, 0.19909413],
                [0.09954706, 0.19909413, 0.02890076],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [-0.08670228, 0.09633587, 0.09954706],
                [0.09633587, 0.05780152, 0.19909413],
                [0.09954706, 0.19909413, 0.02890076],
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
            ("density", "density", 6e-7),  # TODO: get gala and galax to agree
            ("hessian", "hessian", 1e0),  # TODO: THIS IS BAD!!
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
