"""Test the `MultipoleOuterPotential` class."""

from typing import Any
from typing_extensions import override

import equinox as eqx
import pytest
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_abstractmultipole import (
    MultipoleTestMixin,
    ParameterSlmMixin,
    ParameterTlmMixin,
)
from .test_common import ParameterMTotMixin, ParameterRSMixin
from galax._interop.optional_deps import GSL_ENABLED, OptDeps

###############################################################################


class TestMultipoleOuterPotential(
    MultipoleTestMixin,
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
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
        field_units: u.AbstractUnitSystem,
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
        match = "Slm and Tlm must have the shape"
        with pytest.raises(eqx.EquinoxRuntimeError, match=match):
            pot_cls(**fields_)

    # ==========================================================================

    def test_potential(self, pot: gp.MultipoleOuterPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(0.62939434, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), exp, atol=u.Quantity(1e-8, exp.unit))

    def test_gradient(self, pot: gp.MultipoleOuterPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(
            [-0.13487022, -0.26974043, -0.19481253], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_density(self, pot: gp.MultipoleOuterPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(0, unit="solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), exp, atol=u.Quantity(1e-8, exp.unit))

    def test_hessian(self, pot: gp.MultipoleOuterPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(
            [
                [-0.08670228, 0.09633587, 0.09954706],
                [0.09633587, 0.05780152, 0.19909413],
                [0.09954706, 0.19909413, 0.02890076],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(pot.hessian(x, t=0), exp, atol=u.Quantity(1e-8, exp.unit))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        exp = u.Quantity(
            [
                [-0.08670228, 0.09633587, 0.09954706],
                [0.09633587, 0.05780152, 0.19909413],
                [0.09954706, 0.19909413, 0.02890076],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), exp, atol=u.Quantity(1e-8, exp.unit)
        )

    # ==========================================================================
    # Interoperability

    @pytest.mark.skipif(
        not OptDeps.GALA.installed or not GSL_ENABLED, reason="requires gala + GSL"
    )
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
