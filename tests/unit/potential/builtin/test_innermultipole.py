"""Test the `MultipoleInnerPotential` class."""

import sys
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


class TestMultipoleInnerPotential(
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
    def pot_cls(self) -> type[gp.MultipoleInnerPotential]:
        return gp.MultipoleInnerPotential

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

    def test_potential(self, pot: gp.MultipoleInnerPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(32.96969177, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.MultipoleInnerPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [4.74751335e-16, 9.49502670e-16, 10.9898973], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.MultipoleInnerPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            3.04941279e-05 if sys.platform == "darwin" else 2.89194575e-05,
            unit="solMass / kpc3",
        )

        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.MultipoleInnerPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [3.81496608e-16, -1.86509453e-16, 7.62993217e-17],
                [-1.86509453e-16, 1.01732429e-16, 1.52598643e-16],
                [-3.78931294e-16, -7.57862587e-16, 1.15158342e-15],
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
                [-1.63440876e-16, -1.86509453e-16, 7.62993217e-17],
                [-1.86509453e-16, -4.43205056e-16, 1.52598643e-16],
                [-3.78931294e-16, -7.57862587e-16, 6.06645933e-16],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
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
            ("density", "density", 3e-5),  # TODO: get gala and galax to agree
            ("hessian", "hessian", 1e-8),  # TODO: get gala and galax to agree
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
