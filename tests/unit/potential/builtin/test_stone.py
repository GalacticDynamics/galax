from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterFieldMixin, ParameterMTotMixin


class ParameterRCMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_c(self) -> u.Quantity["length"]:
        return u.Q(1.0, "kpc")

    # =====================================================

    def test_r_c_constant(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = u.Q(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Q(0, "Myr")) == u.Q(1.0, "kpc")

    def test_r_c_userfunc(self, pot_cls, fields):
        """Test the `r_c` parameter."""

        def cos_r_c(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_c"] = cos_r_c
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Q(0, "Myr")) == u.Q(10, "kpc")


class ParameterRHMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_h(self) -> u.Quantity["length"]:
        return u.Q(10.0, "kpc")

    # =====================================================

    def test_r_h_constant(self, pot_cls, fields):
        """Test the `r_h` parameter."""
        fields["r_h"] = u.Q(11.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_h(t=u.Q(0, "Myr")) == u.Q(11.0, "kpc")

    def test_r_h_userfunc(self, pot_cls, fields):
        """Test the `r_h` parameter."""

        def cos_r_h(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_h"] = cos_r_h
        pot = pot_cls(**fields)
        assert pot.r_h(t=u.Q(0, "Myr")) == u.Q(10, "kpc")


class TestStoneOstriker15Potential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRCMixin,
    ParameterRHMixin,
):
    """Test the `galax.potential.StoneOstriker15Potential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.StoneOstriker15Potential]:
        return gp.StoneOstriker15Potential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_c: u.Quantity,
        field_r_h: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_c": field_r_c,
            "r_h": field_r_h,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.StoneOstriker15Potential, x: gt.QuSz3) -> None:
        expect = u.Q(-0.51579523, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/stone", atol=1e-8
    )
    def test_gradient(self, pot: gp.StoneOstriker15Potential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.StoneOstriker15Potential, x: gt.QuSz3) -> None:
        expect = u.Q(3.25886848e08, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/stone", atol=1e-8
    )
    def test_hessian(self, pot: gp.StoneOstriker15Potential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/stone", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
