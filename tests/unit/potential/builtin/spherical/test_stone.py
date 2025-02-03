from typing import Any

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractSinglePotential_Test
from ..test_common import ParameterFieldMixin, ParameterMTotMixin
from galax.potential import AbstractPotential, StoneOstriker15Potential


class ParameterRCMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_c(self) -> u.Quantity["length"]:
        return u.Quantity(1.0, "kpc")

    # =====================================================

    def test_r_c_constant(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    def test_r_c_userfunc(self, pot_cls, fields):
        """Test the `r_c` parameter."""

        def cos_r_c(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_c"] = cos_r_c
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


class ParameterRHMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_h(self) -> u.Quantity["length"]:
        return u.Quantity(10.0, "kpc")

    # =====================================================

    def test_r_h_constant(self, pot_cls, fields):
        """Test the `r_h` parameter."""
        fields["r_h"] = u.Quantity(11.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_h(t=u.Quantity(0, "Myr")) == u.Quantity(11.0, "kpc")

    def test_r_h_userfunc(self, pot_cls, fields):
        """Test the `r_h` parameter."""

        def cos_r_h(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_h"] = cos_r_h
        pot = pot_cls(**fields)
        assert pot.r_h(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


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

    def test_potential(self, pot: StoneOstriker15Potential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-0.51579523, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: StoneOstriker15Potential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.01379378, 0.02758755, 0.04138133], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: StoneOstriker15Potential, x: gt.QuSz3) -> None:
        expect = u.Quantity(3.25886848e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: StoneOstriker15Potential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.01215385, -0.00327986, -0.00491978],
                [-0.00327986, 0.00723406, -0.00983957],
                [-0.00491978, -0.00983957, -0.00096558],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.00601307, -0.00327986, -0.00491978],
                [-0.00327986, 0.00109329, -0.00983957],
                [-0.00491978, -0.00983957, -0.00710635],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
