from typing import Any

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterFieldMixin, ParameterMTotMixin
from galax.potential import AbstractPotentialBase, StoneOstriker15Potential


class ParameterRCMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_c(self) -> Quantity["length"]:
        return Quantity(1.0, "kpc")

    # =====================================================

    def test_r_c_constant(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_c(t=0) == Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_c_userfunc(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a1(t=0) == 2


class ParameterRHMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_h(self) -> Quantity["length"]:
        return Quantity(10.0, "kpc")

    # =====================================================

    def test_r_h_constant(self, pot_cls, fields):
        """Test the `r_h` parameter."""
        fields["r_h"] = Quantity(11.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_h(t=0) == Quantity(11.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_h_userfunc(self, pot_cls, fields):
        """Test the `r_h` parameter."""
        fields["r_h"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a1(t=0) == 2


class TestStoneOstriker15Potential(
    AbstractPotential_Test,
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
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_c": field_r_c,
            "r_h": field_r_h,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: StoneOstriker15Potential, x: gt.QVec3) -> None:
        expect = Quantity(-0.51579523, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: StoneOstriker15Potential, x: gt.QVec3) -> None:
        expect = Quantity([0.01379378, 0.02758755, 0.04138133], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: StoneOstriker15Potential, x: gt.QVec3) -> None:
        expect = Quantity(3.25886848e08, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: StoneOstriker15Potential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.01215385, -0.00327986, -0.00491978],
                [-0.00327986, 0.00723406, -0.00983957],
                [-0.00491978, -0.00983957, -0.00096558],
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
                [0.00601307, -0.00327986, -0.00491978],
                [-0.00327986, 0.00109329, -0.00983957],
                [-0.00491978, -0.00983957, -0.00710635],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
