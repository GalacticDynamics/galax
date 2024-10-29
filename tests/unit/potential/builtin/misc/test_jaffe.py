from typing import Any

import astropy.units as u
import pytest
from plum import convert

import quaxed.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMMixin, ParameterScaleRadiusMixin
from galax.potential import AbstractBasePotential, JaffePotential


class TestJaffePotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterScaleRadiusMixin,
):
    """Test the `galax.potential.JaffePotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.JaffePotential]:
        return gp.JaffePotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: JaffePotential, x: gt.QVec3) -> None:
        expect = Quantity(-1.06550653, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: JaffePotential, x: gt.QVec3) -> None:
        expect = Quantity([0.06776567, 0.13553134, 0.20329701], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), Quantity)
        assert jnp.allclose(got, expect, atol=Quantity(1e-8, expect.unit))

    def test_density(self, pot: JaffePotential, x: gt.QVec3) -> None:
        expect = Quantity(2.52814372e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: JaffePotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.05426528, -0.02700078, -0.04050117],
                [-0.02700078, 0.01376411, -0.08100233],
                [-0.04050117, -0.08100233, -0.05373783],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractBasePotential, x: gt.QVec3) -> None:
        """Test the `AbstractBasePotential.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.04950143, -0.02700078, -0.04050117],
                [-0.02700078, 0.00900026, -0.08100233],
                [-0.04050117, -0.08100233, -0.05850169],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
