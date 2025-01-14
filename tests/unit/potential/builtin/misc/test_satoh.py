from typing import Any

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin, ParameterShapeAMixin, ParameterShapeBMixin
from galax.potential import AbstractBasePotential, SatohPotential


class TestSatohPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
):
    """Test the `galax.potential.SatohPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.SatohPotential]:
        return gp.SatohPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "a": field_a,
            "b": field_b,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: SatohPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-0.97415472, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: SatohPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.0456823, 0.0913646, 0.18038493], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: SatohPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(1.08825455e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: SatohPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.03925558, -0.01285344, -0.02537707],
                [-0.01285344, 0.01997543, -0.05075415],
                [-0.02537707, -0.05075415, -0.05307912],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractBasePotential, x: gt.QuSz3) -> None:
        """Test the `AbstractBasePotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.03720495, -0.01285344, -0.02537707],
                [-0.01285344, 0.0179248, -0.05075415],
                [-0.02537707, -0.05075415, -0.05512975],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
