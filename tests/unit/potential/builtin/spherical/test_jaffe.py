from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ...test_core import AbstractSinglePotential_Test
from ..test_common import ParameterMMixin, ParameterScaleRadiusMixin


class TestJaffePotential(
    AbstractSinglePotential_Test,
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
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.JaffePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-1.06550653, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.JaffePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.06776567, 0.13553134, 0.20329701], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.JaffePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(2.52814372e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.JaffePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.05426528, -0.02700078, -0.04050117],
                [-0.02700078, 0.01376411, -0.08100233],
                [-0.04050117, -0.08100233, -0.05373783],
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
                [0.04950143, -0.02700078, -0.04050117],
                [-0.02700078, 0.00900026, -0.08100233],
                [-0.04050117, -0.08100233, -0.05850169],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
