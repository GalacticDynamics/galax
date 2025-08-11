from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMTotMixin, ParameterRSMixin


class TestGaussianPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.GaussianPotential]:
        return gp.GaussianPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m_tot, field_r_s, field_units) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-1.20205548, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [0.08562732, 0.17125464, 0.25688196], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(57898701.53591853, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.06751239, -0.03622985, -0.05434478],
                [-0.03622985, 0.01316762, -0.10868955],
                [-0.05434478, -0.10868955, -0.07740701],
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
                [0.06642139, -0.03622985, -0.05434478],
                [-0.03622985, 0.01207662, -0.10868955],
                [-0.05434478, -0.10868955, -0.07849801],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
