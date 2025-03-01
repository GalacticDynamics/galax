from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ...test_core import AbstractSinglePotential_Test
from ..test_common import ParameterMTotMixin, ParameterScaleRadiusMixin


class TestHernquistPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterScaleRadiusMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.HernquistPotential]:
        return gp.HernquistPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m_tot, field_r_s, field_units) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.HernquistPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-0.94871936, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.HernquistPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [0.05347411, 0.10694822, 0.16042233], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.HernquistPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(3.989933e08, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.HernquistPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.04362645, -0.01969533, -0.02954299],
                [-0.01969533, 0.01408345, -0.05908599],
                [-0.02954299, -0.05908599, -0.03515487],
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
                [0.0361081, -0.01969533, -0.02954299],
                [-0.01969533, 0.00656511, -0.05908599],
                [-0.02954299, -0.05908599, -0.04267321],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
