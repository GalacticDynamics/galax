from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ...test_core import AbstractSinglePotential_Test
from ..test_common import ParameterMTotMixin
from galax._custom_types import QuSz3


class TestKeplerPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.KeplerPotential]:
        return gp.KeplerPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m_tot, field_units) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.KeplerPotential, x: QuSz3) -> None:
        expect = u.Quantity(-1.20227527, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.KeplerPotential, x: QuSz3) -> None:
        expect = u.Quantity(
            [0.08587681, 0.17175361, 0.25763042], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.KeplerPotential, x: QuSz3) -> None:
        expect = u.Quantity(0.0, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.KeplerPotential, x: QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.06747463, -0.03680435, -0.05520652],
                [-0.03680435, 0.01226812, -0.11041304],
                [-0.05520652, -0.11041304, -0.07974275],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.06747463, -0.03680435, -0.05520652],
                [-0.03680435, 0.01226812, -0.11041304],
                [-0.05520652, -0.11041304, -0.07974275],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
