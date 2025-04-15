from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMTotMixin, ParameterRSMixin


class TestIsochronePotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.IsochronePotential]:
        return gp.IsochronePotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m_tot, field_r_s, field_units) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.IsochronePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-0.9231515, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.IsochronePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [0.04891392, 0.09782784, 0.14674175], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.IsochronePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(5.04511665e08, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.IsochronePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.0404695, -0.01688883, -0.02533324],
                [-0.01688883, 0.01513626, -0.05066648],
                [-0.02533324, -0.05066648, -0.0270858],
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
                [0.03096285, -0.01688883, -0.02533324],
                [-0.01688883, 0.00562961, -0.05066648],
                [-0.02533324, -0.05066648, -0.03659246],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
