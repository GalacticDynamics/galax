from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMTotMixin, ParameterRSMixin


class TestPlummerPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
):
    """Test the `galax.potential.PlummerPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.PlummerPotential]:
        return gp.PlummerPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_s: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.PlummerPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-1.16150826, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.PlummerPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.07743388, 0.15486777, 0.23230165], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.PlummerPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(2.73957531e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.PlummerPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.06194711, -0.03097355, -0.04646033],
                [-0.03097355, 0.01548678, -0.09292066],
                [-0.04646033, -0.09292066, -0.06194711],
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
                [0.05678485, -0.03097355, -0.04646033],
                [-0.03097355, 0.01032452, -0.09292066],
                [-0.04646033, -0.09292066, -0.06710937],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
