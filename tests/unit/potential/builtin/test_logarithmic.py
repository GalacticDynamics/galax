from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterRSMixin, ParameterVCMixin


class TestLogarithmicPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterVCMixin,
    ParameterRSMixin,
):
    """Test the `galax.potential.LogarithmicPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.LogarithmicPotential]:
        return gp.LogarithmicPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_v_c: u.Quantity,
        field_r_s: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"v_c": field_v_c, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(0.0685455, unit="kpc2 / Myr2")
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_gradient(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity([0.0033749, 0.0067498, 0.0101247], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_density(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(67661373.89566506, "solMass / kpc3")
        got = pot.density(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_hessian(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(
            [
                [0.00292491, -0.00089997, -0.00134996],
                [-0.00089997, 0.00157495, -0.00269992],
                [-0.00134996, -0.00269992, -0.00067498],
            ],
            "1/Myr2",
        )
        got = pot.hessian(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        exp = u.Quantity(
            [
                [0.00164995, -0.00089997, -0.00134996],
                [-0.00089997, 0.00029999, -0.00269992],
                [-0.00134996, -0.00269992, -0.00194994],
            ],
            "1/Myr2",
        )
        got = pot.tidal_tensor(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))
