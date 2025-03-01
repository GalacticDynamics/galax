from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ...test_core import AbstractSinglePotential_Test
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
        expect = u.Quantity(0.11027593, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.00064902, 0.00129804, 0.00194706], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(30321621.61178864, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [6.32377766e-04, -3.32830403e-05, -4.99245605e-05],
                [-3.32830403e-05, 5.82453206e-04, -9.98491210e-05],
                [-4.99245605e-05, -9.98491210e-05, 4.99245605e-04],
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
                [6.10189073e-05, -3.32830403e-05, -4.99245605e-05],
                [-3.32830403e-05, 1.10943468e-05, -9.98491210e-05],
                [-4.99245605e-05, -9.98491210e-05, -7.21132541e-05],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
