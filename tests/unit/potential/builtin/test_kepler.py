from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMTotMixin
from galax.potential.custom_types import QuSz3


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
        exp = u.Q(-1.20227527, pot.units["specific energy"])
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/kepler", atol=1e-8
    )
    def test_gradient(self, pot: gp.KeplerPotential, x: QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.KeplerPotential, x: QuSz3) -> None:
        exp = u.Q(0.0, pot.units["mass density"])
        got = pot.density(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/kepler", atol=1e-8
    )
    def test_hessian(self, pot: gp.KeplerPotential, x: QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/kepler", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
