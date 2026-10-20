from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
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
        expect = u.Q(-1.16150826, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/plummer", atol=1e-8
    )
    def test_gradient(self, pot: gp.PlummerPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.PlummerPotential, x: gt.QuSz3) -> None:
        expect = u.Q(2.73957531e08, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/plummer", atol=1e-8
    )
    def test_hessian(self, pot: gp.PlummerPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/plummer", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
