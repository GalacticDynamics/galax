from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMTotMixin, ParameterShapeAMixin, ParameterShapeBMixin
from galax.potential.custom_types import Sz3


class TestMiyamotoNagaiPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
):
    """Test the `galax.potential.MiyamotoNagaiPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MiyamotoNagaiPotential]:
        return gp.MiyamotoNagaiPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "a": field_a, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.MiyamotoNagaiPotential, x: Sz3) -> None:
        expect = u.Q(-0.95208676, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/miyamotonagai", atol=1e-8
    )
    def test_gradient(self, pot: gp.MiyamotoNagaiPotential, x: Sz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.MiyamotoNagaiPotential, x: Sz3) -> None:
        expect = u.Q(1.9949418e08, pot.units["mass density"])
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/miyamotonagai", atol=1e-8
    )
    def test_hessian(self, pot: gp.MiyamotoNagaiPotential, x: Sz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/miyamotonagai", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: Sz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
