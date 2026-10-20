"""Test the `galax.potential.TriaxialNFWPotential` class."""

from typing import Any, ClassVar

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
)


class TestTriaxialNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
):
    """Test the `galax.potential.TriaxialNFWPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.TriaxialNFWPotential]:
        return gp.TriaxialNFWPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_q1: u.Quantity,
        field_q2: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "q1": field_q1,
            "q2": field_q2,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.TriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-1.06475915, unit="kpc2 / Myr2")
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/triaxialnfw", atol=1e-8
    )
    def test_gradient(self, pot: gp.TriaxialNFWPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.TriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Q(2.32106514e08, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/triaxialnfw", atol=1e-8
    )
    def test_hessian(self, pot: gp.TriaxialNFWPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/triaxialnfw", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
