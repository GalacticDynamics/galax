from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMTotMixin,
    ParameterShapeHRMixin,
    ParameterShapeHZMixin,
)
from galax.potential.custom_types import Sz3


class TestMN3ExponentialPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeHRMixin,
    ParameterShapeHZMixin,
):
    """Test the `galax.potential.MN3ExponentialPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MN3ExponentialPotential]:
        return gp.MN3ExponentialPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_h_R: u.Quantity,
        field_h_z: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "h_R": field_h_R,
            "h_z": field_h_z,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.MN3ExponentialPotential, x: Sz3) -> None:
        expect = u.Q(-1.15401718, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/mn3/exponential", atol=1e-8
    )
    def test_gradient(self, pot: gp.MN3ExponentialPotential, x: Sz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.MN3ExponentialPotential, x: Sz3) -> None:
        expect = u.Q(731_782_542.3781165, pot.units["mass density"])
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/mn3/exponential", atol=1e-8
    )
    def test_hessian(self, pot: gp.MN3ExponentialPotential, x: Sz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/mn3/exponential", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: Sz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")


class TestMN3Sech2Potential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeHRMixin,
    ParameterShapeHZMixin,
):
    """Test the `galax.potential.MN3Sech2Potential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MN3Sech2Potential]:
        return gp.MN3Sech2Potential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_h_R: u.Quantity,
        field_h_z: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "h_R": field_h_R,
            "h_z": field_h_z,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.MN3Sech2Potential, x: Sz3) -> None:
        expect = u.Q(-1.13545211, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/mn3/sech2", atol=1e-8
    )
    def test_gradient(self, pot: gp.MN3Sech2Potential, x: Sz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.MN3Sech2Potential, x: Sz3) -> None:
        expect = u.Q(211_769_063.98948175, pot.units["mass density"])
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/mn3/sech2", atol=1e-8
    )
    def test_hessian(self, pot: gp.MN3Sech2Potential, x: Sz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/mn3/sech2", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: Sz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
