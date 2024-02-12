from typing import Any

import array_api_jax_compat as xp
import astropy.units as u
import jax.numpy as jnp
import pytest
from quax import quaxify

from galax.potential import AbstractPotentialBase, BarPotential
from galax.typing import Vec3
from galax.units import UnitSystem

from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import (
    MassParameterMixin,
    ShapeAParameterMixin,
    ShapeBParameterMixin,
    ShapeCParameterMixin,
)

allclose = quaxify(jnp.allclose)


class TestBarPotential(
    AbstractPotential_Test,
    # Parameters
    MassParameterMixin,
    ShapeAParameterMixin,
    ShapeBParameterMixin,
    ShapeCParameterMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[BarPotential]:
        return BarPotential

    @pytest.fixture(scope="class")
    def field_Omega(self) -> dict[str, Any]:
        return 0

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_c: u.Quantity,
        field_Omega: u.Quantity,
        field_units: UnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "a": field_a,
            "b": field_b,
            "c": field_c,
            "Omega": field_Omega,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot: BarPotential, x: Vec3) -> None:
        assert jnp.isclose(pot.potential_energy(x, t=0), xp.asarray(-0.94601574))

    def test_gradient(self, pot: BarPotential, x: Vec3) -> None:
        assert jnp.allclose(
            pot.gradient(x, t=0), xp.asarray([0.04011905, 0.08383918, 0.16552719])
        )

    def test_density(self, pot: BarPotential, x: Vec3) -> None:
        assert jnp.isclose(pot.density(x, t=0).value, 1.94669274e08)

    def test_hessian(self, pot: BarPotential, x: Vec3) -> None:
        assert jnp.allclose(
            pot.hessian(x, t=0),
            xp.asarray(
                [
                    [0.03529841, -0.01038389, -0.02050134],
                    [-0.01038389, 0.0195721, -0.04412159],
                    [-0.02050134, -0.04412159, -0.04386589],
                ]
            ),
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = [
            [0.03163021, -0.01038389, -0.02050134],
            [-0.01038389, 0.01590389, -0.04412159],
            [-0.02050134, -0.04412159, -0.04753409],
        ]
        assert allclose(pot.tidal_tensor(x, t=0), xp.asarray(expect))
