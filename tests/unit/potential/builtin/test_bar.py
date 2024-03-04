from typing import Any

import astropy.units as u
import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp
from unxt import Quantity

import galax.typing as gt
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import (
    MassParameterMixin,
    ShapeAParameterMixin,
    ShapeBParameterMixin,
    ShapeCParameterMixin,
)
from galax.potential import AbstractPotentialBase, BarPotential
from galax.units import UnitSystem


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
    def field_Omega(self) -> Quantity["frequency"]:
        return Quantity(0, "Hz")

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

    def test_potential_energy(self, pot: BarPotential, x: gt.Vec3) -> None:
        expect = Quantity(-0.94601574, pot.units["specific energy"])
        assert qnp.isclose(  # TODO: .value & use pytest-arraydiff
            pot.potential_energy(x, t=0).decompose(pot.units).value,
            expect.value,
        )

    def test_gradient(self, pot: BarPotential, x: gt.Vec3) -> None:
        expected = Quantity(
            [0.04011905, 0.08383918, 0.16552719], pot.units["acceleration"]
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.gradient(x, t=0).decompose(pot.units).value, expected.value
        )

    def test_density(self, pot: BarPotential, x: gt.Vec3) -> None:
        expected = Quantity(1.94669274e08, "Msun / kpc3")
        assert jnp.isclose(  # TODO: .value & use pytest-arraydiff
            pot.density(x, t=0).decompose(pot.units).value, expected.value
        )

    def test_hessian(self, pot: BarPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.03529841, -0.01038389, -0.02050134],
                [-0.01038389, 0.0195721, -0.04412159],
                [-0.02050134, -0.04412159, -0.04386589],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.hessian(x, t=0).decompose(pot.units).value, expect.value
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.03163021, -0.01038389, -0.02050134],
                [-0.01038389, 0.01590389, -0.04412159],
                [-0.02050134, -0.04412159, -0.04753409],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.tidal_tensor(x, t=0).decompose(pot.units).value,
            expect.value,
        )
