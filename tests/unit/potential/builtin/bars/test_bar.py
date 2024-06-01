from typing import Any, ClassVar

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import (
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
    ParameterShapeCMixin,
)
from galax.potential import AbstractPotentialBase, BarPotential


class TestBarPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
    ParameterShapeCMixin,
):
    """Test the `galax.potential.BarPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[BarPotential]:
        return BarPotential

    @pytest.fixture(scope="class")
    def field_Omega(self) -> Quantity["frequency"]:
        return Quantity(0, "Hz")

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_c: u.Quantity,
        field_Omega: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "a": field_a,
            "b": field_b,
            "c": field_c,
            "Omega": field_Omega,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: BarPotential, x: gt.QVec3) -> None:
        expect = Quantity(-0.94601574, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: BarPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [0.04011905, 0.08383918, 0.16552719], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: BarPotential, x: gt.QVec3) -> None:
        expect = Quantity(1.94669274e08, "Msun / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
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
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.03163021, -0.01038389, -0.02050134],
                [-0.01038389, 0.01590389, -0.04412159],
                [-0.02050134, -0.04412159, -0.04753409],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
