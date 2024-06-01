from typing import Any

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from .test_common import ParameterRSMixin, ParameterVCMixin
from galax.potential import AbstractPotentialBase, LogarithmicPotential


class TestLogarithmicPotential(
    AbstractPotential_Test,
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
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"v_c": field_v_c, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(0.11027593, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity([0.00064902, 0.00129804, 0.00194706], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(30321621.61178864, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [6.32377766e-04, -3.32830403e-05, -4.99245605e-05],
                [-3.32830403e-05, 5.82453206e-04, -9.98491210e-05],
                [-4.99245605e-05, -9.98491210e-05, 4.99245605e-04],
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
                [6.10189073e-05, -3.32830403e-05, -4.99245605e-05],
                [-3.32830403e-05, 1.10943468e-05, -9.98491210e-05],
                [-4.99245605e-05, -9.98491210e-05, -7.21132541e-05],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
