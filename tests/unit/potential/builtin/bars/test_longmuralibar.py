from typing import Any

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...param.test_field import ParameterFieldMixin
from ...test_core import AbstractPotential_Test
from ..test_common import (
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
    ParameterShapeCMixin,
)
from galax.potential import AbstractPotentialBase, LongMuraliBarPotential


class AlphaParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_alpha(self) -> Quantity["angle"]:
        return Quantity(0.9, "rad")

    # =====================================================

    def test_alpha_constant(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = Quantity(1.0, "rad")
        pot = pot_cls(**fields)
        assert pot.alpha(t=0) == Quantity(1.0, "rad")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_alpha_userfunc(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.alpha(t=0) == 2


class TestLongMuraliBarPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
    ParameterShapeCMixin,
    AlphaParameterMixin,
):
    """Test the `galax.potential.LongMuraliBarPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.LongMuraliBarPotential]:
        return gp.LongMuraliBarPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_c: u.Quantity,
        field_alpha: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "alpha": field_alpha,
            "a": field_a,
            "b": field_b,
            "c": field_c,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: LongMuraliBarPotential, x: gt.QVec3) -> None:
        expect = Quantity(-0.9494695, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: LongMuraliBarPotential, x: gt.QVec3) -> None:
        expect = Quantity([0.04017315, 0.08220449, 0.16854858], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: LongMuraliBarPotential, x: gt.QVec3) -> None:
        expect = Quantity(2.02402357e08, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: LongMuraliBarPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.03722412, -0.01077521, -0.02078279],
                [-0.01077521, 0.02101076, -0.04320745],
                [-0.02078279, -0.04320745, -0.0467931],
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
                [0.0334102, -0.01077521, -0.02078279],
                [-0.01077521, 0.01719683, -0.04320745],
                [-0.02078279, -0.04320745, -0.05060703],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
