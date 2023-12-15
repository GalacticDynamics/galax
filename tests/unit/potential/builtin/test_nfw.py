from typing import Any

import astropy.units as u
import jax.numpy as xp
import pytest

import galax.potential as gp
from galax.potential import ConstantParameter
from galax.units import galactic

from ..params.test_field import ParameterFieldMixin
from ..test_core import TestAbstractPotential
from .test_common import MassParameterMixin


class ScaleRadiusParameterMixin(ParameterFieldMixin):
    """Test the mass parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_r_s(self) -> float:
        return 1.0 * u.kpc

    # =====================================================

    def test_r_s_units(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["r_s"] = 1.0 * u.Unit(10 * u.kpc)
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.r_s, ConstantParameter)
        assert xp.isclose(pot.r_s.value, 10)

    def test_r_s_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["r_s"] = 1.0
        pot = pot_cls(**fields)
        assert pot.r_s(t=0) == 1.0

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_s_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["r_s"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.r_s(t=0) == 2


class TestNFWPotential(
    TestAbstractPotential,
    # Parameters
    MassParameterMixin,
    ScaleRadiusParameterMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.NFWPotential]:
        return gp.NFWPotential

    @pytest.fixture(scope="class")
    def field_softening_length(self) -> dict[str, Any]:
        return 0.001

    @pytest.fixture(scope="class")
    def fields_(
        self, field_m, field_r_s, field_softening_length, field_units
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "softening_length": field_softening_length,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-1.87117234))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.0658867, 0.1317734, 0.19766011])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 9.46039849e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.05558809, -0.02059723, -0.03089585],
                    [-0.02059723, 0.02469224, -0.06179169],
                    [-0.03089585, -0.06179169, -0.02680084],
                ]
            ),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))
