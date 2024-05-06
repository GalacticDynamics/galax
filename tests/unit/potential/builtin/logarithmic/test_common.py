import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import Quantity
from unxt.unitsystems import galactic

import galax.potential as gp
from ...param.test_field import ParameterFieldMixin
from galax.potential import ConstantParameter


class ParameterVCMixin(ParameterFieldMixin):
    """Test the circular velocity parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_v_c(self) -> Quantity["speed"]:
        return Quantity(220, "km/s")

    # =====================================================

    def test_v_c_units(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["v_c"] = Quantity(1.0, u.Unit(220 * u.km / u.s))
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.v_c, ConstantParameter)
        assert pot.v_c.value == Quantity(220, "km/s")

    def test_v_c_constant(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["v_c"] = Quantity(1.0, "km/s")
        pot = pot_cls(**fields)
        assert pot.v_c(t=0) == Quantity(1.0, "km/s")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_v_c_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["v_c"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.v_c(t=0) == 2


class ParameterRSMixin(ParameterFieldMixin):
    """Test the scale radius parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_r_s(self) -> Quantity["length"]:
        return Quantity(8, "kpc")

    # =====================================================

    def test_r_s_units(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["r_s"] = Quantity(1, u.Unit(10 * u.kpc))
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.r_s, ConstantParameter)
        assert qnp.isclose(
            pot.r_s.value, Quantity(10, "kpc"), atol=Quantity(1e-15, "kpc")
        )

    def test_r_s_constant(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["r_s"] = Quantity(11.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_s(t=0) == Quantity(11.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_s_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["r_s"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.r_s(t=0) == 2
