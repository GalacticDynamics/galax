import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ...param.test_field import ParameterFieldMixin
from galax.potential.params import ConstantParameter


class ParameterVCMixin(ParameterFieldMixin):
    """Test the circular velocity parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_v_c(self) -> u.Quantity["speed"]:
        return u.Quantity(220, "km/s")

    # =====================================================

    def test_v_c_units(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["v_c"] = u.Quantity(1.0, u.unit(220 * u.unit("km / s")))
        fields["units"] = u.unitsystems.galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.v_c, ConstantParameter)
        assert pot.v_c.value == u.Quantity(220, "km/s")

    def test_v_c_constant(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["v_c"] = u.Quantity(1.0, "km/s")
        pot = pot_cls(**fields)
        assert pot.v_c(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "km/s")

    def test_v_c_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""

        def cos_v_c(t: u.Quantity["time"]) -> u.Quantity["speed"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "km/s")

        fields["v_c"] = cos_v_c
        pot = pot_cls(**fields)
        assert pot.v_c(t=u.Quantity(0, "Myr")) == u.Quantity(10, "km/s")


class ParameterRSMixin(ParameterFieldMixin):
    """Test the scale radius parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_r_s(self) -> u.Quantity["length"]:
        return u.Quantity(8, "kpc")

    # =====================================================

    def test_r_s_units(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["r_s"] = u.Quantity(1, u.unit(10 * u.unit("kpc")))
        fields["units"] = u.unitsystems.galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.r_s, ConstantParameter)
        assert jnp.isclose(
            pot.r_s.value, u.Quantity(10, "kpc"), atol=u.Quantity(1e-15, "kpc")
        )

    def test_r_s_constant(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["r_s"] = u.Quantity(11.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_s(t=u.Quantity(0, "Myr")) == u.Quantity(11.0, "kpc")

    def test_r_s_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""

        def cos_r_s(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_s"] = cos_r_s
        pot = pot_cls(**fields)
        assert pot.r_s(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")
