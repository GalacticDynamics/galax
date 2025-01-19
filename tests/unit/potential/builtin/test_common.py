from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u
from unxt.unitsystems import galactic

import galax.potential as gp
import galax.potential.params as gpp
from ..param.test_field import ParameterFieldMixin
from galax.potential.params import ConstantParameter

# =============================================================================
# Mass


class ParameterMTotMixin(ParameterFieldMixin):
    """Test the total mass parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_m_tot(self) -> u.Quantity["mass"]:
        return u.Quantity(1e12, "Msun")

    # =====================================================

    def test_m_tot_units(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m_tot"] = u.Quantity(1.0, u.unit(10 * u.unit("Msun")))
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.m_tot, ConstantParameter)
        assert pot.m_tot.value == u.Quantity(10, "Msun")

    def test_m_tot_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m_tot"] = u.Quantity(1.0, "Msun")
        pot = pot_cls(**fields)
        assert pot.m_tot(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "Msun")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_m_tot_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m_tot"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.m_tot(t=u.Quantity(0, "Myr")) == 2


class ParameterMMixin(ParameterFieldMixin):
    """Test the total mass parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_m(self) -> u.Quantity["mass"]:
        return u.Quantity(1e12, "Msun")

    # =====================================================

    def test_m_units(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = u.Quantity(1.0, u.unit(10 * u.unit("Msun")))
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.m, ConstantParameter)
        assert pot.m.value == u.Quantity(10, "Msun")

    def test_m_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = u.Quantity(1.0, "Msun")
        pot = pot_cls(**fields)
        assert pot.m(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "Msun")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_m_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.m(t=u.Quantity(0, "Myr")) == 2


# =============================================================================
# Scale Radius


class ParameterScaleRadiusMixin(ParameterFieldMixin):
    """Test the mass parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_r_s(self) -> u.Quantity["length"]:
        return u.Quantity(1.0, "kpc")

    # =====================================================

    def test_r_s_units(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ) -> None:
        """Test the mass parameter."""
        fields["r_s"] = 1.0 * u.unit(10 * u.unit("kpc"))
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.r_s, gpp.ConstantParameter)
        assert jnp.isclose(
            pot.r_s(0), u.Quantity(10, "kpc"), atol=u.Quantity(1e-8, "kpc")
        )

    def test_r_s_constant(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ):
        """Test the mass parameter."""
        fields["r_s"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_s(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_s_userfunc(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ):
        """Test the mass parameter."""
        fields["r_s"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.r_s(t=u.Quantity(0, "Myr")) == 1.2


# =============================================================================
# Axis Ratios


class ParameterShapeQ1Mixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_q1(self) -> float:
        return u.Quantity(1.1, "")

    # =====================================================

    def test_q1_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q1"] = u.Quantity(1.1, "")
        pot = pot_cls(**fields)
        assert pot.q1(t=u.Quantity(0, "Myr")) == u.Quantity(1.1, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_q1_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q1"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.q1(t=u.Quantity(0, "Myr")) == u.Quantity(1.2, "")


class ParameterShapeQ2Mixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_q2(self) -> float:
        return u.Quantity(0.5, "")

    # =====================================================

    def test_q2_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q2"] = u.Quantity(0.6, "")
        pot = pot_cls(**fields)
        assert pot.q2(t=u.Quantity(0, "Myr")) == u.Quantity(0.6, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_q2_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q2"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.q2(t=u.Quantity(0, "Myr")) == u.Quantity(1.2, "")


class ParameterShapeQ3Mixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_q3(self) -> float:
        return u.Quantity(0.5, "")

    # =====================================================

    def test_q3_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q3"] = u.Quantity(0.6, "")
        pot = pot_cls(**fields)
        assert pot.q3(t=u.Quantity(0, "Myr")) == u.Quantity(0.6, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_q3_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q3"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.q3(t=u.Quantity(0, "Myr")) == u.Quantity(1.2, "")


# =============================================================================


class ParameterShapeAMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a(self) -> u.Quantity["length"]:
        return u.Quantity(1.0, "kpc")

    # =====================================================

    def test_a_constant(self, pot_cls, fields):
        """Test the `a` parameter."""
        fields["a"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.a(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_a_userfunc(self, pot_cls, fields):
        """Test the `a` parameter."""
        fields["a"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a(t=u.Quantity(0, "Myr")) == 2


class ParameterShapeBMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_b(self) -> u.Quantity["length"]:
        return u.Quantity(1.0, "kpc")

    # =====================================================

    def test_b_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["b"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.b(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_b_userfunc(self, pot_cls, fields):
        """Test the `b` parameter."""
        fields["b"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.b(t=u.Quantity(0, "Myr")) == 2


class ParameterShapeCMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_c(self) -> u.Quantity["length"]:
        return u.Quantity(1.0, "kpc")

    # =====================================================

    def test_c_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["c"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.c(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_c_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["c"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.c(t=u.Quantity(0, "Myr")) == 2


class ParameterShapeHRMixin(ParameterFieldMixin):
    """Test the radial scale length parameter."""

    @pytest.fixture(scope="class")
    def field_h_R(self) -> u.Quantity["length"]:
        return u.Quantity(1.0, "kpc")

    # =====================================================

    def test_h_R_constant(self, pot_cls, fields):
        """Test the `h_R` parameter."""
        fields["h_R"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.h_R(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_h_R_userfunc(self, pot_cls, fields):
        """Test the `h_R` parameter."""
        fields["h_R"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.h_R(t=u.Quantity(0, "Myr")) == 2


class ParameterShapeHZMixin(ParameterFieldMixin):
    """Test the scale height parameter."""

    @pytest.fixture(scope="class")
    def field_h_z(self) -> u.Quantity["length"]:
        return u.Quantity(1.0, "kpc")

    # =====================================================

    def test_h_z_constant(self, pot_cls, fields):
        """Test the `h_z` parameter."""
        fields["h_z"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.h_z(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_h_z_userfunc(self, pot_cls, fields):
        """Test the `h_z` parameter."""
        fields["h_z"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.h_z(t=u.Quantity(0, "Myr")) == 2
