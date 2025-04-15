from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.params as gpp
from ..param.test_field import ParameterFieldMixin

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
        fields["units"] = u.unitsystems.galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.m_tot, gpp.ConstantParameter)
        assert pot.m_tot.value == u.Quantity(10, "Msun")

    def test_m_tot_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m_tot"] = u.Quantity(1.0, "Msun")
        pot = pot_cls(**fields)
        assert pot.m_tot(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "Msun")

    def test_m_tot_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""

        def cos_mass(t: u.Quantity["time"]) -> u.Quantity["mass"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "Msun")

        fields["m_tot"] = cos_mass
        pot = pot_cls(**fields)
        assert pot.m_tot(t=u.Quantity(0, "Myr")) == u.Quantity(10, "Msun")


class ParameterMMixin(ParameterFieldMixin):
    """Test the characteristic mass parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_m(self) -> u.Quantity["mass"]:
        return u.Quantity(1e12, "Msun")

    # =====================================================

    def test_m_units(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = u.Quantity(1.0, u.unit(10 * u.unit("Msun")))
        fields["units"] = u.unitsystems.galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.m, gpp.ConstantParameter)
        assert pot.m.value == u.Quantity(10, "Msun")

    def test_m_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = u.Quantity(1.0, "Msun")
        pot = pot_cls(**fields)
        assert pot.m(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "Msun")

    def test_m_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""

        def cos_mass(t: u.Quantity["time"]) -> u.Quantity["mass"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "Msun")

        fields["m"] = cos_mass
        pot = pot_cls(**fields)
        assert pot.m(t=u.Quantity(0, "Myr")) == u.Quantity(10, "Msun")


# =============================================================================
# Scale Radius


class ParameterRSMixin(ParameterFieldMixin):
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
        fields["units"] = u.unitsystems.galactic
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

    def test_r_s_userfunc(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ):
        """Test the scale radius parameter."""

        def cos_scalelength(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_s"] = cos_scalelength
        pot = pot_cls(**fields)
        assert pot.r_s(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


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

    def test_q1_userfunc(self, pot_cls, fields):
        """Test the q1 parameter."""

        def cos_q1(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "")

        fields["q1"] = cos_q1
        pot = pot_cls(**fields)
        assert pot.q1(t=u.Quantity(0, "Myr")) == u.Quantity(10, "")


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

    def test_q2_userfunc(self, pot_cls, fields):
        """Test the q2 parameter."""

        def cos_q2(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "")

        fields["q2"] = cos_q2
        pot = pot_cls(**fields)
        assert pot.q2(t=u.Quantity(0, "Myr")) == u.Quantity(10, "")


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

    def test_q3_userfunc(self, pot_cls, fields):
        """Test the q3 parameter."""

        def cos_q3(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "")

        fields["q3"] = cos_q3
        pot = pot_cls(**fields)
        assert pot.q3(t=u.Quantity(0, "Myr")) == u.Quantity(10, "")


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

    def test_a_userfunc(self, pot_cls, fields):
        """Test the `a` parameter."""

        def cos_a(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["a"] = cos_a
        pot = pot_cls(**fields)
        assert pot.a(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


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

    def test_b_userfunc(self, pot_cls, fields):
        """Test the `b` parameter."""

        def cos_b(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["b"] = cos_b
        pot = pot_cls(**fields)
        assert pot.b(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


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

    def test_c_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""

        def cos_c(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["c"] = cos_c
        pot = pot_cls(**fields)
        assert pot.c(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


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

    def test_h_R_userfunc(self, pot_cls, fields):
        """Test the `h_R` parameter."""

        def cos_h_R(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["h_R"] = cos_h_R
        pot = pot_cls(**fields)
        assert pot.h_R(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


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

    def test_h_z_userfunc(self, pot_cls, fields):
        """Test the `h_z` parameter."""

        def cos_h_z(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["h_z"] = cos_h_z
        pot = pot_cls(**fields)
        assert pot.h_z(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


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
        assert isinstance(pot.v_c, gpp.ConstantParameter)
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
