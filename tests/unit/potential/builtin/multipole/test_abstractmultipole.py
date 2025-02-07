"""Test AbstractMultipolePotential."""

import pytest
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ...param.test_field import ParameterFieldMixin


class ParameterAngularCoefficientsMixin(ParameterFieldMixin):
    """Test the angular coefficients."""

    @pytest.fixture(scope="class")
    def field_l_max(self) -> int:
        """l_max static field."""
        return 3


class ParameterSlmMixin(ParameterAngularCoefficientsMixin):
    """Test the Slm parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_Slm(self, field_l_max) -> Shaped[Array, "3 3"]:
        """Slm parameter."""
        Slm = jnp.zeros((field_l_max + 1, field_l_max + 1))
        Slm = Slm.at[1, 0].set(5.0)
        return Slm

    # =====================================================

    def test_Slm_units(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Slm = jnp.zeros((l_max + 1, l_max + 1))
        Slm = Slm.at[1, :].set(5.0)

        fields["Slm"] = u.Quantity(Slm, "")
        pot = pot_cls(**fields)
        assert isinstance(pot.Slm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.Slm.value, u.Quantity(Slm, ""))

    def test_Slm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Slm = jnp.zeros((l_max + 1, l_max + 1))
        Slm = Slm.at[1, 0].set(5.0)

        fields["Slm"] = Slm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.Slm(t=u.Quantity(0, "Myr")), Slm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_Slm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Slm = jnp.zeros((l_max + 1, l_max + 1))
        Slm = Slm.at[1, 0].set(5.0)

        fields["Slm"] = lambda t: Slm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.Slm(t=u.Quantity(0, "Myr")), Slm)


class ParameterTlmMixin(ParameterAngularCoefficientsMixin):
    """Test the Tlm parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_Tlm(self, field_l_max) -> Shaped[Array, "3 3"]:
        """Tlm parameter."""
        return jnp.zeros((field_l_max + 1, field_l_max + 1))

    # =====================================================

    def test_Tlm_units(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Tlm = jnp.zeros((l_max + 1, l_max + 1))
        Tlm = Tlm.at[1, :].set(5.0)

        fields["Tlm"] = u.Quantity(Tlm, "")
        fields["l_max"] = l_max
        pot = pot_cls(**fields)
        assert isinstance(pot.Tlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.Tlm.value, u.Quantity(Tlm, ""))

    def test_Tlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Tlm = jnp.zeros((l_max + 1, l_max + 1))
        Tlm = Tlm.at[1, 0].set(5.0)

        fields["Tlm"] = Tlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.Tlm(t=u.Quantity(0, "Myr")), Tlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_Tlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Tlm = jnp.zeros((l_max + 1, l_max + 1))
        Tlm = Tlm.at[1, :].set(5.0)

        fields["Tlm"] = lambda t: Tlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.Tlm(t=u.Quantity(0, "Myr")), Tlm)
