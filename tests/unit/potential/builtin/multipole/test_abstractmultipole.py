"""Test AbstractMultipolePotential."""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Shaped

import quaxed.numpy as qnp
from unxt import Quantity

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

    pot_cls: type[gp.AbstractPotential]

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

        fields["Slm"] = Quantity(Slm, "")
        pot = pot_cls(**fields)
        assert isinstance(pot.Slm, gp.params.ConstantParameter)
        assert qnp.allclose(pot.Slm.value, Quantity(Slm, ""))

    def test_Slm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Slm = jnp.zeros((l_max + 1, l_max + 1))
        Slm = Slm.at[1, 0].set(5.0)

        fields["Slm"] = Slm
        pot = pot_cls(**fields)
        assert qnp.allclose(pot.Slm(t=Quantity(0, "Myr")), Slm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_Slm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Slm = jnp.zeros((l_max + 1, l_max + 1))
        Slm = Slm.at[1, 0].set(5.0)

        fields["Slm"] = lambda t: Slm * qnp.exp(-qnp.abs(t))
        pot = pot_cls(**fields)
        assert qnp.allclose(pot.Slm(t=Quantity(0, "Myr")), Slm)


class ParameterTlmMixin(ParameterAngularCoefficientsMixin):
    """Test the Tlm parameter."""

    pot_cls: type[gp.AbstractPotential]

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

        fields["Tlm"] = Quantity(Tlm, "")
        fields["l_max"] = l_max
        pot = pot_cls(**fields)
        assert isinstance(pot.Tlm, gp.params.ConstantParameter)
        assert qnp.allclose(pot.Tlm.value, Quantity(Tlm, ""))

    def test_Tlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Tlm = jnp.zeros((l_max + 1, l_max + 1))
        Tlm = Tlm.at[1, 0].set(5.0)

        fields["Tlm"] = Tlm
        pot = pot_cls(**fields)
        assert qnp.allclose(pot.Tlm(t=Quantity(0, "Myr")), Tlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_Tlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        Tlm = jnp.zeros((l_max + 1, l_max + 1))
        Tlm = Tlm.at[1, :].set(5.0)

        fields["Tlm"] = lambda t: Tlm * qnp.exp(-qnp.abs(t))
        pot = pot_cls(**fields)
        assert qnp.allclose(pot.Tlm(t=Quantity(0, "Myr")), Tlm)
