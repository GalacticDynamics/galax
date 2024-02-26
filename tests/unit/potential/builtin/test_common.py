import astropy.units as u
import pytest

import galax.potential as gp
from ..param.test_field import ParameterFieldMixin
from galax.potential import ConstantParameter
from galax.units import galactic


class MassParameterMixin(ParameterFieldMixin):
    """Test the mass parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_m(self) -> u.Quantity:
        return 1e12 * u.Msun

    # =====================================================

    def test_m_units(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = 1.0 * u.Unit(10 * u.Msun)
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.m, ConstantParameter)
        assert pot.m.value == 10

    def test_m_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = 1.0
        pot = pot_cls(**fields)
        assert pot.m(t=0) == 1.0

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_m_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.m(t=0) == 2


class ShapeAParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a(self) -> float:
        return 1.0

    # =====================================================

    def test_a_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["a"] = 1.0
        pot = pot_cls(**fields)
        assert pot.a(t=0) == 1.0

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_a_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["a"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.a(t=0) == 2


class ShapeBParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_b(self) -> float:
        return 1.0

    # =====================================================

    def test_b_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["b"] = 1.0
        pot = pot_cls(**fields)
        assert pot.b(t=0) == 1.0

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_b_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["b"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.b(t=0) == 2


class ShapeCParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_c(self) -> float:
        return 1.0

    # =====================================================

    def test_c_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["c"] = 1.0
        pot = pot_cls(**fields)
        assert pot.c(t=0) == 1.0

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_c_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["c"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.c(t=0) == 2
