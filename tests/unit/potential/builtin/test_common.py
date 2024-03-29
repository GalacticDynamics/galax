import astropy.units as u
import pytest

from unxt import Quantity
from unxt.unitsystems import galactic

import galax.potential as gp
from ..param.test_field import ParameterFieldMixin
from galax.potential import ConstantParameter


class MassParameterMixin(ParameterFieldMixin):
    """Test the mass parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_m(self) -> Quantity["mass"]:
        return Quantity(1e12, "Msun")

    # =====================================================

    def test_m_units(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = Quantity(1.0, u.Unit(10 * u.Msun))
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.m, ConstantParameter)
        assert pot.m.value == Quantity(10, "Msun")

    def test_m_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = Quantity(1.0, "Msun")
        pot = pot_cls(**fields)
        assert pot.m(t=0) == Quantity(1.0, "Msun")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_m_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["m"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.m(t=0) == 2


class ShapeAParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a(self) -> Quantity["length"]:
        return Quantity(1.0, "kpc")

    # =====================================================

    def test_a_constant(self, pot_cls, fields):
        """Test the `a` parameter."""
        fields["a"] = Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.a(t=0) == Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_a_userfunc(self, pot_cls, fields):
        """Test the `a` parameter."""
        fields["a"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a(t=0) == 2


class ShapeBParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_b(self) -> Quantity["length"]:
        return Quantity(1.0, "kpc")

    # =====================================================

    def test_b_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["b"] = Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.b(t=0) == Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_b_userfunc(self, pot_cls, fields):
        """Test the `b` parameter."""
        fields["b"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.b(t=0) == 2


class ShapeCParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_c(self) -> Quantity["length"]:
        return Quantity(1.0, "kpc")

    # =====================================================

    def test_c_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["c"] = Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.c(t=0) == Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_c_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["c"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.c(t=0) == 2


class ShapeQ1ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_q1(self) -> float:
        return Quantity(1.1, "")

    # =====================================================

    def test_q1_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q1"] = Quantity(1.1, "")
        pot = pot_cls(**fields)
        assert pot.q1(t=0) == Quantity(1.1, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_q1_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q1"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.q1(t=0) == Quantity(1.2, "")


class ShapeQ2ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_q2(self) -> float:
        return Quantity(0.5, "")

    # =====================================================

    def test_q2_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q2"] = Quantity(0.6, "")
        pot = pot_cls(**fields)
        assert pot.q2(t=0) == Quantity(0.6, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_q2_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["q2"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.q2(t=0) == Quantity(1.2, "")
