"""Test :mod:`galax.potential._potential.param.core`."""


import astropy.units as u
import jax.experimental.array_api as xp
import jaxtyping
import pytest
from jax.numpy import array_equal

from galax.potential import AbstractParameter, ConstantParameter, UserParameter
from galax.potential._potential.param.core import ParameterCallable
from galax.typing import Unit


class TestAbstractParameter:
    """Test the `galax.potential.AbstractParameter` class."""

    @pytest.fixture(scope="class")
    def param_cls(self) -> type[AbstractParameter]:
        return AbstractParameter

    @pytest.fixture(scope="class")
    def field_unit(self) -> Unit:
        return u.km

    @pytest.fixture(scope="class")
    def param(self, param_cls, field_unit) -> AbstractParameter:
        class TestParameter(param_cls):
            unit: Unit

            def __call__(self, t, **kwargs):
                return t

        return TestParameter(unit=field_unit)

    # ==========================================================================

    def test_init(self):
        """Test init `galax.potential.AbstractParameter` method."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            AbstractParameter()

    def test_call(self):
        """Test `galax.potential.AbstractParameter` call method."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            AbstractParameter()()

    def test_unit_field(self, param, field_unit):
        """Test `galax.potential.AbstractParameter` unit field."""
        assert param.unit == field_unit


# ==============================================================================


class TestConstantParameter(TestAbstractParameter):
    """Test the `galax.potential.ConstantParameter` class."""

    @pytest.fixture(scope="class")
    def param_cls(self) -> type[AbstractParameter]:
        return ConstantParameter

    @pytest.fixture(scope="class")
    def field_value(self) -> float:
        return 1.0

    @pytest.fixture(scope="class")
    def param(self, param_cls, field_unit, field_value) -> AbstractParameter:
        return param_cls(field_value, unit=field_unit)

    # ==========================================================================

    def test_call(self, param, field_value):
        """Test `galax.potential.ConstantParameter` call method."""
        assert param(t=1.0) == field_value
        assert param(t=1.0 * u.s) == field_value
        assert array_equal(param(t=xp.asarray([1.0, 2.0])), [field_value, field_value])


# ==============================================================================


class TestParameterCallable:
    """Test the `galax.potential.ParameterCallable` class."""

    def test_issubclass(self):
        assert issubclass(AbstractParameter, ParameterCallable)
        assert issubclass(ConstantParameter, ParameterCallable)
        assert issubclass(UserParameter, AbstractParameter)

    def test_issubclass_false(self):
        assert not issubclass(object, ParameterCallable)

    def test_isinstance(self):
        assert isinstance(ConstantParameter(1.0, unit=u.km), ParameterCallable)
        assert isinstance(UserParameter(lambda t: t, unit=u.km), ParameterCallable)


class TestUserParameter(TestAbstractParameter):
    """Test the `galax.potential.UserParameter` class."""

    @pytest.fixture(scope="class")
    def param_cls(self) -> type[AbstractParameter]:
        return UserParameter

    @pytest.fixture(scope="class")
    def field_func(self) -> float:
        def func(t, **kwargs):
            return t

        return func

    @pytest.fixture(scope="class")
    def param(self, param_cls, field_unit, field_func) -> AbstractParameter:
        return param_cls(field_func, unit=field_unit)

    # ==========================================================================

    def test_call(self, param):
        """Test `galax.potential.UserParameter` call method."""
        assert param(t=1.0) == 1.0
        assert param(t=1.0 * u.s) == 1.0 * u.s

        t = xp.asarray([1.0, 2.0])
        with pytest.raises(
            jaxtyping.TypeCheckError,
            match="Type-check error whilst checking the parameters of __call__",
        ):
            array_equal(param(t=t), t)
