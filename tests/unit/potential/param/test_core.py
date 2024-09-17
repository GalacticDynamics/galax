"""Test :mod:`galax.potential._src.param.core`."""

from dataclasses import replace
from typing import Any, Generic, TypeVar

import astropy.units as u
import pytest

import quaxed.numpy as jnp
from unxt import Quantity, ustrip

from galax.potential._src.params.core import ParameterCallable
from galax.potential.params import AbstractParameter, ConstantParameter, UserParameter
from galax.typing import Unit

T = TypeVar("T", bound=AbstractParameter)


class TestAbstractParameter(Generic[T]):
    """Test the `galax.potential.AbstractParameter` class."""

    @pytest.fixture(scope="class")
    def param_cls(self) -> type[T]:
        return AbstractParameter

    @pytest.fixture(scope="class")
    def field_unit(self) -> Unit:
        return u.km

    @pytest.fixture(scope="class")
    def param(self, param_cls: type[T], field_unit: Unit) -> T:
        class TestParameter(param_cls):
            unit: Unit

            def __call__(self, t: Any, **kwargs: Any) -> Any:
                return t

        return TestParameter(unit=field_unit)

    # ===============================================================

    def test_init(self, param_cls) -> None:
        """Test init `galax.potential.AbstractParameter` method."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            param_cls()

    def test_call(self, param_cls) -> None:
        """Test `galax.potential.AbstractParameter` call method."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            param_cls()()


##############################################################################


class TestConstantParameter(TestAbstractParameter[ConstantParameter]):
    """Test the `galax.potential.ConstantParameter` class."""

    @pytest.fixture(scope="class")
    def param_cls(self) -> type[T]:
        return ConstantParameter

    @pytest.fixture(scope="class")
    def field_value(self, field_unit) -> float:
        return Quantity(1.0, field_unit)

    @pytest.fixture(scope="class")
    def param(self, param_cls: type[T], field_unit: Unit, field_value: float) -> T:
        return param_cls(Quantity.constructor(field_value, unit=field_unit))

    # ===============================================================

    def test_call(self, param: T, field_value: float) -> None:
        """Test `galax.potential.ConstantParameter` call method."""
        assert param(t=1.0) == field_value
        assert param(t=1.0 * u.s) == field_value
        # Calling the parameter doesn't broadcast the shape, only the number of
        # dimensions
        assert jnp.array_equal(param(t=jnp.asarray([1.0, 2.0])), [field_value])

    def test_mul(self, param: T, field_value: float) -> None:
        """Test `galax.potential.ConstantParameter` multiplication."""
        expected = replace(param, value=field_value * 2)
        assert param * 2 == expected
        assert 2 * param == expected


##############################################################################


class TestParameterCallable:
    """Test the `galax.potential.ParameterCallable` class."""

    def test_issubclass(self) -> None:
        assert issubclass(AbstractParameter, ParameterCallable)
        assert issubclass(ConstantParameter, ParameterCallable)
        assert issubclass(UserParameter, AbstractParameter)

    def test_issubclass_false(self) -> None:
        assert not issubclass(object, ParameterCallable)

    def test_isinstance(self) -> None:
        assert isinstance(ConstantParameter(Quantity(1.0, "km")), ParameterCallable)
        assert isinstance(UserParameter(lambda t: t << u.km), ParameterCallable)


class TestUserParameter(TestAbstractParameter[UserParameter]):
    """Test the `galax.potential.UserParameter` class."""

    @pytest.fixture(scope="class")
    def param_cls(self) -> type[T]:
        return UserParameter

    @pytest.fixture(scope="class")
    def field_func(self) -> ParameterCallable:
        def func(t: Quantity["time"], **kwargs: Any) -> Any:
            return Quantity(ustrip("Gyr", t), "kpc")

        return func

    @pytest.fixture(scope="class")
    def param(
        self, param_cls: type[T], field_unit: Unit, field_func: ParameterCallable
    ) -> T:
        return param_cls(field_func)

    # ===============================================================

    def test_call(self, param: T) -> None:
        """Test :class:`galax.potential.UserParameter` call method."""
        assert param(t=Quantity(1.0, "Gyr")) == Quantity(1.0, "kpc")

        # TODO: sort out what this tests
        # assert param(t=Quantity(1.0, u.s)) == Quantity(0.97779222, "km")

        # t = jnp.asarray([1.0, 2.0])
        # assert array_equal(param(t=t), t)
