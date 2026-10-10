"""Test :mod:`galax.potential._src.param.core`."""

import pathlib
import tempfile

from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import pytest

import unxt as u

from galax.potential._src.params.base import ParameterCallable
from galax.potential.params import AbstractParameter, ConstantParameter, CustomParameter

T = TypeVar("T", bound=AbstractParameter)


class TestAbstractParameter(Generic[T]):
    """Test the `galax.potential.AbstractParameter` class."""

    @pytest.fixture(scope="class")
    def param_cls(self) -> type[T]:
        return AbstractParameter

    @pytest.fixture(scope="class")
    def field_unit(self) -> u.AbstractUnit:
        return u.unit("km")

    @pytest.fixture(scope="class")
    def param(self, param_cls: type[T], field_unit: u.AbstractUnit) -> T:
        class TestParameter(param_cls):
            unit: u.AbstractUnit

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
        return u.Q(1.0, field_unit)

    @pytest.fixture(scope="class")
    def param(
        self, param_cls: type[T], field_unit: u.AbstractUnit, field_value: float
    ) -> T:
        return param_cls(u.Q.from_(field_value, unit=field_unit))

    # ===============================================================

    def test_call(self, param: T, field_value: float) -> None:
        """Test `galax.potential.ConstantParameter` call method."""
        assert param(t=1.0) == field_value
        assert param(t=u.Q(1.0, "s")) == field_value

    def test_mul(self, param: T, field_value: float) -> None:
        """Test `galax.potential.ConstantParameter` multiplication."""
        expected = 2 * field_value
        assert param * 2 == expected
        assert 2 * param == expected


##############################################################################


class TestParameterCallable:
    """Test the `galax.potential.ParameterCallable` class."""

    def test_issubclass(self) -> None:
        assert issubclass(AbstractParameter, ParameterCallable)
        assert issubclass(ConstantParameter, ParameterCallable)
        assert issubclass(CustomParameter, AbstractParameter)

    def test_issubclass_false(self) -> None:
        assert not issubclass(object, ParameterCallable)

    def test_isinstance(self) -> None:
        assert isinstance(ConstantParameter(u.Q(1.0, "km")), ParameterCallable)
        assert isinstance(
            CustomParameter(lambda t: u.Q.from_(t, "km")), ParameterCallable
        )


class TestCustomParameter(TestAbstractParameter[CustomParameter]):
    """Test the `galax.potential.CustomParameter` class."""

    @pytest.fixture(scope="class")
    def param_cls(self) -> type[T]:
        return CustomParameter

    @pytest.fixture(scope="class")
    def field_func(self) -> ParameterCallable:
        def func(t: u.Quantity["time"], **kwargs: Any) -> Any:
            return u.Q(u.ustrip("Gyr", t), "kpc")

        return func

    @pytest.fixture(scope="class")
    def param(
        self,
        param_cls: type[T],
        field_unit: u.AbstractUnit,
        field_func: ParameterCallable,
    ) -> T:
        return param_cls(field_func)

    # ===============================================================

    def test_call(self, param: T) -> None:
        """Test :class:`galax.potential.CustomParameter` call method."""
        assert param(t=u.Q(1.0, "Gyr")) == u.Q(1.0, "kpc")

        # TODO: sort out what this tests
        # assert param(t=u.Q(1.0, u.unit("s"))) == u.Q(0.97779222, "km")

        # t = jnp.asarray([1.0, 2.0])
        # assert array_equal(param(t=t), t)


class TestCustomParameterData:
    """`args` and `kwargs` carry data as leaves; a closure cannot."""

    @staticmethod
    def _scaled(t, m0):
        return m0 * u.ustrip(u.unit("Gyr"), t)

    def test_data_in_args_are_pytree_leaves(self) -> None:
        """A closure hides its arrays from JAX; `args` does not.

        This is the whole reason `args` exists. `func` is a *static* field,
        so anything captured in a closure is part of the pytree structure
        rather than its leaves -- which costs a recompile per rebuild and,
        worse, writes nothing on serialisation.
        """
        m0 = u.Q(1e9, "Msun")

        def closed(t):
            return m0 * u.ustrip(u.unit("Gyr"), t)

        closure = CustomParameter(func=closed)
        carried = CustomParameter(func=self._scaled, args=(m0,))

        assert len(jax.tree_util.tree_leaves(eqx.filter(closure, eqx.is_array))) == 0
        assert len(jax.tree_util.tree_leaves(eqx.filter(carried, eqx.is_array))) == 1
        # Same answer either way; only visibility to JAX differs.
        assert closure(u.Q(2.0, "Gyr")) == carried(u.Q(2.0, "Gyr"))

    def test_a_carried_parameter_survives_serialisation(self) -> None:
        """REGRESSION: a closed-over table serialised to a zero-length file."""
        m0 = u.Q(1e9, "Msun")
        p = CustomParameter(func=self._scaled, args=(m0,))

        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "p.eqx"
            eqx.tree_serialise_leaves(path, p)
            assert path.stat().st_size > 0
            blank = CustomParameter(func=self._scaled, args=(u.Q(0.0, "Msun"),))
            back = eqx.tree_deserialise_leaves(path, blank)

        assert back(u.Q(2.0, "Gyr")) == p(u.Q(2.0, "Gyr"))

    def test_rebuilding_over_the_same_data_does_not_retrace(self) -> None:
        """A closure is hashed by identity, so rebuilding it recompiles."""
        m0 = u.Q(1e9, "Msun")
        traces = [0]

        @eqx.filter_jit
        def ev(p, t):
            traces[0] += 1
            return p(t)

        for _ in range(3):
            ev(CustomParameter(func=self._scaled, args=(m0,)), u.Q(2.0, "Gyr"))
        assert traces[0] == 1

    def test_call_site_keywords_override_stored_ones(self) -> None:
        """Stored keywords are defaults, not a second hidden call site."""

        def ramp(t, *, m0, rate):
            return m0 + rate * u.ustrip(u.unit("Gyr"), t)

        p = CustomParameter(
            func=ramp, kwargs={"m0": u.Q(1e9, "Msun"), "rate": u.Q(1e9, "Msun")}
        )
        assert p(u.Q(2.0, "Gyr")) == u.Q(3e9, "Msun")
        assert p(u.Q(2.0, "Gyr"), rate=u.Q(0.0, "Msun")) == u.Q(1e9, "Msun")
