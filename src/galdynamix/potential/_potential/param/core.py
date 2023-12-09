"""Parameters on a Potential."""

__all__ = ["AbstractParameter", "ConstantParameter", "UserParameter"]

import abc
from dataclasses import KW_ONLY
from typing import Any, Protocol, runtime_checkable

import equinox as eqx

from galdynamix.typing import (
    BatchableFloatOrIntScalarLike,
    FloatArrayAnyShape,
    FloatOrIntScalar,
    FloatScalar,
    Unit,
)
from galdynamix.utils import partial_jit, vectorize_method
from galdynamix.utils.dataclasses import converter_float_array


class AbstractParameter(eqx.Module):  # type: ignore[misc]
    """Abstract Base Class for Parameters on a Potential.

    Parameters are time-dependent quantities that are used to define a
    Potential. They can be constant (see `ConstantParameter`), or they can be
    functions of time.

    Parameters
    ----------
    unit : Unit
        The output unit of the parameter.
    """

    _: KW_ONLY
    unit: Unit = eqx.field(static=True, converter=u.Unit)

    @abc.abstractmethod
    def __call__(self, t: FloatScalar, **kwargs: Any) -> FloatArrayAnyShape:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : Array
            The time(s) at which to compute the parameter value.
        **kwargs
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array
            The parameter value at times ``t``.
        """
        ...


class ConstantParameter(AbstractParameter):
    """Time-independent potential parameter."""

    # TODO: unit handling
    # TODO: link this shape to the return shape from __call__
    value: FloatArrayAnyShape = eqx.field(converter=converter_float_array)

    # This is a workaround since vectorized methods don't support kwargs.
    @partial_jit()
    @vectorize_method(signature="()->()")
    def _call_helper(self, _: FloatOrIntScalar) -> FloatArrayAnyShape:
        return self.value

    @partial_jit()
    def __call__(
        self, t: BatchableFloatOrIntScalarLike = 0, **kwargs: Any
    ) -> FloatArrayAnyShape:
        """Return the constant parameter value.

        Parameters
        ----------
        t : Array, optional
            This is ignored and is thus optional.
            Note that for most :class:`~galdynamix.potential.AbstractParameter`
            the time is required.
        **kwargs : Any
            This is ignored.

        Returns
        -------
        Array
            The constant parameter value.
        """
        return self._call_helper(t)


#####################################################################
# User-defined Parameter
# For passing a function as a parameter.


@runtime_checkable
class ParameterCallable(Protocol):
    """Protocol for a Parameter callable."""

    def __call__(self, t: FloatScalar, **kwargs: Any) -> FloatArrayAnyShape:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : Array
            Time(s) at which to compute the parameter value.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array
            Parameter value(s) at the given time(s).
        """
        ...


class UserParameter(AbstractParameter):
    """User-defined Parameter.

    Parameters
    ----------
    func : Callable[[Array[float, ()] | float | int], Array[float, (*shape,)]]
        The function to use to compute the parameter value.
    unit : Unit, keyword-only
        The output unit of the parameter.
    """

    # TODO: unit handling
    func: ParameterCallable = eqx.field(static=True)

    @partial_jit()
    def __call__(self, t: FloatOrIntScalar, **kwargs: Any) -> FloatArrayAnyShape:
        return self.func(t, **kwargs)
