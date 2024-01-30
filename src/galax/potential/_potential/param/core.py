"""Parameters on a Potential."""

__all__ = ["AbstractParameter", "ConstantParameter", "UserParameter"]

import abc
from dataclasses import KW_ONLY, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import astropy.units as u
import equinox as eqx
import jax

from galax.typing import (
    BatchableFloatOrIntScalarLike,
    FloatArrayAnyShape,
    FloatOrIntScalar,
    FloatOrIntScalarLike,
    FloatScalar,
    Unit,
)
from galax.utils._jax import vectorize_method
from galax.utils.dataclasses import converter_float_array

if TYPE_CHECKING:
    from typing import Self


class AbstractParameter(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
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
    def __call__(self, t: FloatOrIntScalarLike, **kwargs: Any) -> FloatArrayAnyShape:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : Array[float | int, ()] | float | int
            The time(s) at which to compute the parameter value.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array[float, "*shape"]
            The parameter value at times ``t``.
        """
        ...


class ConstantParameter(AbstractParameter):
    """Time-independent potential parameter."""

    # TODO: unit handling
    # TODO: link this shape to the return shape from __call__
    value: FloatArrayAnyShape = eqx.field(converter=converter_float_array)
    _: KW_ONLY
    unit: Unit = eqx.field(static=True, converter=u.Unit)

    # This is a workaround since vectorized methods don't support kwargs.
    @partial(jax.jit)
    @vectorize_method(signature="()->()")
    def _call_helper(self, _: FloatOrIntScalar) -> FloatArrayAnyShape:
        return self.value

    @partial(jax.jit)
    def __call__(
        self, t: BatchableFloatOrIntScalarLike = 0, **kwargs: Any
    ) -> FloatArrayAnyShape:
        """Return the constant parameter value.

        Parameters
        ----------
        t : float | Array[float, ()], optional
            This is ignored and is thus optional.
            Note that for most :class:`~galax.potential.AbstractParameter`
            the time is required.
        **kwargs : Any
            This is ignored.

        Returns
        -------
        Array[float, "*shape"]
            The constant parameter value.
        """
        return self._call_helper(t)

    # -------------------------------------------

    def __mul__(self, other: Any) -> "Self":
        return replace(self, value=self.value * other)

    def __rmul__(self, other: Any) -> "Self":
        return replace(self, value=other * self.value)


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
        t : float | Array[float, ()]
            Time(s) at which to compute the parameter value.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array[float, "*shape"]
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
    _: KW_ONLY
    unit: Unit = eqx.field(static=True, converter=u.Unit)

    @partial(jax.jit)
    def __call__(self, t: FloatOrIntScalar, **kwargs: Any) -> FloatArrayAnyShape:
        return self.func(t, **kwargs)
