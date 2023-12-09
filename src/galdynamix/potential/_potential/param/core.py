"""Parameters on a Potential."""

__all__ = ["AbstractParameter", "ConstantParameter", "UserParameter"]

import abc
from dataclasses import KW_ONLY
from typing import Any, Protocol, runtime_checkable

import astropy.units as u
import equinox as eqx

from galdynamix.typing import (
    ArrayAnyShape,
    BatchableFloatOrIntScalarLike,
    FloatArrayAnyShape,
    FloatOrIntScalar,
    FloatScalar,
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
    unit: u.Unit = eqx.field(static=True)  # TODO: move this to an annotation?

    @abc.abstractmethod
    def __call__(self, t: FloatScalar, **kwargs: Any) -> ArrayAnyShape:
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

    @partial_jit()
    @vectorize_method(signature="()->()")
    def _call_helper(self, _: FloatOrIntScalar) -> ArrayAnyShape:
        return self.value

    @partial_jit()
    def __call__(
        self, t: BatchableFloatOrIntScalarLike = 0, **kwargs: Any
    ) -> ArrayAnyShape:
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


@runtime_checkable
class ParameterCallable(Protocol):
    """Protocol for a Parameter callable."""

    def __call__(self, t: FloatScalar, **kwargs: Any) -> ArrayAnyShape:
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
    """User-defined Parameter."""

    # TODO: unit handling
    func: ParameterCallable

    @partial_jit()
    def __call__(self, t: FloatScalar, **kwargs: Any) -> ArrayAnyShape:
        return self.func(t, **kwargs)
