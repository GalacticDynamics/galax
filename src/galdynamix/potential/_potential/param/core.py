"""Parameters on a Potential."""

__all__ = ["AbstractParameter", "ConstantParameter", "UserParameter"]

import abc
from dataclasses import KW_ONLY
from typing import Protocol, runtime_checkable

import astropy.units as u
import equinox as eqx

from galdynamix.typing import ArrayAnyShape, FloatScalar
from galdynamix.utils import partial_jit


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
    def __call__(self, t: FloatScalar) -> ArrayAnyShape:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : Array
            The time(s) at which to compute the parameter value.

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
    value: ArrayAnyShape

    @partial_jit()
    def __call__(self, t: FloatScalar = 0) -> ArrayAnyShape:
        """Return the constant parameter value.

        Parameters
        ----------
        t : Array, optional
            This is ignored and is thus optional.
            Note that for most :class:`~galdynamix.potential.AbstractParameter`
            the time is required.

        Returns
        -------
        Array
            The constant parameter value.
        """
        return self.value


#####################################################################
# User-defined Parameter


@runtime_checkable
class ParameterCallable(Protocol):
    """Protocol for a Parameter callable."""

    def __call__(self, t: FloatScalar) -> ArrayAnyShape:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : Array
            Time(s) at which to compute the parameter value.

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
    def __call__(self, t: FloatScalar) -> ArrayAnyShape:
        return self.func(t)
