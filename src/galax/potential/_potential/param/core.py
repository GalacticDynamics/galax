"""Parameters on a Potential."""

__all__ = [
    "ParameterCallable",
    "AbstractParameter",
    "ConstantParameter",
    "UserParameter",
]

import abc
from dataclasses import KW_ONLY, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, final, runtime_checkable

import astropy.units as u
import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import Quantity

from galax.typing import BatchableRealQScalar, FloatQAnyShape, Unit

if TYPE_CHECKING:
    from typing import Self


t0 = Quantity(0, "Myr")


@runtime_checkable
class ParameterCallable(Protocol):
    """Protocol for a Parameter callable."""

    def __call__(self, t: BatchableRealQScalar, **kwargs: Any) -> FloatQAnyShape:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : `~galax.typing.BatchableRealQScalar`
            Time(s) at which to compute the parameter value.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array[float, "*shape"]
            Parameter value(s) at the given time(s).
        """
        ...


class AbstractParameter(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class for Parameters on a Potential.

    Parameters are time-dependent quantities that are used to define a
    Potential. They can be constant (see `ConstantParameter`), or they can be
    functions of time.

    Parameters
    ----------
    unit : Unit
        The output unit of the parameter.
    """

    @abc.abstractmethod
    def __call__(self, t: BatchableRealQScalar, **kwargs: Any) -> FloatQAnyShape:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : `~galax.typing.BatchableRealQScalar`
            The time(s) at which to compute the parameter value.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array[float, "*shape"]
            The parameter value at times ``t``.
        """
        ...


#####################################################################


@final
class ConstantParameter(AbstractParameter):
    """Time-independent potential parameter."""

    # TODO: link this shape to the return shape from __call__
    value: FloatQAnyShape = eqx.field(converter=Quantity.constructor)

    @property
    def unit(self) -> Unit:
        """Return the unit of the parameter."""
        return self.value.unit

    @partial(jax.jit)
    def __call__(self, t: BatchableRealQScalar = t0, **_: Any) -> FloatQAnyShape:
        """Return the constant parameter value.

        Parameters
        ----------
        t : `~galax.typing.BatchableRealQScalar`, optional
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
        return xp.broadcast_to(self.value, t.shape)

    # -------------------------------------------

    def __mul__(self, other: Any) -> "Self":
        return replace(self, value=self.value * other)

    def __rmul__(self, other: Any) -> "Self":
        return replace(self, value=other * self.value)


#####################################################################
# User-defined Parameter
# For passing a function as a parameter.


@final
class UserParameter(AbstractParameter):
    """User-defined Parameter.

    Parameters
    ----------
    func : Callable[[BatchableRealQScalar], Array[float, (*shape,)]]
        The function to use to compute the parameter value.
    unit : Unit, keyword-only
        The output unit of the parameter.
    """

    # TODO: unit handling
    func: ParameterCallable = eqx.field(static=True)
    _: KW_ONLY
    unit: Unit = eqx.field(static=True, converter=u.Unit)

    @partial(jax.jit)
    def __call__(self, t: BatchableRealQScalar, **kwargs: Any) -> FloatQAnyShape:
        # TODO: think more about unit handling
        return Quantity.constructor(self.func(t, **kwargs), self.unit)
