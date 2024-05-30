"""Parameters on a Potential."""

__all__ = [
    "ParameterCallable",
    "AbstractParameter",
    "ConstantParameter",
    "LinearParameter",
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

    _: KW_ONLY
    unit: Unit = eqx.field(static=True, converter=u.Unit)

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
    _: KW_ONLY
    unit: Unit = eqx.field(static=True, converter=u.Unit)

    def __check_init__(self) -> None:
        """Check the initialization of the class."""
        _ = eqx.error_if(
            self.value,
            self.value.unit.physical_type != self.unit.physical_type,
            "The value must have the same dimensions as the parameter.",
        )

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
        value = self.value * other
        return replace(self, value=value, unit=value.unit)

    def __rmul__(self, other: Any) -> "Self":
        value = other * self.value
        return replace(self, value=value, unit=value.unit)


#####################################################################
# Linear time dependence Parameter


class LinearParameter(AbstractParameter):
    """Linear time dependence Parameter.

    This is in point-slope form, where the parameter is given by

    .. math::

        p(t) = m * (t - ti) + p(ti)

    Parameters
    ----------
    slope : Quantity[float, (), "[parameter]/[time]"]
        The slope of the linear parameter.
    point_time : Array[float, (), "time"]
        The time at which the parameter is equal to the intercept.
    point_value : Quantity[float, (), "[parameter]"]
        The value of the parameter at the ``point_time``.

    Examples
    --------
    >>> from galax.potential import LinearParameter
    >>> from unxt import Quantity

    >>> lp = LinearParameter(slope=Quantity(-1, "Msun/yr"),
    ...                      point_time=Quantity(0, "Myr"),
    ...                      point_value=Quantity(1e9, "Msun"),
    ...                      unit="Msun")

    >>> lp(Quantity(0, "Gyr"))
    Quantity['mass'](Array(1.e+09, dtype=float64), unit='solMass')
    >>> lp(Quantity(1, "Gyr"))
    Quantity['mass'](Array(0., dtype=float64), unit='solMass')
    """

    slope: FloatQAnyShape = eqx.field(
        converter=lambda x: Quantity.constructor(x, dtype=float)
    )
    point_time: BatchableRealQScalar = eqx.field(
        converter=lambda x: Quantity["time"].constructor(x, dtype=float)
    )
    point_value: FloatQAnyShape = eqx.field(
        converter=lambda x: Quantity.constructor(x, dtype=float)
    )
    _: KW_ONLY
    unit: Unit = eqx.field(static=True, converter=u.Unit)

    def __check_init__(self) -> None:
        """Check the initialization of the class."""
        # TODO: check point_value and slope * point_time have the same dimensions

    def __call__(self, t: BatchableRealQScalar, **_: Any) -> FloatQAnyShape:
        """Return the parameter value.

        .. math::

            p(t) = m * (t - ti) + p(ti)

        Parameters
        ----------
        t : Quantity[float | int, (*batch,), "time"], optional

        Returns
        -------
        Array[float, "*shape"]
            The constant parameter value.

        Examples
        --------
        >>> from galax.potential import LinearParameter
        >>> from unxt import Quantity

        >>> lp = LinearParameter(slope=Quantity(-1, "Msun/yr"),
        ...                      point_time=Quantity(0, "Myr"),
        ...                      point_value=Quantity(1e9, "Msun"),
        ...                      unit="Msun")

        >>> lp(Quantity(0, "Gyr"))
        Quantity['mass'](Array(1.e+09, dtype=float64), unit='solMass')
        >>> lp(Quantity(1, "Gyr"))
        Quantity['mass'](Array(0., dtype=float64), unit='solMass')
        """
        return Quantity.constructor(
            self.slope * (t - self.point_time) + self.point_value, self.unit
        )


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
