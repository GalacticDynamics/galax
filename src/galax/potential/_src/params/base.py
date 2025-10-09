"""Parameters on a Potential."""

__all__ = [
    "ParameterCallable",
    "AbstractParameter",
]

import abc
from typing import Any, Protocol, runtime_checkable

import equinox as eqx

from unxt._src.units.api import AstropyUnits

import galax._custom_types as gt


@runtime_checkable
class ParameterCallable(Protocol):
    """Protocol for a Parameter callable."""

    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: AstropyUnits | None = None, **kwargs: Any
    ) -> gt.QuSzAny | gt.SzAny:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : `~galax._custom_types.BBtQuSz0`
            Time(s) at which to compute the parameter value.
        ustrip : Unit | None
            Unit to strip from the parameter value.
            If None, the parameter value is returned with its original unit.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array[float, "*shape"]
            Parameter value(s) at the given time(s).
        """
        ...


# -------------------------------------------


class AbstractParameter(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class for Parameters on a Potential.

    Parameters are time-dependent quantities that are used to define a
    Potential. They can be constant (see `ConstantParameter`), or they can be
    functions of time.

    """

    @abc.abstractmethod
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: AstropyUnits | None = None, **kwargs: Any
    ) -> gt.QuSzAny:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : `~galax._custom_types.BBtQuSz0`
            The time(s) at which to compute the parameter value.
        ustrip : Unit | None
            The unit to strip from the parameter value. If None, the
            parameter value is returned with its original unit.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array[float, "*shape"]
            The parameter value at times ``t``.
        """
        ...
