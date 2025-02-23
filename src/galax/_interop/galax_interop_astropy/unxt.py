"""Compatibility."""

__all__: list[str] = []

from typing import Any

import astropy.units as apyu
from plum import dispatch

import unxt as u
from unxt.quantity import AllowValue


@dispatch  # TODO: type annotate by value
def ustrip(flag: type[AllowValue], unit: Any, x: apyu.Quantity, /) -> Any:  # noqa: ARG001
    """Strip the units from a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> q = u.Quantity(1000, "m")
    >>> u.ustrip(AllowValue, "km", q)
    Array(1., dtype=float64, ...)

    """
    return u.ustrip(unit, x)
