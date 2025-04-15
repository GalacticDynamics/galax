"""Descriptor for a Parameters attributes."""

__all__: list[str] = []

import dataclasses
import functools as ft
import inspect
import operator
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .field import ParameterField


def all_vars(obj: object | type, /) -> dict[str, Any]:
    """Get all vars."""
    # Get all the class variables
    cls = obj if inspect.isclass(obj) else type(obj)
    return ft.reduce(operator.__or__, map(vars, cls.mro()[::-1]))


def all_parameters(obj: object | type, /) -> "dict[str, ParameterField]":
    """Get all fields of a dataclass, including those not-yet finalized.

    Parameters
    ----------
    obj : object | type
        A dataclass.

    Returns
    -------
    dict[str, Field | Parameter]
        All fields of the dataclass, including those not yet finalized in the class, if
        it's still under construction, e.g. in ``__init_subclass__``.
    """
    from galax.potential.params import ParameterField

    # Get all the class variables
    all_cls_vars = all_vars(obj)

    # Get all the fields
    return {
        k: cast(ParameterField, v if isinstance(v, ParameterField) else v.default)
        for k, v in all_cls_vars.items()
        if (
            isinstance(v, ParameterField)
            or (
                isinstance(v, dataclasses.Field)
                and isinstance(v.default, ParameterField)
            )
        )
    }
