"""Descriptor for a Parameters attributes."""

__all__: tuple[str, ...] = (
    "all_vars",
    "all_parameters",
)

import dataclasses
import functools as ft
import inspect
import operator
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import galax.potential.params


def all_vars(obj: object | type, /) -> dict[str, Any]:
    """Get all vars.

    This is similar to ``vars()``, but works for both instances and classes, and
    includes parent classes.

    """
    cls = obj if inspect.isclass(obj) else type(obj)
    return ft.reduce(operator.__or__, map(vars, cls.mro()[::-1]))


def all_parameters(
    obj: object | type, /
) -> "dict[str, galax.potential.params.ParameterField]":
    """Get all parameter fields of a dataclass, even during construction.

    Parameters
    ----------
    obj : object | type
        A dataclass with ParameterField attributes.
        If an instance is given, its class is used.

    Returns
    -------
    dict[str, Field | ParameterField]
        All fields of the dataclass, including those not yet finalized in the
        class, if it's still under construction, e.g. in ``__init_subclass__``.
    """
    from galax.potential.params import ParameterField

    # Get all the class variables
    all_cls_vars = all_vars(obj)

    # Get all the fields
    return {
        k: cast("ParameterField", v if isinstance(v, ParameterField) else v.default)
        for k, v in all_cls_vars.items()
        if (
            isinstance(v, ParameterField)
            or (
                isinstance(v, dataclasses.Field)
                and isinstance(v.default, ParameterField)
            )
        )
    }
