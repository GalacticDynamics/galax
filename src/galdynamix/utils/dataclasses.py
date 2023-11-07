"""galdynamix: Galactic Dynamix in Jax"""

from __future__ import annotations

__all__ = ["field"]

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Generic, NotRequired, TypedDict, TypeVar

import astropy.units as u
from typing_extensions import ParamSpec, Unpack

P = ParamSpec("P")
R = TypeVar("R")


# TODO: how to express default_factory is mutually exclusive with default?
class _DataclassFieldKwargsDefault(TypedDict, Generic[R]):
    default: NotRequired[R]
    init: NotRequired[bool]
    repr: NotRequired[bool]
    hash: NotRequired[bool | None]
    compare: NotRequired[bool]
    metadata: NotRequired[Mapping[Any, Any] | None]
    kw_only: NotRequired[bool]


def field(
    *,
    # Equinox stuff
    converter: Callable[[Any], R] | None = None,
    static: bool = False,
    # Units stuff
    physical_type: str | u.PhysicalType | None = None,
    equivalencies: u.Equivalency | tuple[u.Equivalency, ...] | None = None,
    # Dataclass stuff
    **kwargs: Unpack[_DataclassFieldKwargsDefault[R]],
) -> R:
    """See Equinox for `field` documentation."""
    metadata = dict(kwargs.pop("metadata", {}) or {})  # safety copy

    metadata["physical_type"] = (
        u.get_physical_type(physical_type)
        if isinstance(physical_type, str)
        else physical_type
    )
    metadata["equivalencies"] = equivalencies

    # --------------------------------
    # Equinox stuff

    if "converter" in metadata:
        msg = "Cannot use metadata with `static` already set."
        raise ValueError(msg)
    if "static" in metadata:
        "Cannot use metadata with `static` already set."
        raise ValueError(msg)

    if converter is not None:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True

    # --------------------------------

    kwargs["metadata"] = metadata  # done only for typing purposes

    out: R = dataclasses.field(**kwargs)
    return out
