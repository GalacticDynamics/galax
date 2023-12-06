"""galdynamix: Galactic Dynamix in Jax."""


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
    dimensions: str | u.PhysicalType | None = None,
    equivalencies: u.Equivalency | tuple[u.Equivalency, ...] | None = None,
    # Dataclass stuff
    **kwargs: Unpack[_DataclassFieldKwargsDefault[R]],
) -> R:
    """See Equinox for `field` documentation."""
    metadata = dict(kwargs.pop("metadata", {}) or {})  # safety copy

    if dimensions is not None:
        metadata["dimensions"] = (
            u.get_physical_type(dimensions)
            if isinstance(dimensions, str)
            else dimensions
        )
    if equivalencies is not None:
        metadata["equivalencies"] = equivalencies

    # --------------------------------
    # Equinox stuff

    if "converter" in metadata:
        msg = "Cannot use metadata with `converter` already set."
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
