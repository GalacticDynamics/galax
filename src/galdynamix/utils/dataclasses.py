"""galdynamix: Galactic Dynamix in Jax."""


__all__ = ["field"]

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from typing import (
    Any,
    ClassVar,
    Generic,
    NotRequired,
    Protocol,
    TypedDict,
    TypeVar,
    dataclass_transform,
    runtime_checkable,
)

import astropy.units as u
from typing_extensions import ParamSpec, Unpack

T = TypeVar("T")
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


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


def _identity(x: T) -> T:
    return x


def _dataclass_with_converter(
    **kwargs_for_dataclass: Any,
) -> Callable[[type[T]], type[T]]:
    """Dataclass decorator that allows for custom converters.

    Example:
    -------
    >>> @dataclass_with_converter()
    ... class A:
    ... x: int = field(converter=int)

    >>> a = A("1.0")
    >>> print(a)
    A(x=1)
    """

    @dataclass_transform()
    def dataclass_with_converter(cls: type[T]) -> type[T]:
        cls = dataclasses.dataclass(**kwargs_for_dataclass)(cls)

        sig = inspect.signature(cls.__init__)

        def init_with_converter(
            self: DataclassInstance, *args: Any, **kwargs: Any
        ) -> None:
            ba = sig.bind(self, *args, **kwargs)
            ba.apply_defaults()
            for f in dataclasses.fields(self):
                converter = f.metadata.get("converter", _identity)
                ba.arguments[f.name] = converter(ba.arguments[f.name])
            cls.__init__.__wrapped__(*ba.args, **ba.kwargs)  # type: ignore[attr-defined]

        cls.__init__ = functools.wraps(cls.__init__)(init_with_converter)  # type: ignore[assignment]

        return cls

    return dataclass_with_converter


def converter_float_array(
    x: Any, /
) -> Float[Array, "*shape"] | Integer[Array, "*shape"]:
    """Convert to a batched vector."""
    x = xp.array(x, dtype=None)
    return xp.array(x, dtype=xp.promote_types(x.dtype, float))
