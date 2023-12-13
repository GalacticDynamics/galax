"""galax: Galactic Dynamix in Jax."""


__all__ = ["field"]

import dataclasses
import functools as ft
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
    runtime_checkable,
)

import astropy.units as u
import jax.numpy as xp
from equinox._module import _has_dataclass_init, _ModuleMeta
from jaxtyping import Array, Float, Integer
from typing_extensions import ParamSpec, Unpack

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

##############################################################################
# ModuleMeta


@runtime_checkable
class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


class ModuleMeta(_ModuleMeta):  # type: ignore[misc]
    """Equinox-compatible module metaclass.

    This metaclass extends Equinox's :class:`equinox._module._ModuleMeta` to
    support the following features:

    - Application of ``converter`` to default values on fields.
    - Application of ``converter`` to values passed to ``__init__``.

    Examples
    --------
    >>> import equinox as eqx
    >>> class Class(eqx.Module, metaclass=ModuleMeta):
    ...     a: int = eqx.field(default=1.0, converter=int)
    ...     def __post_init__(self): pass

    >>> Class.a
    1

    >>> Class(a=2.0)
    Class(a=2)
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        dict_: Mapping[str, Any],
        /,
        *,
        strict: bool = False,
        abstract: bool = False,
        **kwargs: Any,
    ) -> type:
        # [Step 1] Create the class using `_ModuleMeta`.
        cls: type = super().__new__(
            mcs, name, bases, dict_, strict=strict, abstract=abstract, **kwargs
        )

        # [Step 2] Convert the defaults.
        for k, v in dict_.items():
            if not isinstance(v, dataclasses.Field):
                continue
            # Apply the converter to the default value.
            if "converter" in v.metadata and not isinstance(
                v.default,
                dataclasses._MISSING_TYPE,  # noqa: SLF001
            ):
                setattr(cls, k, v.metadata["converter"](v.default))

        # [Step 3] Ensure conversion happens before `__init__`.
        if _has_dataclass_init[cls]:
            original_init = cls.__init__  # type: ignore[misc]
            sig = inspect.signature(original_init)

            @ft.wraps(original_init)
            def init(self: _DataclassInstance, *args: Any, **kwargs: Any) -> None:
                __tracebackhide__ = True

                # Apply any converter to its argument.
                ba = sig.bind(self, *args, **kwargs)
                for f in dataclasses.fields(self):
                    if f.name in ba.arguments and "converter" in f.metadata:
                        ba.arguments[f.name] = f.metadata["converter"](
                            ba.arguments[f.name]
                        )
                # Call the original `__init__`.
                init.__wrapped__(*ba.args, **ba.kwargs)  # type: ignore[attr-defined]

            cls.__init__ = init  # type: ignore[misc]

        return cls


##############################################################################
# Field


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
    """Equinox-compatible field with unit information.

    Parameters
    ----------
    converter : callable, optional
        Callable to convert the input value to the desired output type.  See
        Equinox's ``field`` for more information.
    static : bool, optional
        Whether the field is static (i.e., not a leaf in the PyTree).  See
        Equinox's ``field`` for more information.

    dimensions : str or `~astropy.units.physical.PhysicalType`, optional
        The physical type of the field. See Astropy's
        `~astropy.units.physical.PhysicalType` for more information.
    equivalencies : ``Equivalency`` or tuple thereof, optional
        Equivalencies to use for the field. See Astropy's
        `~astropy.units.Equivalency` for more information.

    **kwargs : Any
        Additional keyword arguments to pass to ``dataclasses.field``.

    Returns
    -------
    Field
        The field object.
    """
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


##############################################################################
# Converters


def converter_float_array(
    x: Any, /
) -> Float[Array, "*shape"] | Integer[Array, "*shape"]:
    """Convert to a batched vector."""
    x = xp.array(x, dtype=None)
    return xp.array(x, dtype=xp.promote_types(x.dtype, float))
