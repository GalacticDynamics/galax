"""galax: Galactic Dynamix in Jax."""

__all__ = ["field", "dataclass_with_converter", "ModuleMeta"]

import dataclasses
import functools as ft
import inspect
from collections.abc import Callable, Iterator, Mapping
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
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
import jax.numpy as jnp
from equinox._module import _has_dataclass_init, _ModuleMeta
from jax.dtypes import canonicalize_dtype
from jaxtyping import Array, Float
from typing_extensions import ParamSpec, Unpack

import quaxed.array_api as xp
from unxt import Quantity

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


##############################################################################
# Field


class Sentinel(Enum):
    MISSING = auto()


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
    :class:`~dataclasses.Field`
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
        msg = "Cannot use metadata with `static` already set."
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
class _DataclassInstance(Protocol):
    """Protocol for dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


def _add_converter_init_to_class(cls: type[T], /) -> type[T]:
    """Make a new `__init__` method that applies the converters."""
    original_init = cls.__init__
    sig = inspect.signature(original_init)

    @ft.wraps(original_init)
    def init(self: _DataclassInstance, *args: Any, **kwargs: Any) -> None:
        __tracebackhide__ = True  # pylint: disable=unused-variable

        # Apply any converter to its argument.
        ba = sig.bind(self, *args, **kwargs)
        for f in dataclasses.fields(self):
            if f.name in ba.arguments and "converter" in f.metadata:
                ba.arguments[f.name] = f.metadata["converter"](ba.arguments[f.name])
        # Call the original `__init__`.
        init.__wrapped__(*ba.args, **ba.kwargs)

    cls.__init__ = init  # type: ignore[assignment,method-assign]

    return cls


@dataclass_transform(
    eq_default=True,
    order_default=False,
    kw_only_default=False,
    frozen_default=True,
    field_specifiers=(dataclasses.Field, field),
)
def dataclass_with_converter(
    *,
    init: bool = True,
    repr: bool = True,  # noqa: A002
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = True,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Add dunder methods based on the fields defined in the class.

    See :func:`dataclasses.dataclass` for more information.
    """
    normal_dataclass = dataclasses.dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
        match_args=match_args,
        kw_only=kw_only,
        slots=slots,
        weakref_slot=weakref_slot,
    )

    def wrapper(cls: type[T], /) -> type[T]:
        cls = normal_dataclass(cls)
        return _add_converter_init_to_class(cls)

    return wrapper


##############################################################################
# ModuleMeta


# TODO: upstream this to Equinox
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

    def __new__(  # noqa: D102  # pylint: disable=signature-differs
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: Mapping[str, Any],
        /,
        *,
        strict: bool = False,
        **kwargs: Any,
    ) -> type:
        # [Step 1] Create the class using `_ModuleMeta`.
        cls: type = super().__new__(
            mcs, name, bases, namespace, strict=strict, **kwargs
        )

        # [Step 2] Convert the defaults.
        for k, v in namespace.items():
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
            cls = _add_converter_init_to_class(cls)

        return cls


##############################################################################
# Converters


@ft.singledispatch
def converter_float_array(x: Any, /) -> Float[Array, "*shape"]:
    """Convert to a batched vector."""
    x = xp.asarray(x, dtype=None)
    dtype = jnp.promote_types(x.dtype, canonicalize_dtype(float))
    return xp.asarray(x, dtype=dtype)


@converter_float_array.register
def _converter_float_quantity(x: Quantity, /) -> Float[Array, "*shape"]:
    """Convert to a batched vector."""
    return converter_float_array(x.to_units_value(u.dimensionless_unscaled))


##############################################################################
# Utils


def dataclass_items(obj: "DataclassInstance") -> Iterator[tuple[str, Any]]:
    """Return the field names and values of a dataclass instance."""
    yield from ((f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj))
