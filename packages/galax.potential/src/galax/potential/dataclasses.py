"""galax: Galactic Dynamix in Jax."""

__all__ = ["ModuleMeta"]

import dataclasses
import functools as ft
import inspect
from enum import Enum, auto

from typing import Any

import optype as op
from equinox._module._module import _has_dataclass_init, _ModuleMeta

##############################################################################
# Converter


class Sentinel(Enum):
    MISSING = auto()


##############################################################################
# ModuleMeta


def _add_converter_init_to_class[T](cls: type[T], /) -> type[T]:
    """Make a new `__init__` method that applies the converters."""
    original_init = cls.__init__
    sig = inspect.signature(original_init)

    @ft.wraps(original_init)
    def init(
        self: op.dataclasses.HasDataclassFields, *args: Any, **kwargs: Any
    ) -> None:
        __tracebackhide__ = True  # pylint: disable=unused-variable

        # Apply any converter to its argument.
        ba = sig.bind(self, *args, **kwargs)
        for f in dataclasses.fields(self):
            if f.name in ba.arguments and "converter" in f.metadata:
                ba.arguments[f.name] = f.metadata["converter"](ba.arguments[f.name])
        # Call the original `__init__`.
        init.__wrapped__(*ba.args, **ba.kwargs)

    cls.__init__ = init  # type: ignore[method-assign,assignment]

    return cls


# TODO: upstream this to Equinox
# TODO: Equinox doesn't seem to respect the conversion of the default value anymore.
class ModuleMeta(_ModuleMeta):
    """Equinox-compatible module metaclass.

    This metaclass extends Equinox's :class:`equinox._module._ModuleMeta` to
    support the following features:

    - Application of ``converter`` to default values on fields, unless the
      default has a true ``convert_per_instance`` attribute. Such a default is
      left as is on the class and converted only by ``__init__``, for each
      instance (``galax.potential``'s default constants use this so that no
      class holds a JAX array built at import).
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
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> type:
        # [Step 1] Create the class using `_ModuleMeta`.
        cls: type = super().__new__(mcs, name, bases, namespace, **kwargs)

        # [Step 2] Convert the defaults.
        for k, v in namespace.items():
            if not isinstance(v, dataclasses.Field):
                continue
            # Apply the converter to the default value, unless the default asks
            # to be converted per instance (as `default_constants` does, so the
            # class never holds a JAX array built at import).
            if (
                "converter" in v.metadata
                and not isinstance(
                    v.default,
                    dataclasses._MISSING_TYPE,  # noqa: SLF001
                )
                and not getattr(v.default, "convert_per_instance", False)
            ):
                setattr(cls, k, v.metadata["converter"](v.default))

        # [Step 3] Ensure conversion happens before `__init__`.
        if _has_dataclass_init[cls]:
            cls = _add_converter_init_to_class(cls)

        return cls
