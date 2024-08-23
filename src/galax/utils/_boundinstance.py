"""Plotting on Potentials.

This module uses code from the `bound-class` package, which is licensed under a
BSD 3-Clause License. The original code can be found at
https:://github.com/nstarman/bound-class. See the license in the LICENSE files.
"""

__all__: list[str] = []

import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, NoReturn, Self, TypeVar, overload

BndTo = TypeVar("BndTo")


# TODO: transition this to https://github.com/nstarman/bound-class
# Currently this is a copy of the code. See the license in the LICENSE file.
class BoundClassRef(weakref.ReferenceType[BndTo]):
    """`weakref.ref` keeping a `BoundClass` connected to its referent.

    Attributes
    ----------
    _bound_ref : `weakref.ReferenceType`[`BoundClass`]
        A weak reference to a |BoundClass| instance. See Notes for details.

    Notes
    -----
    `weakref.ProxyType` autodetects and cleans up deletion of the referent.
    However, unlike a dereferenced `weakref.ReferenceType`, `~weakref.ProxyType`
    fails ``is`` and ``issubclass`` checks. To emulate the auto-cleanup of
    `weakref.ProxyType`, this class adds a custom finalizer to
    `~weakref.ReferenceType` that will clean the referent on the bound instance.
    It is therefore also necessary to store a weak reference (using the base
    `weakref.ref`) to the bound object in the attribute ``_bound_ref``.::

        bound object  --> BoundClassRef  --> referent
            ^------- ref <-----|

    """

    __slots__ = ("_bound_ref",)

    # `__new__` is needed for type hint tracing because the superclass defines
    # `__new__` without `bound`.
    def __new__(
        cls: type[Self],
        ob: BndTo,
        callback: Callable[[weakref.ReferenceType[BndTo]], Any] | None = None,
        *,
        bound: "InstanceDescriptor[BndTo]",  # noqa: ARG003
    ) -> Self:
        ref: Self = super().__new__(cls, ob, callback)
        return ref

    def __init__(
        self,
        ob: BndTo,
        _: Callable[[weakref.ReferenceType[BndTo]], Any] | None = None,
        *,
        bound: "InstanceDescriptor[BndTo]",
    ) -> None:
        # Add a reference to the BoundClass object (it holds ``ob``)
        self._bound_ref = weakref.ref(bound)
        # Create a finalizer that will be called when the referent is deleted,
        # setting ``bound.__selfref__ = None``.
        weakref.finalize(ob, self._finalizer_callback)

    def _finalizer_callback(self) -> None:
        """Set ``bound.__selfref__ = None``."""
        bound = self._bound_ref()
        if bound is not None:  # check that reference to bound is alive.
            # del bound.__self__
            bound._del__self__()  # noqa: SLF001


# TODO: transition this to https://github.com/nstarman/bound-class
# Currently this is a copy of the code. See the license in the LICENSE file.
@dataclass
class InstanceDescriptor(Generic[BndTo]):
    """Instance-level descriptor."""

    def __post_init__(self) -> None:
        self._enclosing_attr: str
        self.__selfref__: BoundClassRef[BndTo] | None
        object.__setattr__(self, "__selfref__", None)

    # ===============================================================
    # Descriptor

    def __set_name__(self, _: Any, name: str) -> None:
        """Store the name of the attribute on the enclosing object."""
        # Store the name of the attribute on the enclosing object
        object.__setattr__(self, "_enclosing_attr", name)

    @overload
    def __get__(
        self: "InstanceDescriptor[BndTo]", enclosing: BndTo, enclosing_cls: None
    ) -> "InstanceDescriptor[BndTo]": ...

    @overload
    def __get__(
        self: "InstanceDescriptor[BndTo]", enclosing: None, enclosing_cls: type[BndTo]
    ) -> NoReturn: ...

    def __get__(
        self: "InstanceDescriptor[BndTo]",
        enclosing: BndTo | None,
        enclosing_cls: type[BndTo] | None,
    ) -> "InstanceDescriptor[BndTo]":
        """Return a copy of this descriptor bound to the enclosing object.

        Parameters
        ----------
        enclosing : BndTo | None
            The object this descriptor is being accessed from. If ``None`` then
            this is being accessed from the class, not an instance.
        enclosing_cls : type[BndTo] | None
            The class this descriptor is being accessed from. If ``None`` then
            this is being accessed from an instance, not the class.

        Returns
        -------
        InstanceDescriptor[BndTo]
            A copy of this descriptor bound to the enclosing object.

        Raises
        ------
        AttributeError
            If ``enclosing`` is ``None``.
        TypeError
            If the descriptor stored on the enclosing object is not of the same
            type as this descriptor.

        """
        # When called without an instance, return self to allow access
        # to descriptor attributes.
        if enclosing is None:
            msg = f"{self._enclosing_attr!r} can only be accessed from " + (
                "its enclosing object."
                if enclosing_cls is None
                else f"a {enclosing_cls.__name__!r} object"
            )
            raise AttributeError(msg)

        # accessed from an enclosing
        dsc = replace(self)

        # We set `__self__` on every call, since if one makes copies of objs,
        # 'dsc' will be copied as well, which will lose the reference.
        dsc._set__self__(enclosing)  # noqa: SLF001
        # TODO: is it faster to check the reference then always make a new one?

        return dsc

    def __set__(self, _: str, __: object) -> NoReturn:
        """Raise an error when trying to set the value."""
        raise AttributeError  # TODO: useful error message

    # ===============================================================

    @property
    def __self__(self) -> BndTo:
        """Return object to which this one is bound.

        Returns
        -------
        object

        Raises
        ------
        `weakref.ReferenceError`
            If no referent was assigned, if it was deleted, or if it was
            de-referenced (e.g. by ``del self.__self__``).

        """
        if hasattr(self, "__selfref__") and isinstance(self.__selfref__, BoundClassRef):
            boundto = self.__selfref__()  # dereference
            if boundto is not None:
                return boundto

            msg = "weakly-referenced object no longer exists"
            raise ReferenceError(msg)

        msg = "no weakly-referenced object"
        raise ReferenceError(msg)

    # TODO: https://github.com/python/mypy/issues/13231
    # @__self__.setter
    # def __self__(self, value: BndTo) -> None:
    def _set__self__(self, value: BndTo) -> None:
        # Set the reference.
        self.__selfref__: BoundClassRef[BndTo] | None
        object.__setattr__(self, "__selfref__", BoundClassRef(value, bound=self))
        # Note: we use ReferenceType over ProxyType b/c the latter fails ``is``
        # and ``issubclass`` checks. ProxyType autodetects and cleans up
        # deletion of the referent, which ReferenceType does not, so we need a
        # custom ReferenceType subclass to emulate this behavior.

    # @__self__.deleter
    # def __self__(self) -> None:
    def _del__self__(self) -> None:
        # Remove reference without deleting the attribute.
        object.__setattr__(self, "__selfref__", None)

    @property
    def enclosing(self) -> BndTo:
        """Return the enclosing instance to which this one is bound.

        Each access of this property dereferences a `weakref.RefernceType`, so
        it is sometimes better to assign this property to a variable and work
        with that.

        """
        return self.__self__
