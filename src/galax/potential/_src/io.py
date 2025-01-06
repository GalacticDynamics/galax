"""Input/output/conversion of potential objects.

This module contains the machinery for I/O and conversion of potential objects.
Conversion is useful for e.g. converting a
:class:`galax.potential.AbstractPotential` object to a
:class:`gala.potential.PotentialBase` object.
"""

__all__ = [
    "convert_potential",
    "AbstractInteroperableLibrary",
    "GalaxLibrary",
    "GalaLibrary",
    "GalpyLibrary",
]


from typing import Annotated as Ann, Any, Never, final
from typing_extensions import Doc

from plum import dispatch


class AbstractInteroperableLibrary:
    """Abstract base class for library type on which to dispatch.

    These classes are used as flags to dispatch to the correct conversion
    function. They are not meant to be instantiated!

    Raises
    ------
    ValueError
        If an attempt is made to instantiate this class.

    Examples
    --------
    In this example we show how to use this class as a target for dispatching

    >>> from numbers import Number
    >>> from galax.potential.io import AbstractInteroperableLibrary
    >>> from plum import dispatch

    We define a library class that is a subclass of
    `AbstractInteroperableLibrary`. Here it represents a library that deals with
    numbers.

    >>> class NumbersLibrary(AbstractInteroperableLibrary):
    ...    pass

    We can now define a function that checks if an object is an instance of the
    library.

    >>> @dispatch
    ... def check_library_isinstance(lib: type[NumbersLibrary], obj: Number, /) -> bool:
    ...    return True

    >>> check_library_isinstance(NumbersLibrary, 1)
    True

    This single dispatch is trivial, but this becomes useful when we have
    multiple libraries, and / or different inheritance structures, so that an
    `isinstance` check is insufficient.

    Just as a reminder, this class is not meant to be instantiated:

    >>> try: NumbersLibrary()
    ... except ValueError as e: print(e)
    cannot instantiate AbstractInteroperableLibrary

    """

    def __new__(cls: "type[AbstractInteroperableLibrary]") -> Never:
        msg = "cannot instantiate AbstractInteroperableLibrary"

        raise ValueError(msg)


@final
class GalaxLibrary(AbstractInteroperableLibrary):
    """The :mod:`galax` library."""


@final
class GalaLibrary(AbstractInteroperableLibrary):
    """The :mod:`gala` library."""


@final
class GalpyLibrary(AbstractInteroperableLibrary):
    """The :mod:`galpy` library."""


@dispatch.abstract
def convert_potential(
    to_: Ann[
        AbstractInteroperableLibrary | Any,
        Doc("The type (or library) to which to convert the potential"),
    ],
    from_: Ann[Any, Doc("The potential object to be converted")],
    /,
    **_: Ann[Any, Doc("extra arguments used in the conversion process")],
) -> object:
    msg = f"cannot convert {from_} to {to_}"  # pragma: no cover
    raise NotImplementedError(msg)  # pragma: no cover
