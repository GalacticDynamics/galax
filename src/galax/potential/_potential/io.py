"""Input/output/conversion of potential objects.

This module contains the machinery for I/O and conversion of potential objects.
Conversion is useful for e.g. converting a
:class:`galax.potential.AbstractPotential` object to a
:class:`gala.potential.PotentialBase` object.
"""

__all__ = [
    "convert_potential",
    "AbstractInteroperableLibrary",
    "InteroperateGalaxLibrary",
    "InteroperableGalaLibrary",
]


from typing import Any, Never, final

from plum import dispatch


class AbstractInteroperableLibrary:
    """Abstract base class for library type on which to dispatch."""

    def __new__(cls: type["AbstractInteroperableLibrary"]) -> Never:
        msg = "cannot instantiate AbstractInteroperableLibrary"

        raise ValueError(msg)


@final
class InteroperateGalaxLibrary(AbstractInteroperableLibrary):
    """The :mod:`galax` library."""


@final
class InteroperableGalaLibrary(AbstractInteroperableLibrary):
    """The :mod:`gala` library."""


@dispatch.abstract  # type: ignore[misc]
def convert_potential(
    from_: Any,
    to_: AbstractInteroperableLibrary | Any,
    /,
    **kwargs: Any,  # noqa: ARG001
) -> object:
    msg = f"cannot convert {from_} to {to_}"
    raise NotImplementedError(msg)