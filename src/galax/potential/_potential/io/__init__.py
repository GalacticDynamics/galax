"""Input/output/conversion of potential objects.

This module contains the machinery for I/O and conversion of potential objects.
Conversion is useful for e.g. converting a
:class:`galax.potential.AbstractPotential` object to a
:class:`gala.potential.PotentialBase` object.
"""

__all__ = ["gala_to_galax"]

import sys

from galax.utils._optional_deps import HAS_GALA


def __dir__() -> list[str]:
    """Return the list of names in the module."""
    return sorted(__all__)


def __getattr__(name: str) -> object:
    """Get the attribute."""
    match name:
        case "gala_to_galax":
            if HAS_GALA:
                from ._gala import gala_to_galax as out
            else:
                from ._gala_noop import gala_to_galax as out

        case _:
            msg = f"module {__name__!r} has no attribute {name!r}"
            raise AttributeError(msg)

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
