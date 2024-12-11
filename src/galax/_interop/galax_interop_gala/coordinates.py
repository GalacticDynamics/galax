"""Compatibility with :mod:`gala` coordinates."""

__all__: list[str] = []

import warnings
from typing import Any, cast

import gala.dynamics as gd
from astropy.coordinates import BaseDifferential, BaseRepresentation
from plum import conversion_method, convert

import galax.coordinates as gcx

# --------------------------------------------------
# Gala to Galax


@conversion_method(type_from=gd.PhaseSpacePosition, type_to=gcx.PhaseSpacePosition)  # type: ignore[misc]
def gala_psp_to_galax_psp(
    obj: gd.PhaseSpacePosition, /, t: Any | None = None
) -> gcx.PhaseSpacePosition:
    """Gala to galax ``PhaseSpacePosition``."""
    return gcx.PhaseSpacePosition(q=obj.pos, p=obj.vel, t=t)


@gcx.PhaseSpacePosition.from_.register  # type: ignore[misc]
def from_(
    _: type[gcx.PhaseSpacePosition], obj: gd.PhaseSpacePosition, /, t: Any | None = None
) -> gcx.PhaseSpacePosition:
    """Construct a :mod:`galax` PhaseSpacePosition from a :mod:`gala` one."""
    return cast(gcx.PhaseSpacePosition, gala_psp_to_galax_psp(obj, t=t))


# --------------------------------------------------
# Galax to Gala


@conversion_method(type_from=gcx.PhaseSpacePosition, type_to=gd.PhaseSpacePosition)  # type: ignore[misc]
def galax_psp_to_gala_psp(obj: gcx.PhaseSpacePosition, /) -> gd.PhaseSpacePosition:
    """Galax to gala ``PhaseSpacePosition``.

    .. warning::

        The time is not preserved in the conversion!

    """
    warnings.warn("The time is not preserved in the conversion!", stacklevel=2)

    return gd.PhaseSpacePosition(
        pos=convert(obj.q, BaseRepresentation), vel=convert(obj.p, BaseDifferential)
    )
