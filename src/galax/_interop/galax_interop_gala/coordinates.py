"""Compatibility with :mod:`gala` coordinates."""

__all__: list[str] = []

import warnings
from typing import Any

import gala.dynamics as galad
from astropy.coordinates import BaseDifferential, BaseRepresentation
from plum import conversion_method, convert

import galax.coordinates as gc

# --------------------------------------------------
# Gala to Galax


@conversion_method(type_from=galad.PhaseSpacePosition, type_to=gc.PhaseSpacePosition)  # type: ignore[arg-type]
def gala_psp_to_galax_psp(
    obj: galad.PhaseSpacePosition, /, t: Any | None = None
) -> gc.PhaseSpacePosition:
    """Gala to galax ``PhaseSpacePosition``."""
    return gc.PhaseSpacePosition(q=obj.pos, p=obj.vel, t=t)


@gc.PhaseSpacePosition.from_.dispatch  # type: ignore[attr-defined,misc]
def from_(
    _: type[gc.PhaseSpacePosition],
    obj: galad.PhaseSpacePosition,
    /,
    t: Any | None = None,
) -> gc.PhaseSpacePosition:
    """Construct a :mod:`galax` PhaseSpacePosition from a :mod:`gala` one."""
    return gala_psp_to_galax_psp(obj, t=t)  # type: ignore[call-arg]


# --------------------------------------------------
# Galax to Gala


@conversion_method(type_from=gc.PhaseSpacePosition, type_to=galad.PhaseSpacePosition)  # type: ignore[arg-type]
def galax_psp_to_gala_psp(obj: gc.PhaseSpacePosition, /) -> galad.PhaseSpacePosition:
    """Galax to gala ``PhaseSpacePosition``.

    .. warning::

        The time is not preserved in the conversion!

    """
    warnings.warn("The time is not preserved in the conversion!", stacklevel=2)

    return galad.PhaseSpacePosition(
        pos=convert(obj.q, BaseRepresentation),
        vel=convert(obj.p, BaseDifferential),
    )
