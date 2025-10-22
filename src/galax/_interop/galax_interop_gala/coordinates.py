"""Compatibility with :mod:`gala` coordinates."""

__all__: list[str] = []

import warnings
from typing import Any

import gala.dynamics as galad
import plum
from astropy.coordinates import BaseDifferential, BaseRepresentation

import galax.coordinates as gc

# --------------------------------------------------
# Gala to Galax


@plum.conversion_method(  # type: ignore[arg-type]
    type_from=galad.PhaseSpacePosition, type_to=gc.PhaseSpacePosition
)
def gala_psp_to_galax_psp(obj: galad.PhaseSpacePosition, /) -> gc.PhaseSpacePosition:
    """Gala to galax ``PhaseSpacePosition``.

    Examples
    --------
    With the following imports:

    >>> import gala.dynamics as gd
    >>> import galax.coordinates as gcx
    >>> import astropy.units as u
    >>> from plum import convert

    We can create a :class:`gala.dynamics.PhaseSpacePosition` and convert it to
    a :class:`galax.coordinates.PhaseSpacePosition`

    >>> gala_w = gd.PhaseSpacePosition(pos=[1, 2, 3] * u.kpc,
    ...                                vel=[4, 5, 6] * u.km / u.s)
    >>> gala_w
    <PhaseSpacePosition cartesian, dim=3, shape=()>

    >>> galax_w = convert(gala_w, gcx.PhaseSpacePosition)
    >>> galax_w
    PhaseSpacePosition(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        frame=SimulationFrame()
    )
    """
    return gc.PhaseSpacePosition(q=obj.pos, p=obj.vel, frame=gc.frames.simulation_frame)


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


@plum.conversion_method(
    type_from=gc.PhaseSpacePosition, type_to=galad.PhaseSpacePosition
)
def galax_psp_to_gala_psp(obj: gc.PhaseSpacePosition, /) -> galad.PhaseSpacePosition:
    """Galax to gala ``PhaseSpacePosition``.

    .. warning::

        The frame is not preserved in the conversion!

    Examples
    --------
    With the following imports:

    >>> from warnings import catch_warnings, filterwarnings
    >>> import gala.dynamics as gd
    >>> import galax.coordinates as gcx
    >>> import astropy.units as u
    >>> from plum import convert

    We can create a :class:`galax.coordinates.PhaseSpacePosition` and convert it
    to a :class:`gala.dynamics.PhaseSpacePosition`.

    >>> galax_w = gcx.PhaseSpacePosition(
    ...     q=[1, 2, 3] * u.kpc, p=[4, 5, 6] * u.km / u.s
    ... )

    >>> with catch_warnings(action="ignore"):
    ...     gala_w = convert(galax_w, gd.PhaseSpacePosition)
    >>> gala_w
    <PhaseSpacePosition cartesian, dim=3, shape=()>

    """
    warnings.warn("The frame is not preserved in the conversion!", stacklevel=2)

    return galad.PhaseSpacePosition(
        pos=plum.convert(obj.q, BaseRepresentation),
        vel=plum.convert(obj.p, BaseDifferential),
    )


@plum.conversion_method(
    type_from=gc.PhaseSpaceCoordinate, type_to=galad.PhaseSpacePosition
)
def galax_psp_to_gala_psp(obj: gc.PhaseSpaceCoordinate, /) -> galad.PhaseSpacePosition:
    """Galax to gala ``PhaseSpacePosition``.

    .. warning::

        The frame and time are not preserved in the conversion!

    Examples
    --------
    With the following imports:

    >>> from warnings import catch_warnings, filterwarnings
    >>> import gala.dynamics as gd
    >>> import galax.coordinates as gcx
    >>> import astropy.units as u
    >>> from plum import convert

    We can create a :class:`galax.coordinates.PhaseSpaceCoordinate` and convert
    it to a :class:`gala.dynamics.PhaseSpacePosition`.

    >>> galax_w = gcx.PhaseSpaceCoordinate(
    ...     q=[1, 2, 3] * u.kpc, p=[4, 5, 6] * u.km / u.s, t=2 * u.Myr
    ... )

    >>> with catch_warnings(action="ignore"):
    ...     gala_w = convert(galax_w, gd.PhaseSpacePosition)
    >>> gala_w
    <PhaseSpacePosition cartesian, dim=3, shape=()>

    Note that the time is not preserved in the conversion!

    """
    warnings.warn(
        "The time and frame are not preserved in the conversion!", stacklevel=2
    )

    return galad.PhaseSpacePosition(
        pos=plum.convert(obj.q, BaseRepresentation),
        vel=plum.convert(obj.p, BaseDifferential),
    )
