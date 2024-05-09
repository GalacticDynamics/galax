"""Compatibility.

TODO: make all the `gala` compat be in a linked package.

"""

__all__: list[str] = []

from typing import Any, cast

try:  # TODO: less hacky way of supporting optional dependencies
    import pytest
except ImportError:  # pragma: no cover
    pass
else:
    _ = pytest.importorskip("gala")

import astropy.coordinates as apyc
import gala.dynamics as gd
from plum import conversion_method, convert

import galax.coordinates as gcx

# --------------------------------------------------
# Astropy to Galax


@gcx.PhaseSpacePosition.constructor._f.register  # noqa: SLF001
def constructor(
    _: type[gcx.PhaseSpacePosition],
    vec: apyc.BaseRepresentation,
    /,
    t: Any | None = None,
) -> gcx.PhaseSpacePosition:
    """Construct a :mod:`galax` PhaseSpacePosition from a :mod:`astropy` coordinate.

    Parameters
    ----------
    vec : :class:`~astropy.coordinates.BaseRepresentation`
        The astropy representation.
    t : Any, optional
        The time.

    Examples
    --------
    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> import galax.coordinates as gc

    >>> vec = coord.SphericalRepresentation(
    ...     lon=u.Quantity(10, u.deg), lat=u.Quantity(34, u.deg),
    ...     distance=u.Quantity(3, u.kpc),
    ...     differentials=coord.SphericalCosLatDifferential(
    ...         d_lon_coslat=1*u.deg/u.Myr, d_lat=1*u.deg/u.Myr,
    ...         d_distance=1*u.kpc/u.Myr) )
    >>> vec
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, kpc)
        (10., 34., 3.)
     (has differentials w.r.t.: 's')>

    >>> gc.PhaseSpacePosition.constructor(vec, t=u.Quantity(0, "Myr"))
    PhaseSpacePosition(
        q=LonLatSphericalVector( lon=..., lat=..., distance=... ),
        p=LonCosLatSphericalDifferential( d_lon_coslat=..., d_lat=..., d_distance=... ),
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Myr"))
    )

    """
    if "s" not in vec.differentials:
        msg = "The astropy representation does not have a velocity differential."
        raise ValueError(msg)

    return gcx.PhaseSpacePosition(
        q=vec.without_differentials(), p=vec.differentials["s"], t=t
    )


@gcx.PhaseSpacePosition.constructor._f.register  # noqa: SLF001
def constructor(
    _: type[gcx.PhaseSpacePosition],
    vec: apyc.BaseRepresentation,
    dif: apyc.BaseDifferential,
    /,
    t: Any | None = None,
) -> gcx.PhaseSpacePosition:
    """Construct a :mod:`galax` PhaseSpacePosition from a :mod:`astropy` coordinate.

    Parameters
    ----------
    vec : :class:`~astropy.coordinates.BaseRepresentation`
        The astropy representation.
    dif : :class:`~astropy.coordinates.BaseDifferential`
        The astropy differential.
    t : Any, optional
        The time.

    Examples
    --------
    >>> import astropy.coordinates as coord
    >>> import astropy.units as u
    >>> import galax.coordinates as gc

    >>> vec = coord.SphericalRepresentation(
    ...     lon=u.Quantity(10, u.deg), lat=u.Quantity(34, u.deg),
    ...     distance=u.Quantity(3, u.kpc))
    >>> dif = coord.SphericalCosLatDifferential(
    ...         d_lon_coslat=1*u.deg/u.Myr, d_lat=1*u.deg/u.Myr,
    ...         d_distance=1*u.kpc/u.Myr)
    >>> vec
    <SphericalRepresentation (lon, lat, distance) in (deg, deg, kpc)
        (10., 34., 3.)>

    >>> gc.PhaseSpacePosition.constructor(vec, dif, t=u.Quantity(0, "Myr"))
    PhaseSpacePosition(
        q=LonLatSphericalVector( lon=..., lat=..., distance=... ),
        p=LonCosLatSphericalDifferential( d_lon_coslat=..., d_lat=..., d_distance=... ),
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Myr"))
    )

    """
    return gcx.PhaseSpacePosition(q=vec.without_differentials(), p=dif, t=t)


# --------------------------------------------------
# Gala to Galax


@conversion_method(type_from=gd.PhaseSpacePosition, type_to=gcx.PhaseSpacePosition)  # type: ignore[misc]
def gala_psp_to_galax_psp(
    obj: gd.PhaseSpacePosition, /, t: Any | None = None
) -> gcx.PhaseSpacePosition:
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
        q=Cartesian3DVector( ... ),
        p=CartesianDifferential3D( ... ),
        t=None
    )
    """
    return gcx.PhaseSpacePosition(q=obj.pos, p=obj.vel, t=t)


@gcx.PhaseSpacePosition.constructor._f.register  # type: ignore[misc]  # noqa: SLF001
def constructor(
    _: type[gcx.PhaseSpacePosition], obj: gd.PhaseSpacePosition, /, t: Any | None = None
) -> gcx.PhaseSpacePosition:
    """Construct a :mod:`galax` PhaseSpacePosition from a :mod:`gala` one.

    Examples
    --------
    With the following imports:

    >>> import gala.dynamics as gd
    >>> import galax.coordinates as gcx
    >>> import astropy.units as u

    We can create a :class:`gala.dynamics.PhaseSpacePosition` and construct a
    :class:`galax.coordinates.PhaseSpacePosition` from it.

    >>> gala_w = gd.PhaseSpacePosition(pos=[1, 2, 3] * u.kpc,
    ...                                vel=[4, 5, 6] * u.km / u.s)
    >>> galax_w = gcx.PhaseSpacePosition.constructor(gala_w)
    >>> galax_w
    PhaseSpacePosition(
        q=Cartesian3DVector( ... ),
        p=CartesianDifferential3D( ... ),
        t=None
    )

    We can also pass a time:

    >>> galax_w = gcx.PhaseSpacePosition.constructor(gala_w, t=2 * u.Myr)

    """
    return cast(gcx.PhaseSpacePosition, gala_psp_to_galax_psp(obj, t=t))


# --------------------------------------------------
# Galax to Gala


@conversion_method(type_from=gcx.PhaseSpacePosition, type_to=gd.PhaseSpacePosition)  # type: ignore[misc]
def galax_psp_to_gala_psp(obj: gcx.PhaseSpacePosition, /) -> gd.PhaseSpacePosition:
    """Galax to gala ``PhaseSpacePosition``.

    .. warning::

        The time is not preserved in the conversion!

    Examples
    --------
    With the following imports:

    >>> import gala.dynamics as gd
    >>> import galax.coordinates as gcx
    >>> import astropy.units as u
    >>> from plum import convert

    We can create a :class:`galax.coordinates.PhaseSpacePosition` and convert it
    to a :class:`gala.dynamics.PhaseSpacePosition`.

    >>> galax_w = gcx.PhaseSpacePosition(
    ...     q=[1, 2, 3] * u.kpc, p=[4, 5, 6] * u.km / u.s, t=2 * u.Myr
    ... )

    >>> gala_w = convert(galax_w, gd.PhaseSpacePosition)
    >>> gala_w
    <PhaseSpacePosition cartesian, dim=3, shape=()>

    Note that the time is not preserved in the conversion!

    """
    from astropy.coordinates import BaseDifferential, BaseRepresentation

    return gd.PhaseSpacePosition(
        pos=convert(obj.q, BaseRepresentation), vel=convert(obj.p, BaseDifferential)
    )
