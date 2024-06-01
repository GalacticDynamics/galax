"""Compatibility.

TODO: make all the `astropy` compat be in a linked package.

"""

__all__: list[str] = []

from typing import Any

import astropy.coordinates as apyc

from .core import PhaseSpacePosition

# --------------------------------------------------
# Astropy to Galax


@PhaseSpacePosition.constructor._f.register  # noqa: SLF001
def constructor(
    _: type[PhaseSpacePosition],
    vec: apyc.BaseRepresentation,
    /,
    t: Any | None = None,
) -> PhaseSpacePosition:
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
        q=LonLatSphericalPosition( lon=..., lat=..., distance=... ),
        p=LonCosLatSphericalVelocity( d_lon_coslat=..., d_lat=..., d_distance=... ),
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Myr"))
    )

    """
    if "s" not in vec.differentials:
        msg = "The astropy representation does not have a velocity differential."
        raise ValueError(msg)

    return PhaseSpacePosition(
        q=vec.without_differentials(), p=vec.differentials["s"], t=t
    )


@PhaseSpacePosition.constructor._f.register  # noqa: SLF001
def constructor(
    _: type[PhaseSpacePosition],
    vec: apyc.BaseRepresentation,
    dif: apyc.BaseDifferential,
    /,
    t: Any | None = None,
) -> PhaseSpacePosition:
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
        q=LonLatSphericalPosition( lon=..., lat=..., distance=... ),
        p=LonCosLatSphericalVelocity( d_lon_coslat=..., d_lat=..., d_distance=... ),
        t=Quantity[PhysicalType('time')](value=f64[], unit=Unit("Myr"))
    )

    """
    return PhaseSpacePosition(q=vec.without_differentials(), p=dif, t=t)
