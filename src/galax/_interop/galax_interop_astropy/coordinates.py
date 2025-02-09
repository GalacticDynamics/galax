"""Compatibility."""

__all__: list[str] = []

from typing import Any

import astropy.coordinates as apyc

import galax.coordinates as gc

# =============================================================================
# PhaseSpacePosition


@gc.AbstractPhaseSpaceObject.from_.dispatch
def from_(
    _: type[gc.PhaseSpacePosition], vec: apyc.BaseRepresentation, /
) -> gc.PhaseSpacePosition:
    """Construct a :mod:`galax` PhaseSpacePosition from a :mod:`astropy` coordinate.

    Parameters
    ----------
    vec : :class:`~astropy.coordinates.BaseRepresentation`
        The astropy representation.

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

    >>> gc.PhaseSpacePosition.from_(vec)
    PhaseSpacePosition(
        q=LonLatSphericalPos( lon=..., lat=..., distance=... ),
        p=LonCosLatSphericalVel( lon_coslat=..., lat=..., distance=... ),
        frame=SimulationFrame()
    )

    """
    if "s" not in vec.differentials:
        msg = "The astropy representation does not have a velocity differential."
        raise ValueError(msg)

    return gc.PhaseSpacePosition(
        q=vec.without_differentials(), p=vec.differentials["s"]
    )


@gc.AbstractPhaseSpaceObject.from_.dispatch
def from_(
    _: type[gc.PhaseSpaceCoordinate], vec: apyc.BaseRepresentation, t: Any, /
) -> gc.PhaseSpaceCoordinate:
    """Construct a :mod:`galax` PhaseSpaceCoordinate from a :mod:`astropy` coordinate.

    Parameters
    ----------
    vec : :class:`~astropy.coordinates.BaseRepresentation`
        The astropy representation.
    t : :class:`~unxt.AbstractQuantity`
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

    >>> gc.PhaseSpaceCoordinate.from_(vec, u.Quantity(0, "Myr"))
    PhaseSpaceCoordinate(
        q=LonLatSphericalPos( lon=..., lat=..., distance=... ),
        p=LonCosLatSphericalVel( lon_coslat=..., lat=..., distance=... ),
        t=Quantity['time'](Array(0., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )

    """
    if "s" not in vec.differentials:
        msg = "The astropy representation does not have a velocity differential."
        raise ValueError(msg)

    return gc.PhaseSpaceCoordinate(
        q=vec.without_differentials(), p=vec.differentials["s"], t=t
    )


@gc.PhaseSpacePosition.from_.dispatch
def from_(
    _: type[gc.PhaseSpacePosition],
    vec: apyc.BaseRepresentation,
    dif: apyc.BaseDifferential,
    /,
) -> gc.PhaseSpacePosition:
    """Construct a :mod:`galax` PhaseSpacePosition from a :mod:`astropy` coordinate.

    Parameters
    ----------
    vec : :class:`~astropy.coordinates.BaseRepresentation`
        The astropy representation.
    dif : :class:`~astropy.coordinates.BaseDifferential`
        The astropy differential.

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

    >>> gc.PhaseSpacePosition.from_(vec, dif)
    PhaseSpacePosition(
        q=LonLatSphericalPos( lon=..., lat=..., distance=... ),
        p=LonCosLatSphericalVel( lon_coslat=..., lat=..., distance=... ),
        frame=SimulationFrame()
    )

    """
    return gc.PhaseSpacePosition(q=vec.without_differentials(), p=dif)


@gc.PhaseSpaceCoordinate.from_.dispatch
def from_(
    _: type[gc.PhaseSpaceCoordinate],
    vec: apyc.BaseRepresentation,
    dif: apyc.BaseDifferential,
    t: Any,
    /,
) -> gc.PhaseSpaceCoordinate:
    """Construct a :mod:`galax` PhaseSpaceCoordinate from a :mod:`astropy` coordinate.

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

    >>> gc.PhaseSpaceCoordinate.from_(vec, dif, u.Quantity(0, "Myr"))
    PhaseSpaceCoordinate(
        q=LonLatSphericalPos( lon=..., lat=..., distance=... ),
        p=LonCosLatSphericalVel( lon_coslat=..., lat=..., distance=... ),
        t=Quantity['time'](Array(0., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )

    """
    return gc.PhaseSpaceCoordinate(q=vec.without_differentials(), p=dif, t=t)
