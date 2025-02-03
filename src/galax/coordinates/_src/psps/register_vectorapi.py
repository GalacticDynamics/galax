"""Register PSPs with `coordinax`."""

__all__: list[str] = []

from typing import Any, cast

import coordinax as cx

from .base import AbstractPhaseSpacePosition


@cx.frames.AbstractCoordinate.vconvert.dispatch  # type: ignore[misc]
def vconvert(
    self: AbstractPhaseSpacePosition,
    position_cls: type[cx.vecs.AbstractPos],
    velocity_cls: type[cx.vecs.AbstractVel] | None = None,
    /,
    **kwargs: Any,
) -> AbstractPhaseSpacePosition:
    """Return with the components transformed.

    Parameters
    ----------
    position_cls : type[:class:`~vector.AbstractPos`]
        The target position class.
    velocity_cls : type[:class:`~vector.AbstractVel`], optional
        The target differential class. If `None` (default), the differential
        class of the target position class is used.
    **kwargs
        Additional keyword arguments are passed through to `coordinax.vconvert`.

    Returns
    -------
    w : :class:`~galax.coordinates.AbstractOnePhaseSpacePosition`
        The phase-space position with the components transformed.

    Examples
    --------
    With the following imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to a 6-vector:

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

    We can also convert it to a different representation:

    >>> psp.vconvert(cx.vecs.CylindricalPos)
    PhaseSpacePosition( q=CylindricalPos(...),
                        p=CylindricalVel(...),
                        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
                        frame=SimulationFrame() )

    We can also convert it to a different representation with a different
    differential class:

    >>> psp.vconvert(cx.vecs.LonLatSphericalPos, cx.vecs.LonCosLatSphericalVel)
    PhaseSpacePosition( q=LonLatSphericalPos(...),
                        p=LonCosLatSphericalVel(...),
                        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
                        frame=SimulationFrame() )
    """
    return cast(
        AbstractPhaseSpacePosition,
        cx.vconvert({"q": position_cls, "p": velocity_cls}, self, **kwargs),
    )
