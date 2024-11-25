"""galax: Galactic Dynamics in Jax."""

__all__ = ["AbstractPhaseSpacePosition"]

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from plum import dispatch

import coordinax as cx
from unxt import uconvert, unitsystem

from .base import AbstractBasePhaseSpacePosition

if TYPE_CHECKING:
    from typing import Self


class AbstractPhaseSpacePosition(AbstractBasePhaseSpacePosition):
    r"""Abstract base class of phase-space positions.

    The phase-space position is a point in the 3+3+1-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}\in\mathbb{R}^3`, the conjugate momentum
    :math:`\boldsymbol{p}\in\mathbb{R}^3`, and the time
    :math:`t\in\mathbb{R}^1`.

    Parameters
    ----------
    q : :class:`~vector.AbstractPos3D`
        Positions.
    p : :class:`~vector.AbstractVel3D`
        Conjugate momenta at positions ``q``.
    t : :class:`~unxt.Quantity`
        Time corresponding to the positions and momenta.

    """

    # ==========================================================================
    # Convenience methods

    def to_units(self, units: Any) -> "Self":
        """Return a new object with the components converted to the given units."""
        usys = unitsystem(units)
        return replace(
            self,
            q=self.q.uconvert(usys),
            p=self.p.uconvert(usys),
            t=uconvert(usys["time"], self.t) if self.t is not None else None,
        )


# =============================================================================
# helper functions


# -----------------------------------------------
# Register AbstractPhaseSpacePosition with `coordinax.represent_as`
@dispatch  # type: ignore[misc]
def represent_as(
    psp: AbstractPhaseSpacePosition,
    position_cls: type[cx.AbstractPos],
    velocity_cls: type[cx.AbstractVel] | None = None,
    /,
) -> AbstractPhaseSpacePosition:
    """Return with the components transformed.

    Parameters
    ----------
    psp : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
        The phase-space position.
    position_cls : type[:class:`~vector.AbstractPos`]
        The target position class.
    velocity_cls : type[:class:`~vector.AbstractVel`], optional
        The target differential class. If `None` (default), the differential
        class of the target position class is used.

    Examples
    --------
    With the following imports:

    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> from galax.coordinates import PhaseSpacePosition

    We can create a phase-space position and convert it to a 6-vector:

    >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                          p=Quantity([4, 5, 6], "km/s"),
    ...                          t=Quantity(0, "Gyr"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64)

    We can also convert it to a different representation:

    >>> psp.represent_as(cx.CylindricalPos)
    PhaseSpacePosition( q=CylindricalPos(...),
                        p=CylindricalVel(...),
                        t=Quantity[...](value=f64[], unit=Unit("Gyr")) )

    We can also convert it to a different representation with a different
    differential class:

    >>> psp.represent_as(cx.LonLatSphericalPos, cx.LonCosLatSphericalVel)
    PhaseSpacePosition( q=LonLatSphericalPos(...),
                        p=LonCosLatSphericalVel(...),
                        t=Quantity[...](value=f64[], unit=Unit("Gyr")) )

    """
    diff_cls = position_cls.differential_cls if velocity_cls is None else velocity_cls
    return replace(
        psp,
        q=psp.q.represent_as(position_cls),
        p=psp.p.represent_as(diff_cls, psp.q),
    )
