"""galax: Galactic Dynamics in Jax."""

__all__ = ["AbstractPhaseSpacePosition"]

from dataclasses import replace
from typing import Any

from plum import dispatch

import coordinax as cx
from unxt import uconvert, unitsystem

from .base import AbstractBasePhaseSpacePosition
from .utils import PSPVConvertOptions


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

    def to_units(self, units: Any) -> "AbstractPhaseSpacePosition":
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
# `coordinax.vconvert` dispatches


@dispatch
def vconvert(
    target: PSPVConvertOptions,
    psp: AbstractPhaseSpacePosition,
    /,
    **kwargs: Any,
) -> AbstractPhaseSpacePosition:
    """Convert the phase-space position to a different representation.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax.vecs as cxv
    >>> from galax.coordinates import PhaseSpacePosition

    We can create a phase-space position and convert it to a 6-vector:

    >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                          p=Quantity([4, 5, 6], "km/s"),
    ...                          t=Quantity(0, "Gyr"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64)

    Converting it to a different representation and differential class:

    >>> cx.vconvert({"q": cxv.LonLatSphericalPos, "p": cxv.LonCosLatSphericalVel}, psp)
    PhaseSpacePosition( q=LonLatSphericalPos(...),
                        p=LonCosLatSphericalVel(...),
                        t=Quantity[...](value=f64[], unit=Unit("Gyr")) )

    """
    q_cls = target["q"]
    p_cls = q_cls.differential_cls if (mayp := target.get("p")) is None else mayp
    return replace(
        psp,
        q=psp.q.vconvert(q_cls, **kwargs),
        p=psp.p.vconvert(p_cls, psp.q, **kwargs),
    )


@dispatch
def vconvert(
    target_position_cls: type[cx.vecs.AbstractPos],
    psp: AbstractPhaseSpacePosition,
    /,
    **kwargs: Any,
) -> AbstractPhaseSpacePosition:
    """Convert the phase-space position to a different representation.

    Examples
    --------
    >>> from unxt import Quantity

    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to a 6-vector:

    >>> psp = PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                          p=Quantity([4, 5, 6], "km/s"),
    ...                          t=Quantity(0, "Gyr"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64)

    Converting it to a different representation:

    >>> cx.vconvert(cx.vecs.CylindricalPos, psp)
    PhaseSpacePosition( q=CylindricalPos(...),
                        p=CylindricalVel(...),
                        t=Quantity[...](value=f64[], unit=Unit("Gyr")) )

    If the new representation requires keyword arguments, they can be passed
    through:

    >>> cx.vconvert(cx.vecs.ProlateSpheroidalPos, psp, Delta=Quantity(2.0, "kpc"))
    PhaseSpacePosition( q=ProlateSpheroidalPos(...),
                        p=ProlateSpheroidalVel(...),
                        t=Quantity[...](value=f64[], unit=Unit("Gyr")) )

    """
    target = {"q": target_position_cls, "p": target_position_cls.differential_cls}
    return vconvert(target, psp, **kwargs)
