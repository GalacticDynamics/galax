"""galax: Galactic Dynamics in Jax."""

__all__ = ["AbstractOnePhaseSpacePosition"]

from dataclasses import replace
from typing import Any

from plum import dispatch

import coordinax as cx
import unxt as u

from .base import AbstractBasePhaseSpacePosition
from .utils import PSPVConvertOptions


class AbstractOnePhaseSpacePosition(AbstractBasePhaseSpacePosition):
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

    def to_units(self, units: Any) -> "AbstractOnePhaseSpacePosition":
        """Return a new object with the components converted to the given units."""
        usys = u.unitsystem(units)
        return replace(
            self,
            q=self.q.uconvert(usys),
            p=self.p.uconvert(usys),
            t=u.uconvert(usys["time"], self.t) if self.t is not None else None,
        )


# =============================================================================
# helper functions


# -----------------------------------------------
# `coordinax.vconvert` dispatches


@dispatch
def vconvert(
    target: PSPVConvertOptions,
    psp: AbstractOnePhaseSpacePosition,
    /,
    **kwargs: Any,
) -> AbstractOnePhaseSpacePosition:
    """Convert the phase-space position to a different representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to a 6-vector:

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

    Converting it to a different representation and differential class:

    >>> cx.vconvert({"q": cxv.LonLatSphericalPos, "p": cxv.LonCosLatSphericalVel}, psp)
    PhaseSpacePosition( q=LonLatSphericalPos(...),
                        p=LonCosLatSphericalVel(...),
                        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
                        frame=SimulationFrame() )

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
    psp: AbstractOnePhaseSpacePosition,
    /,
    **kwargs: Any,
) -> AbstractOnePhaseSpacePosition:
    """Convert the phase-space position to a different representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to a 6-vector:

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

    Converting it to a different representation:

    >>> cx.vconvert(cx.vecs.CylindricalPos, psp)
    PhaseSpacePosition( q=CylindricalPos(...),
                        p=CylindricalVel(...),
                        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
                        frame=SimulationFrame() )

    If the new representation requires keyword arguments, they can be passed
    through:

    >>> cx.vconvert(cx.vecs.ProlateSpheroidalPos, psp, Delta=u.Quantity(2.0, "kpc"))
    PhaseSpacePosition( q=ProlateSpheroidalPos(...),
                        p=ProlateSpheroidalVel(...),
                        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
                        frame=SimulationFrame() )

    """
    target = {"q": target_position_cls, "p": target_position_cls.differential_cls}
    return vconvert(target, psp, **kwargs)
