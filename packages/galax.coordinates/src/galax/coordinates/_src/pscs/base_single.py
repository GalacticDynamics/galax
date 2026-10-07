"""galax: Galactic Dynamics in Jax."""

__all__ = ["AbstractBasicPhaseSpaceCoordinate"]

from dataclasses import replace

from plum import dispatch

import unxt as u

from .base import AbstractPhaseSpaceCoordinate


class AbstractBasicPhaseSpaceCoordinate(AbstractPhaseSpaceCoordinate):
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


    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to different units:

    >>> psp = gc.PhaseSpaceCoordinate(q=u.Q([1, 2, 3], "kpc"),
    ...                               p=u.Q([4, 5, 6], "km/s"),
    ...                               t=u.Q(0, "Gyr"))
    >>> psp.uconvert("solarsystem")
    PhaseSpaceCoordinate(
      q=CartesianPos3D( x=Q(2.06264806e+08, 'AU'), ... ),
      p=CartesianVel3D( x=Q(0.84379811, 'AU / yr'), ... ),
      t=Q(0., 'yr'), frame=SimulationFrame() )

    """


# =============================================================================
# Dispatches


# -----------------------------------------------
# `unxt.uconvert` dispatches


# TODO: make precedence=0
@dispatch(precedence=1)  # type: ignore[call-overload,misc]
def uconvert(
    units: u.AbstractUnitSystem | str, wt: AbstractBasicPhaseSpaceCoordinate
) -> AbstractBasicPhaseSpaceCoordinate:
    """Convert the components to the given units."""
    usys = u.unitsystem(units)
    return replace(
        wt, q=wt.q.uconvert(usys), p=wt.p.uconvert(usys), t=wt.t.uconvert(usys["time"])
    )
