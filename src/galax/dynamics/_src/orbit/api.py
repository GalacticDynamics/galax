"""Orbit interfacing with `diffrax`.

This is private API.

"""

__all__ = ["compute_orbit"]


from typing import Any

from plum import dispatch

import galax.potential as gp
from .orbit import Orbit
from galax.dynamics._src.orbit.field_base import AbstractOrbitField


@dispatch.abstract
def compute_orbit(
    field: AbstractOrbitField | gp.AbstractPotential, /, *args: Any, **kwargs: Any
) -> Orbit:
    """Compute an orbit.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    We can then create a point-mass potential, with galactic units:

    >>> potential = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([10, 0, 0], "kpc"),
    ...                              p=u.Quantity([0, 200, 0], "km/s"),
    ...                              t=u.Quantity(-100, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0, 1, 4), "Gyr")
    >>> orbit = gd.compute_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D(...), p=CartesianVel3D(...),
      t=Quantity([...], unit='Myr'),
      frame=SimulationFrame(),
      interpolant=None
    )

    Note how there are 4 points in the orbit, corresponding to the 4 requested
    return times. These are the times at which the orbit is evaluated, not the
    times at which the orbit is integrated. The phase-space position `w0` is
    defined at `t=-100`, but the orbit is integrated from `t=0` to `t=1000`.
    Changing the number of times is easy:

    >>> ts = u.Quantity(jnp.linspace(0, 1, 10), "Gyr")
    >>> orbit = gd.compute_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D(...), p=CartesianVel3D(...),
      t=Quantity([...], unit='Myr'),
      frame=SimulationFrame(),
      interpolant=None
    )

    Or evaluating at a single time:

    >>> orbit = gd.compute_orbit(potential, w0, u.Quantity(0.5, "Gyr"))
    >>> orbit
    Orbit(
        q=CartesianPos3D(...), p=CartesianVel3D(...),
        t=Quantity(500., unit='Myr'),
        frame=SimulationFrame(),
        interpolant=None
    )

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([[10, 0, 0], [10., 0, 0]], "kpc"),
    ...                              p=u.Quantity([[0, 200, 0], [0, 220, 0]], "km/s"),
    ...                              t=u.Quantity([-100, -150], "Myr"))
    >>> orbit = gd.compute_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D( x=Quantity([...], unit='kpc'), ... ),
      p=CartesianVel3D(...),
      t=Quantity([...], unit='Myr'),
      frame=SimulationFrame(),
      interpolant=None
    )

    :class:`~galax.dynamics.PhaseSpacePosition` has a ``t`` argument for the
    time at which the position is given. As noted earlier, this can be used to
    integrate from a different time than the initial time of the position:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([10, 0, 0], "kpc"),
    ...                              p=u.Quantity([0, 200, 0], "km/s"),
    ...                              t=u.Quantity(0, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0.3, 1, 8), "Gyr")
    >>> orbit = gd.compute_orbit(potential, w0, ts)
    >>> orbit.q[0]  # doctest: +SKIP
    Array([ 9.779, -0.3102,  0.        ], dtype=float64)

    Note that IS NOT the same as ``w0``. ``w0`` is integrated from ``t=0`` to
    ``t=300`` and then from ``t=300`` to ``t=1000``.

    .. note::

        If you want to reproduce :mod:`gala`'s behavior, you can use
        :class:`~galax.dynamics.PhaseSpacePosition`. `evaluate_orbit` will then
        assume ``w0`` is defined at `t`[0].

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s"))
    >>> ts = u.Quantity(jnp.linspace(0, 1, 4), "Gyr")
    >>> orbit = gd.compute_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D(...), p=CartesianVel3D(...),
      t=Quantity([...], unit='Myr'),
      frame=SimulationFrame(),
      interpolant=None
    )

    """
    raise NotImplementedError  # pragma: no cover
