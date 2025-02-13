"""Orbit objects."""

__all__ = ["Orbit"]

from dataclasses import KW_ONLY

import equinox as eqx

import coordinax as cx
import unxt as u

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from .base import AbstractOrbit


class Orbit(AbstractOrbit):
    """Represents an orbit.

    An orbit is a set of positions and velocities (conjugate momenta) as a
    function of time resulting from the integration of the equations of motion
    in a given potential.

    Examples
    --------
    We can create an orbit by integrating a point mass in a Kepler
    potential:

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    >>> potential = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([8., 0., 0.], "kpc"),
    ...     p=u.Quantity([0., 230, 0.], "km/s"),
    ...     t=u.Quantity(0, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0., 1., 10), "Gyr")

    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
      frame=SimulationFrame(),
      potential=KeplerPotential( ... ),
      interpolant=None
    )

    >>> orbit = gd.evaluate_orbit(potential, w0, ts, dense=True)
    >>> orbit
    Orbit(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
      frame=SimulationFrame(),
      potential=KeplerPotential( ... ),
      interpolant=PhaseSpaceInterpolation( ... )
    )

    >>> orbit(u.Quantity(0.5, "Gyr"))
    Orbit(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(500., dtype=float64, ...), unit='Myr'),
      frame=SimulationFrame(),
      potential=KeplerPotential( ... ),
      interpolant=None
    )

    """

    q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
    """Positions (x, y, z)."""

    p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
    r"""Conjugate momenta ($v_x$, $v_y$, $v_z$)."""

    # TODO: consider how this should be vectorized
    t: gt.QuSzTime | gt.QuSz1 = eqx.field(converter=u.Quantity["time"].from_)
    """Array of times corresponding to the positions."""

    _: KW_ONLY

    frame: gc.frames.SimulationFrame  # TODO: support frames
    """The reference frame of the phase-space position."""

    potential: gp.AbstractPotential
    """Potential in which the orbit was integrated."""

    interpolant: gc.PhaseSpaceObjectInterpolant | None = None
    """The interpolation function."""

    # TODO: figure out public API. This is used by `evaluate_orbit`
    @classmethod
    def _from_psp(
        cls,
        w: gc.PhaseSpaceCoordinate,
        t: gt.QuSzTime,
        potential: gp.AbstractPotential,
    ) -> "Orbit":
        """Create an orbit from a phase-space position."""
        return Orbit(
            q=w.q,
            p=w.p,
            t=t,
            frame=w.frame,
            potential=potential,
            interpolant=getattr(w, "interpolant", None),
        )
