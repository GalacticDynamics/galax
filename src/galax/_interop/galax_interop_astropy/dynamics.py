"""Compatibility."""

__all__: list[str] = []

from typing import Literal

from astropy.units import Quantity as APYQuantity
from plum import convert, dispatch

from unxt import Quantity

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import galax.typing as gt

# =============================================================================
# evaluate_orbit


@dispatch  # type: ignore[misc]
def evaluate_orbit(
    pot: gp.AbstractPotentialBase,
    w0: gc.PhaseSpacePosition | gt.BatchVec6,
    t: APYQuantity,
    *,
    integrator: gd.integrate.Integrator | None = None,
    interpolated: Literal[True, False] = False,
) -> gd.Orbit:
    """Compute an orbit in a potential.

    This is the Astropy-compatible version of the function.

    Examples
    --------
    We start by integrating a single orbit in the potential of a point mass.  A
    few standard imports are needed:

    >>> import numpy as np
    >>> from astropy.units import Quantity
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    We can then create the point-mass' potential, with galactic units:

    >>> potential = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

    We can then integrate an initial phase-space position in this potential to
    get an orbit:

    >>> w0 = gc.PhaseSpacePosition(q=Quantity([10., 0., 0.], "kpc"),
    ...                            p=Quantity([0., 0.1, 0.], "km/s"),
    ...                            t=Quantity(-100, "Myr"))
    >>> ts = Quantity(np.linspace(0., 1., 4), "Gyr")

    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPosition3D(...), p=CartesianVelocity3D(...),
      t=Quantity[...](value=f64[4], unit=Unit("Myr")),
      potential=KeplerPotential(...),
      interpolant=None
    )

    >>> ts = Quantity(np.linspace(0., 1., 10), "Gyr")
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPosition3D(...), p=CartesianVelocity3D(...),
      t=Quantity[...](value=f64[10], unit=Unit("Myr")),
      potential=KeplerPotential(...),
      interpolant=None
    )

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpacePosition(q=Quantity([[10., 0, 0], [10., 0, 0]], "kpc"),
    ...                            p=Quantity([[0, 0.1, 0], [0, 0.2, 0]], "km/s"),
    ...                            t=Quantity([-100, -150], "Myr"))
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPosition3D(
        x=Quantity[PhysicalType('length')](value=f64[2,10], unit=Unit("kpc")),
        ...
      ),
      p=CartesianVelocity3D(...),
      t=Quantity[...](value=f64[10], unit=Unit("Myr")),
      potential=KeplerPotential(...),
      interpolant=None
    )

    :class:`~galax.dynamics.PhaseSpacePosition` has a ``t`` argument for the
    time at which the position is given. As noted earlier, this can be used to
    integrate from a different time than the initial time of the position:

    >>> w0 = gc.PhaseSpacePosition(q=Quantity([10., 0., 0.], "kpc"),
    ...                            p=Quantity([0., 0.1, 0.], "km/s"),
    ...                            t=Quantity(0, "Myr"))
    >>> ts = Quantity(np.linspace(0.3, 1.0, 8), "Gyr")
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit.q[0]  # doctest: +SKIP
    Array([ 9.779, -0.3102,  0.        ], dtype=float64)

    """
    orbit: gd.Orbit = gd.evaluate_orbit(
        pot,
        w0,
        convert(t, Quantity),
        integrator=integrator,
        interpolated=interpolated,
    )
    return orbit
