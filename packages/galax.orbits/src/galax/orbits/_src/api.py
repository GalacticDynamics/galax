"""Orbit interfacing with `diffrax`.

This is private API.

"""

__all__ = ["align_circulation", "circulation", "compute_orbit"]


from typing import Any

from plum import dispatch

import galax.potential as gp
from .field_base import AbstractOrbitField
from .orbit import Orbit


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
    >>> import galax.orbits as go

    We can then create a point-mass potential, with galactic units:

    >>> potential = gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([10, 0, 0], "kpc"),
    ...                              p=u.Q([0, 200, 0], "km/s"),
    ...                              t=u.Q(-100, "Myr"))
    >>> ts = u.Q(jnp.linspace(0, 1, 4), "Gyr")
    >>> orbit = go.compute_orbit(potential, w0, ts)
    >>> orbit
    Orbit( q=CartesianPos3D(...), p=CartesianVel3D(...),
           t=Q([...], 'Myr'), frame=SimulationFrame(), interpolant=None )

    Note how there are 4 points in the orbit, corresponding to the 4 requested
    return times. These are the times at which the orbit is evaluated, not the
    times at which the orbit is integrated. The phase-space position `w0` is
    defined at `t=-100`, but the orbit is integrated from `t=0` to `t=1000`.
    Changing the number of times is easy:

    >>> ts = u.Q(jnp.linspace(0, 1, 10), "Gyr")
    >>> orbit = go.compute_orbit(potential, w0, ts)
    >>> orbit
    Orbit( q=CartesianPos3D(...), p=CartesianVel3D(...), t=Q( [...], 'Myr' ),
           frame=SimulationFrame(), interpolant=None )

    Or evaluating at a single time:

    >>> orbit = go.compute_orbit(potential, w0, u.Q(0.5, "Gyr"))
    >>> orbit
    Orbit( q=CartesianPos3D(...), p=CartesianVel3D(...), t=Q(500., 'Myr'),
           frame=SimulationFrame(), interpolant=None )

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([[10, 0, 0], [10., 0, 0]], "kpc"),
    ...                              p=u.Q([[0, 200, 0], [0, 220, 0]], "km/s"),
    ...                              t=u.Q([-100, -150], "Myr"))
    >>> orbit = go.compute_orbit(potential, w0, ts)
    >>> orbit
    Orbit( q=CartesianPos3D(...), p=CartesianVel3D(...), t=Q( [...], 'Myr' ),
           frame=SimulationFrame(), interpolant=None )

    :class:`~galax.coordinates.PhaseSpacePosition` has a ``t`` argument for the
    time at which the position is given. As noted earlier, this can be used to
    integrate from a different time than the initial time of the position:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([10, 0, 0], "kpc"),
    ...                              p=u.Q([0, 200, 0], "km/s"),
    ...                              t=u.Q(0, "Myr"))
    >>> ts = u.Q(jnp.linspace(0.3, 1, 8), "Gyr")
    >>> orbit = go.compute_orbit(potential, w0, ts)
    >>> orbit.q[0]  # doctest: +SKIP
    Array([ 9.779, -0.3102,  0.        ], dtype=float64)

    Note that IS NOT the same as ``w0``. ``w0`` is integrated from ``t=0`` to
    ``t=300`` and then from ``t=300`` to ``t=1000``.

    .. note::

        If you want to reproduce :mod:`gala`'s behavior, you can use
        :class:`~galax.coordinates.PhaseSpacePosition`. `compute_orbit` will then
        assume ``w0`` is defined at `t`[0].

    >>> w0 = gc.PhaseSpacePosition(q=u.Q([10, 0, 0], "kpc"),
    ...                            p=u.Q([0, 200, 0], "km/s"))
    >>> ts = u.Q(jnp.linspace(0, 1, 4), "Gyr")
    >>> orbit = go.compute_orbit(potential, w0, ts)
    >>> orbit
    Orbit( q=CartesianPos3D(...), p=CartesianVel3D(...), t=Q([...], 'Myr'),
           frame=SimulationFrame(), interpolant=None )

    A constant-step solver needs a step size, given by ``dt0``. A plain number
    is in the field's time unit, and a `unxt.Quantity` is converted. Only its
    size is used; the direction of integration sets its sign, so the same
    ``dt0`` works forward and backward:

    >>> import diffrax as dfx
    >>> import galax.fields as gf
    >>> solver = go.OrbitSolver(gf.Leapfrog(),
    ...                         stepsize_controller=dfx.ConstantStepSize())
    >>> ts = u.Q(jnp.linspace(0, -1, 4), "Gyr")
    >>> orbit = go.compute_orbit(potential, w0, ts, solver=solver,
    ...                          dt0=u.Q(1, "Myr"))
    >>> orbit.t
    Q([    0.        ,  -333.33333333,  -666.66666667, -1000.        ], 'Myr')

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def circulation(*args: Any) -> Any:
    r"""Return which coordinate axes an orbit circulates about.

    Axis ``i`` circulates when the angular momentum component
    :math:`L_i = (\mathbf{q} \times \mathbf{p})_i` keeps one sign at every
    sample and never falls below
    :math:`\max(10^{-13}, 64\,\epsilon)` times
    :math:`|\mathbf{q}|\,|\mathbf{p}|` at that sample, where :math:`\epsilon`
    is the machine epsilon of the float dtype in use (``1e-13`` for float64,
    about ``7.6e-6`` for float32). Half-precision inputs (float16, bfloat16)
    are promoted to float32 first, so they use the float32 floor. A z-tube
    orbit gives
    ``[False, False, True]``, an x-tube orbit gives ``[True, False, False]``
    and a box orbit gives all `False`. Samples where
    ``q`` or ``p`` is not finite are left out of the check, and an orbit with
    no finite samples circulates about no axis. Each sample is checked on its
    own, so the result does not depend on the order of the samples. An input
    with no samples raises a `ValueError`.

    This follows :meth:`gala.dynamics.Orbit.circulation` with these
    differences. The floor here is relative to
    :math:`|\mathbf{q}|\,|\mathbf{p}|`, so the result does not depend on the
    units, while gala uses an absolute floor of ``1e-13``. A sample with ``q``
    parallel to ``p`` therefore does not circulate, even when rounding leaves a
    small nonzero :math:`\mathbf{L}`. gala skips the floor at the first sample,
    so in gala an orbit with one sample circulates about every axis, while
    here every sample is checked. Samples that are not finite are skipped
    here, while in gala a single NaN sample makes the orbit circulate about no
    axis. And this function returns booleans where gala returns integers.

    The overloads are:

    - ``circulation(orbit)`` checks the stored samples of an
      `~galax.orbits.AbstractOrbit`.
    - ``circulation(orbit, t)`` first evaluates the orbit's interpolant at the
      times ``t`` with ``orbit(t)`` and then checks those samples. This finds
      sign flips that fall between stored samples and lets you choose the time
      window. ``t`` is a time Quantity or an array in the orbit's time unit,
      with shape ``(T,)``, and the orbit must have been computed with
      ``dense=True``. Any other shape raises a `ValueError`. ``t=None`` uses
      the stored samples, as ``circulation(orbit)`` does.
    - ``circulation(w)`` checks a
      `~galax.coordinates.AbstractPhaseSpaceCoordinate` whose last batch axis
      is time. This includes composite coordinates, whose components are
      joined along the time axis. A coordinate has no interpolant, so
      ``circulation(w, t)`` raises a `TypeError`.
    - ``circulation(q, p)`` checks arrays or Quantities of Cartesian positions
      and velocities with shape ``(..., T, 3)``. Axis ``-2`` is the sample
      (time) axis.

    An orbit or coordinate stores positions and velocities with shape
    ``(*batch, T)``, where the time axis is the last batch axis, and an
    orbit's ``t`` has shape ``(T,)``. Its Cartesian components therefore have
    shape ``(*batch, T, 3)``, and the result has one row of 3 booleans per
    orbit.

    Returns
    -------
    Bool[Array, "*batch 3"]
        Whether each orbit circulates about x, y and z.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.orbits as go

    Two circular orbits in a Kepler potential, one in the xy plane and one in
    the yz plane:

    >>> pot = gp.KeplerPotential(m_tot=u.Q(1e11, "Msun"), units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Q([[8.0, 0, 0], [0, 8.0, 0]], "kpc"),
    ...     p=u.Q([[0, 230.0, 0], [0, 0, 230.0]], "km/s"),
    ...     t=u.Q([0.0, 0.0], "Myr"))
    >>> ts = u.Q(jnp.linspace(0, 1, 50), "Gyr")
    >>> orbit = go.compute_orbit(pot, w0, ts, dense=True)
    >>> orbit.shape
    (2, 50)

    >>> go.circulation(orbit)
    Array([[False, False,  True],
           [ True, False, False]], dtype=bool)

    The method calls this function:

    >>> orbit.circulation()
    Array([[False, False,  True],
           [ True, False, False]], dtype=bool)

    The orbit can be checked on a finer time grid with its interpolant:

    >>> go.circulation(orbit, u.Q(jnp.linspace(0, 1, 500), "Gyr"))
    Array([[False, False,  True],
           [ True, False, False]], dtype=bool)

    Cartesian arrays or Quantities work too:

    >>> q = u.Q([[8.0, 0, 0], [0, 8.0, 0]], "kpc")
    >>> p = u.Q([[0, 230.0, 0], [-230.0, 0, 0]], "km/s")
    >>> go.circulation(q, p)
    Array([False, False,  True], dtype=bool)

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def align_circulation(*args: Any) -> Any:
    """Swap an orbit's circulation axis with z.

    The circulation ``circ`` is a boolean array of shape ``(*batch, 3)`` from
    `galax.orbits.circulation`. If it is omitted, it is computed from the
    stored samples. A ``circ`` of any other shape raises a `ValueError`. An
    orbit that circulates about z, or about no axis, is left unchanged.
    Otherwise the circulating axis is swapped with z in both the positions
    and velocities, and x is chosen over y when both circulate.
    This follows :meth:`gala.dynamics.Orbit.align_circulation_with_z`.

    The swap is a selection between permuted copies of the inputs, so it is
    differentiable in the positions and velocities.

    The overloads are:

    - ``align_circulation(orbit, circ=None)`` returns ``(orbit', axis)``,
      where ``orbit'`` has the same type as ``orbit``. The returned orbit has
      no interpolant, because the interpolant describes the unswapped axes,
      so calling ``orbit'(t)`` raises an error.
    - ``align_circulation(w, circ=None)`` returns ``(w', axis)`` for a
      `~galax.coordinates.AbstractPhaseSpaceCoordinate` with a time axis,
      where ``w'`` has the same type as ``w``. Each component of a composite
      coordinate is swapped with the same axis.
    - ``align_circulation(q, p, circ=None)`` returns ``(q', p', axis)`` for
      Cartesian arrays or Quantities of shape ``(..., T, 3)``.

    The returned positions and velocities are Cartesian. ``axis`` has shape
    ``(*batch,)``. It is 2 when nothing changed, and otherwise it is the axis
    that was swapped with z.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.orbits as go

    A circular orbit in the yz plane circulates about x:

    >>> pot = gp.KeplerPotential(m_tot=u.Q(1e11, "Msun"), units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([0, 8.0, 0], "kpc"),
    ...                              p=u.Q([0, 0, 230.0], "km/s"),
    ...                              t=u.Q(0.0, "Myr"))
    >>> ts = u.Q(jnp.linspace(0, 1, 50), "Gyr")
    >>> orbit = go.compute_orbit(pot, w0, ts)

    >>> aligned, axis = go.align_circulation(orbit)
    >>> int(axis)
    0
    >>> go.circulation(aligned)
    Array([False, False,  True], dtype=bool)

    The method calls this function:

    >>> aligned, axis = orbit.align_circulation()
    >>> type(aligned).__name__
    'Orbit'

    With arrays the swapped positions and velocities are returned:

    >>> q = u.Q([[0, 8.0, 0], [0, 0, 8.0]], "kpc")
    >>> p = u.Q([[0, 0, 230.0], [0, -230.0, 0]], "km/s")
    >>> q2, p2, axis = go.align_circulation(q, p)
    >>> q2
    Q([[0., 8., 0.],
       [8., 0., 0.]], 'kpc')

    """
    raise NotImplementedError  # pragma: no cover
