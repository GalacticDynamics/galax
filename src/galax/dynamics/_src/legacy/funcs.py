"""galax: Galactic Dynamix in Jax."""

__all__ = ["evaluate_orbit", "default_integrator"]

from collections.abc import Callable
from typing import Any, Literal

import jax
from jaxtyping import Array
from plum import dispatch

import quaxed.numpy as jnp
from unxt.quantity import BareQuantity as FastQ

import galax._custom_types as gt
import galax.coordinates as gc
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
from .integrator import Integrator
from galax.dynamics._src.orbit import HamiltonianField, Orbit

# TODO: enable setting the default integrator
default_integrator: Integrator = Integrator()


_select_w0: Callable[[Array, Array, Array], Array] = jax.numpy.vectorize(
    jax.lax.select, signature="(),(6),(6)->(6)"
)


def orbit_from_psp(w: gc.PhaseSpaceCoordinate, t: gt.QuSzTime) -> Orbit:
    """Create an orbit object from the phase-space position."""
    return Orbit(
        q=w.q,
        p=w.p,
        t=t,
        frame=w.frame,
        interpolant=getattr(w, "interpolant", None),
    )


@dispatch
def evaluate_orbit(
    pot: gp.AbstractPotential,
    w0: gc.PhaseSpaceCoordinate | gc.PhaseSpacePosition | gdt.BtQParr | gt.BtSz6,
    t: Any,
    /,
    *,
    integrator: Integrator | None = None,
    dense: Literal[True, False] = False,
) -> Orbit:
    """Compute an orbit in a potential.

    :class:`~galax.coordinates.PhaseSpaceCoordinate` includes a time in addition
    to the position (and velocity) information, enabling the orbit to be
    evaluated over a time range that is different from the initial time of the
    position.

    Parameters
    ----------
    pot : :class:`~galax.potential.AbstractPotential`
        The potential in which to integrate the orbit.
    w0 : PhaseSpacePosition | Array[number, (*batch, 6)]
        The phase-space position (includes velocity and time) from which to
        integrate. Integration includes the time of the initial position, so be
        sure to set the initial time to the desired value. See the `t` argument
        for more details.

        - :class:`~galax.dynamics.PhaseSpaceCoordinate`[number, (*batch,)]:
            The full phase-space position, including position, velocity, and
            time. `w0` will be integrated from ``w0.t`` to ``t[0]``, then
            integrated from ``t[0]`` to ``t[1]``, returning the orbit calculated
            at `t`.
        - :class:`~galax.dynamics.PhaseSpacePosition`[number, (*batch,)]:
            The partial phase-space position, including position and velocity.
            `w0` will be integrated from ``t[0]`` to ``t[1]``, returning the
            orbit calculated at `t`.
        - tuple[Array[number, (*batch, 3)], Array[number, (*batch, 3)]]:
            A :class:`~galax.coordinates.PhaseSpacePosition` will be
            constructed, interpreting the array as the  'q', 'p', with 't' set
            to ``t[0]``.
        - Array[number, (*batch, 6)]:
            A :class:`~galax.coordinates.PhaseSpacePosition` will be
            constructed, interpreting the array as the  'q', 'p' (each
            Array[number, (*batch, 3)]) arguments, with 't' set to ``t[0]``.
    t: Quantity[number, (time,), 'time']
        Array of times at which to compute the orbit. The first element should
        be the initial time and the last element should be the final time and
        the array should be monotonically moving from the first to final time.
        See the Examples section for options when constructing this argument.

        .. note::

            This is NOT the timesteps to use for integration, which are
            controlled by the `integrator`; the default integrator
            :class:`~galax.integrator.Integrator` uses adaptive timesteps.

    integrator : :class:`~galax.integrate.Integrator`, keyword-only
        Integrator to use.  If `None`, the default integrator
        :class:`~galax.integrator.Integrator` is used.  This integrator is used
        twice: once to integrate from `w0.t` to `t[0]` and then from `t[0]` to
        `t[1]`.

    dense: bool, optional keyword-only
        If `True`, return a dense (interpolated) orbit.  If `False`, return the
        orbit at the requested times.  Default is `False`.

    Examples
    --------
    We start by integrating a single orbit in the potential of a point mass.  A
    few standard imports are needed:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    We can then create the point-mass' potential, with galactic units:

    >>> potential = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can then integrate an initial phase-space position in this potential to
    get an orbit:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([10, 0, 0], "kpc"),
    ...                              p=u.Quantity([0, 200, 0], "km/s"),
    ...                              t=u.Quantity(-100, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0, 1, 4), "Gyr")
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
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
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D(...), p=CartesianVel3D(...),
      t=Quantity([...], unit='Myr'),
      frame=SimulationFrame(),
      interpolant=None
    )

    Or evaluating at a single time:

    >>> orbit = gd.evaluate_orbit(potential, w0, u.Quantity(0.5, "Gyr"))
    >>> orbit
    Orbit(
        q=CartesianPos3D(...), p=CartesianVel3D(...),
        t=Quantity([500.], unit='Myr'),
        frame=SimulationFrame(),
        interpolant=None
    )

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([[10, 0, 0], [10., 0, 0]], "kpc"),
    ...                              p=u.Quantity([[0, 200, 0], [0, 220, 0]], "km/s"),
    ...                              t=u.Quantity([-100, -150], "Myr"))
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D( x=Quantity([...], unit='kpc'), ... ),
      p=CartesianVel3D( ... ),
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
    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit.q[0]  # doctest: +SKIP
    Array([ 9.779, -0.3102,  0.        ], dtype=float64)

    Note that IS NOT the same as ``w0``. ``w0`` is integrated from ``t=0`` to
    ``t=300`` and then from ``t=300`` to ``t=1000``.

    .. note::

        If you want to reproduce :mod:`gala`'s behavior, you can use
        :class:`~galax.dynamics.PhaseSpacePosition`. `evaluate_orbit` will then
        assume ``w0`` is defined at `t`[0].

    """
    # Setup
    units = pot.units
    integrator = default_integrator if integrator is None else integrator
    t = jnp.atleast_1d(FastQ.from_(t, units["time"]))  # ensure t units

    field = HamiltonianField(pot)

    # Parse t0 for the initial integration
    tw0 = w0.t if hasattr(w0, "t") else t[0]

    # Initial integration `w0.t` to `t[0]`.
    # TODO: get diffrax's `solver_state` to speed the second integration.
    # TODO: get diffrax's `controller_state` to speed the second integration.
    # TODO: `max_steps` as kwarg.
    qp0 = integrator(
        field,
        w0,
        tw0,
        jnp.full_like(tw0, fill_value=t[0]),
        dense=False,
    )

    # Orbit integration `t[0]` to `t[-1]`
    # TODO: `max_steps` as kwarg.
    ws = integrator(field, qp0, t[0], t[-1], saveat=t, dense=dense)

    # Return the orbit object
    return orbit_from_psp(ws, t)


@dispatch
def evaluate_orbit(
    pot: gp.AbstractPotential,
    w0: gc.PhaseSpaceCoordinate | gc.PhaseSpacePosition | gt.BtSz6,
    /,
    *,
    t: Any,
    **kwargs: Any,
) -> Orbit:
    """Compute an orbit in a potential, supporting `t` as a keyword argument.

    Examples
    --------
    First some imports:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    We can then create the point-mass' potential, with galactic units:

    >>> potential = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can then integrate an initial phase-space position in this potential to
    get an orbit:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([10, 0, 0], "kpc"),
    ...                              p=u.Quantity([0, 200, 0], "km/s"),
    ...                              t=u.Quantity(-100, "Myr"))
    >>> ts = jnp.linspace(0, 1000, 4)  # (1 Gyr, 4 steps)
    >>> orbit = gd.evaluate_orbit(potential, w0, t=ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D(...), p=CartesianVel3D(...),
      t=Quantity([...], unit='Myr'),
      frame=SimulationFrame(),
      interpolant=None
    )

    """
    return evaluate_orbit(pot, w0, t, **kwargs)
