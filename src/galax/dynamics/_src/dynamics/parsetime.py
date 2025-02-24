"""Utilities for dynamics."""

__all__ = ["parse_time_specification"]


from collections.abc import Sequence
from numbers import Number
from typing import Annotated as Ann, Any
from typing_extensions import Doc

from jaxtyping import Array, ArrayLike, Shaped
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt._src.units.api import AstropyUnits as Unit


def parse_time_specification(
    units: Ann[
        u.AbstractUnitSystem | None, Doc("The unit system in which to express times.")
    ] = None,
    *,
    t: Ann[u.AbstractQuantity | None, Doc("full time array.")] = None,
    t0: Ann[u.AbstractQuantity | None, Doc("The starting time.")] = None,
    dt: Ann[u.AbstractQuantity | None, Doc("The time step(s).")] = None,
    n_steps: Ann[int | None, Doc("The number of steps.")] = None,
    t1: Ann[u.AbstractQuantity | None, Doc("The final time (inclusive).")] = None,
) -> Ann[u.AbstractQuantity, Doc("An array of times.")]:
    """Construct and return an array of times.

    This is useful for constructing the `ts` argument of `diffrax.SaveAt`.

    Options:

    - t
        An array of times.
    - t0, t1, n_steps
        A starting time, final time (inclusive), and a number of steps between.
        Equivalent to using :func:`jax.numpy.linspace`.
    - t0, dt, n_steps
        A starting time, fixed timestep dt, and a number of steps to run for.
    - t0, dt, t1
        A starting time, fixed timestep dt, and final time.
        Equivalent to using :func:`jax.numpy.arange`.
    - t0, dt (array or sequence)
        A starting time and an array / sequence of timesteps.

    Parameters
    ----------
    units : UnitSystem-like
        The unit system to use. See :func:`unxt.unitsystem`.

    t : Quantity['time'], optional
        An array of times.
    t0 : Quantity['time'], optional
        The starting time.
    dt : Quantity['time'], optional
        The timestep.
    n_steps : int, optional
        The number of time steps.
    t1 : Quantity['time'], optional
        The final time.

    Returns
    -------
    Quantity['time']
        An array of times.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.integrate as gdi

    From a Quantity:

    >>> gdi.parse_time_specification(t=u.Quantity([1, 2, 3], "Myr"))
    Quantity['time'](Array([1, 2, 3], dtype=int64), unit='Myr')

    In the units of a specific unit system:

    >>> gdi.parse_time_specification(units="galactic", t=u.Quantity([1, 2, 3], "Myr"))
    Quantity['time'](Array([1, 2, 3], dtype=int64), unit='Myr')

    >>> gdi.parse_time_specification(units="solarsystem", t=u.Quantity([1, 2, 3], "Myr"))
    Quantity['time'](Array([1000000., 2000000., 3000000.], dtype=float64, weak_type=True), unit='yr')

    From a non-Quantity array, interpreted as having the time units of the given
    unit system:

    >>> gdi.parse_time_specification(units="solarsystem", t=[1, 2, 3])
    Quantity['time'](Array([1, 2, 3], dtype=int64), unit='yr')

    >>> gdi.parse_time_specification(units="galactic", t=[1, 2, 3])
    Quantity['time'](Array([1, 2, 3], dtype=int64), unit='Myr')

    From the starting time, ending time, and number of steps:

    >>> gdi.parse_time_specification(t0=u.Quantity(0, "Myr"), t1=u.Quantity(1, "Myr"), n_steps=3)
    Quantity['time'](Array([0. , 0.5, 1. ], dtype=float64), unit='Myr')

    In the units of a specific unit system:

    >>> gdi.parse_time_specification(units="solarsystem", t0=u.Quantity(0, "Myr"), t1=u.Quantity(1, "Myr"), n_steps=3)
    Quantity['time'](Array([      0.,  500000., 1000000.], dtype=float64), unit='yr')

    From non-Quantity numbers, interpreted as having the time units of the given
    unit system:

    >>> gdi.parse_time_specification(units="solarsystem", t0=0, t1=1, n_steps=3)
    Quantity['time'](Array([0. , 0.5, 1. ], dtype=float64), unit='yr')

    From the starting time, timestep, and number of steps:

    >>> gdi.parse_time_specification(t0=u.Quantity(0, "Myr"), dt=u.Quantity(1, "Myr"), n_steps=3)
    Quantity['time'](Array([0. , 1.5, 3. ], dtype=float64), unit='Myr')

    In the units of a specific unit system:

    >>> gdi.parse_time_specification(units="galactic", t0=u.Quantity(0, "Myr"), dt=u.Quantity(1, "Myr"), n_steps=3)
    Quantity['time'](Array([0. , 1.5, 3. ], dtype=float64), unit='Myr')

    From non-Quantity numbers, interpreted as having the time units of the given
    unit system:

    >>> gdi.parse_time_specification(units="solarsystem", t0=0, dt=1, n_steps=3)
    Quantity['time'](Array([0. , 1.5, 3. ], dtype=float64), unit='yr')

    From the starting time, time step, and ending time:

    >>> gdi.parse_time_specification(t0=u.Quantity(0, "Myr"), dt=u.Quantity(1, "Myr"), t1=u.Quantity(3, "Myr"))
    Quantity['time'](Array([0, 1, 2], dtype=int64), unit='Myr')

    From non-Quantity numbers, interpreted as having the time units of the given
    unit system:

    >>> gdi.parse_time_specification(units="galactic", t0=0, dt=1, t1=3)
    Quantity['time'](Array([0, 1, 2], dtype=int64), unit='Myr')

    >>> gdi.parse_time_specification(units="solarsystem", t0=0, dt=1, t1=3)
    Quantity['time'](Array([0, 1, 2], dtype=int64), unit='yr')

    From the starting time and an array of time steps:

    >>> gdi.parse_time_specification(t0=u.Quantity(0, "Myr"), dt=u.Quantity([1, 2], "Myr"))
    Quantity['time'](Array([0, 1, 3], dtype=int64), unit='Myr')

    From non-Quantity numbers, interpreted as having the time units of the given
    unit system:

    >>> import jax.numpy as jnp
    >>> gdi.parse_time_specification(units="galactic", t0=0, dt=jnp.asarray([1, 2]))
    Quantity['time'](Array([0, 1, 3], dtype=int64), unit='Myr')

    From non-Quantity numbers, interpreted as having the time units of the given
    unit system:

    >>> import jax.numpy as jnp
    >>> gdi.parse_time_specification(units="galactic", t0=0, dt=[1, 2])
    Quantity['time'](Array([0, 1, 3], dtype=int64), unit='Myr')

    """  # noqa: E501
    return parse_time_spec(units, t, t0, dt, n_steps, t1)


# -----------------------------------------------


def _parse_to_time_unit(obj: Any, /) -> Unit | None:
    return u.unitsystem(obj)["time"] if obj is not None else None


@dispatch
def parse_time_spec(
    units: Any,
    t: u.AbstractQuantity | ArrayLike | list[int] | tuple[int, ...],
    t0: None,  # noqa: ARG001
    dt: None,  # noqa: ARG001
    n_steps: None,  # noqa: ARG001
    t1: None,  # noqa: ARG001
    /,
) -> u.AbstractQuantity:
    """Return the given array of times as a :class:`unxt.Quantity`."""
    unit = _parse_to_time_unit(units)
    return u.Quantity.from_(t, unit)


@dispatch
def parse_time_spec(
    units: Any,
    t: None,  # noqa: ARG001
    t0: Any,
    dt: None,  # noqa: ARG001
    n_steps: int,
    t1: Any,
    /,
) -> Shaped[u.AbstractQuantity, "{n_steps}"]:
    """Return a time array from the initial and final times and number of steps."""
    unit = _parse_to_time_unit(units)
    return u.Quantity.from_(jnp.linspace(t0, t1, n_steps), unit)


@dispatch
def parse_time_spec(
    units: Any,
    t: None,
    t0: Any,
    dt: Any,
    n_steps: int,
    t1: None,  # noqa: ARG001
    /,
) -> Shaped[u.AbstractQuantity, "{n_steps}"]:
    """Return a time array from the initial time, timestep, and number of steps."""
    return parse_time_spec(units, t, t0, None, n_steps, t0 + dt * n_steps)


@dispatch
def parse_time_spec(
    units: Any,
    t: None,  # noqa: ARG001
    t0: Any,
    dt: Any,
    n_steps: None,  # noqa: ARG001
    t1: Any,
    /,
) -> u.AbstractQuantity:
    """Return a time array from the initial and final times and the time step."""
    unit = _parse_to_time_unit(units)
    return u.Quantity.from_(jnp.arange(t0, t1, dt), unit)


@dispatch
def parse_time_spec(
    units: Any,
    t: None,  # noqa: ARG001
    t0: Shaped[u.AbstractQuantity, ""] | Shaped[Array, ""] | Number | float | int,
    dt: Shaped[u.AbstractQuantity, "N"] | Shaped[Array, "N"],
    n_steps: None,  # noqa: ARG001
    t1: None,  # noqa: ARG001
    /,
) -> Shaped[u.AbstractQuantity, "N+1"]:
    """Return a time array from the initial time and an array of time steps."""
    unit = _parse_to_time_unit(units)
    t0 = u.Quantity.from_(t0, unit)
    dt = u.Quantity.from_(dt, unit)
    return jnp.concat((t0[None], jnp.cumsum(t0 + dt)))


@dispatch
def parse_time_spec(
    units: Any, t: None, t0: Any, dt: Sequence[Any], n_steps: None, t1: None, /
) -> u.AbstractQuantity:
    """Return a time array from the initial time and a Sequence of time steps."""
    return parse_time_spec(units, t, t0, jnp.asarray(dt), n_steps, t1)
