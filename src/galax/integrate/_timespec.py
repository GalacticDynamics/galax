"""Time specification helper function.

Helper function for turning different ways of specifying the integration times
into an array of times.
"""

__all__ = ["parse_time_specification"]

from typing import TypeVar, cast, overload

import jax
import jax.numpy as jnp
from astropy.units import Quantity
from jaxtyping import Array, Float
from plum import dispatch as dispatch_

from galax.typing import FloatScalar
from galax.units import UnitSystem

T = TypeVar("T")


def dispatch(func: T) -> T:  # TODO: fix mypy complaint about untyped decorator
    return cast("T", dispatch_(func))


# ===========================================================================


# Case 1: No inputs
@dispatch
def parse_time_specification_dispatch(
    units: UnitSystem,
    t: None,
    t1: None,
    t2: None,
    n_steps: None,
    dt: None,
) -> Float[Array, "N"]:
    msg = "must specify some combination of t, t1, t2, n_steps, dt"
    raise ValueError(msg)


# Case 2a: t is given
@dispatch  # type: ignore[no-redef]
def parse_time_specification_dispatch(  # noqa: F811
    units: UnitSystem,
    t: Float[Array, "N"] | jax.Array,
    t1: None,
    t2: None,
    n_steps: None,
    dt: None,
) -> Float[Array, "N"]:
    return jnp.asarray(t, dtype=float)


# Case 2b: t is given as a Quantity
@dispatch  # type: ignore[no-redef]
def parse_time_specification_dispatch(  # noqa: F811
    units: UnitSystem,
    t: Quantity,
    t1: None,
    t2: None,
    n_steps: None,
    dt: None,
) -> Float[Array, "N"]:
    return jnp.asarray(t.decompose(units).value, dtype=float)


# Case 3: t1, t2, n_steps are given
@dispatch  # type: ignore[no-redef]
def parse_time_specification_dispatch(  # noqa: F811
    units: UnitSystem,
    t: None,
    t1: int | float | Quantity | FloatScalar,
    t2: int | float | Quantity | FloatScalar,
    n_steps: int,
    dt: None,
) -> Float[Array, "N"]:
    if isinstance(t1, Quantity):
        t1 = t1.decompose(units).value
    if isinstance(t2, Quantity):
        t2 = t2.decompose(units).value

    return jnp.linspace(t1, t2, n_steps, dtype=float)


# Case 4: t1, n_steps, dt are given
@dispatch  # type: ignore[no-redef]
def parse_time_specification_dispatch(  # noqa: F811
    units: UnitSystem,
    t: None,
    t1: float | Quantity | FloatScalar,
    t2: None,
    n_steps: int,
    dt: float | Quantity | FloatScalar,
) -> Float[Array, "N"]:
    if isinstance(t1, Quantity):
        t1 = t1.decompose(units).value
    if isinstance(dt, Quantity):
        dt = dt.decompose(units).value

    return jnp.arange(t1, t1 + dt * n_steps, dt, dtype=float)


# Case 5: t1, t2, dt are given
@dispatch  # type: ignore[no-redef]
def parse_time_specification_dispatch(  # noqa: F811
    units: UnitSystem,
    t: None,
    t1: float | Quantity | FloatScalar,
    t2: float | Quantity | FloatScalar,
    n_steps: None,
    dt: float | Quantity | FloatScalar,
) -> Float[Array, "N"]:
    if isinstance(t1, Quantity):
        t1 = t1.decompose(units).value
    if isinstance(t2, Quantity):
        t2 = t2.decompose(units).value
    if isinstance(dt, Quantity):
        dt = dt.decompose(units).value

    return jnp.arange(t1, t2, dt, dtype=float)


# Case 6: t1, dt array are given
@dispatch  # type: ignore[no-redef]
def parse_time_specification_dispatch(  # noqa: F811
    units: UnitSystem,
    t: None,
    t1: float | Quantity | FloatScalar,
    t2: None,
    n_steps: None,
    dt: Quantity | Float[Array, "N"],
) -> Float[Array, "N"]:
    if isinstance(t1, Quantity):
        t1 = t1.decompose(units).value
    if isinstance(dt, Quantity):
        dt = dt.decompose(units).value

    return t1 + jnp.cumsum(dt, dtype=float)


# ===========================================================================


@overload
def parse_time_specification(
    units: UnitSystem,
    *,
    t: Quantity | Float[Array, "N"],
    t1: None,
    t2: None,
    n_steps: None,
    dt: None,
) -> Float[Array, "N"]:
    ...


@overload
def parse_time_specification(
    units: UnitSystem,
    *,
    t: None,
    t1: float | Quantity | FloatScalar,
    t2: float | Quantity | FloatScalar,
    n_steps: int,
    dt: None,
) -> Float[Array, "N"]:
    ...


@overload
def parse_time_specification(
    units: UnitSystem,
    *,
    t: None,
    t1: float | Quantity | FloatScalar,
    t2: None,
    n_steps: int,
    dt: float | Quantity | FloatScalar,
) -> Float[Array, "N"]:
    ...


@overload
def parse_time_specification(
    units: UnitSystem,
    *,
    t: None,
    t1: float | Quantity | FloatScalar,
    t2: float | Quantity | FloatScalar,
    n_steps: None,
    dt: float | Quantity | FloatScalar,
) -> Float[Array, "N"]:
    ...


@overload
def parse_time_specification(
    units: UnitSystem,
    *,
    t: None,
    t1: float | Quantity | FloatScalar,
    t2: None,
    n_steps: None,
    dt: Quantity | Float[Array, "N"],
) -> Float[Array, "N"]:
    ...


def parse_time_specification(
    units: UnitSystem,
    *,
    t: Quantity | Float[Array, "N"] = None,
    t1: float | Quantity | FloatScalar | None = None,
    t2: float | Quantity | FloatScalar | None = None,
    n_steps: int | None = None,
    dt: float | Quantity | FloatScalar | Float[Array, "N"] | None = None,
) -> Float[Array, "N"]:
    """Return an array of times given a few combinations of kwargs.

    Parameters
    ----------
    units : UnitSystem
        The unit system to use.

    t : Array[N], optional keyword-only
        An array of times.
    t1, t2 : numeric scalar, optional keyword-only
        An initial/final time.
    n_steps : int, optional keyword-only
        The number of steps between an initial and final time.
    dt : numeric scalar, optional keyword-only
        A fixed timestep.

    Notes
    -----
    The following combinations of kwargs are allowed: The groups are separated
    by whether they have a pre-fixed length (suitable for jit) or not.

    - (t: array)
        The full array of times (dts = t[1:] - t[:-1]).
    - (t1: float, t2: float, n_steps: int)
        Number of steps between an initial time, and a final time.
    - (t1: float, n_steps: float, dt: float)
        An initial time, a number of steps, and a fixed timestep.

    - (t1: float, t2: float, dt: int)
        An initial time, a final time, and a fixed timestep.
    - (t1: float, dt: array)
        An initial time and an array of timesteps dt.
    """
    return parse_time_specification_dispatch(units, t, t1, t2, n_steps, dt)
