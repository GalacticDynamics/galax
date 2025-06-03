"""Register primitives for PSPs."""

__all__: list[str] = []

from dataclasses import replace

import jax
from quax import register

import quaxed.numpy as jnp

from .base import AbstractPhaseSpaceCoordinate


@register(jax.lax.add_p)  # type: ignore[misc]
def add_wts(
    wt1: AbstractPhaseSpaceCoordinate, wt2: AbstractPhaseSpaceCoordinate, /
) -> AbstractPhaseSpaceCoordinate:
    """Add two phase-space positions.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc

    >>> w1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                              p=u.Quantity([4, 5, 6], "km/s"),
    ...                              t=u.Quantity(0, "Gyr"))
    >>> w2 = gc.PhaseSpaceCoordinate(q=u.Quantity([-1, -2, -3], "kpc"),
    ...                              p=u.Quantity([-4, -5, -6], "km/s"),
    ...                              t=u.Quantity(0, "Gyr"))
    >>> w3 = w1 + w2
    >>> w3
    PhaseSpaceCoordinate(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity(0, unit='Gyr'),
      frame=SimulationFrame()
    )

    >>> w3.q.x.value
    Array(0, dtype=int64)

    If the times are different, an error is raised:

    >>> from dataclassish import replace
    >>> w4 = replace(w2, t=u.Quantity(1, "Gyr"))
    >>> try: w1 + w4
    ... except Exception: print("Error")
    Error

    """
    if not isinstance(wt2, type(wt1)):
        msg = f"Cannot add {type(wt1)} and {type(wt2)}"
        raise TypeError(msg)

    # Check the times are the same
    if jnp.any(wt1.t != wt2.t):
        msg = "Cannot add phase-space positions with different times"
        raise ValueError(msg)

    # Add the fields
    return replace(wt1, q=wt1.q + wt2.q, p=wt1.p + wt2.p)
