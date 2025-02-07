"""Register primitives for PSPs."""

__all__: list[str] = []

from dataclasses import replace

import jax
from quax import register

import quaxed.numpy as jnp

from .base import AbstractPhaseSpacePosition


@register(jax.lax.add_p)  # type: ignore[misc]
def add_psps(
    psp1: AbstractPhaseSpacePosition, psp2: AbstractPhaseSpacePosition, /
) -> AbstractPhaseSpacePosition:
    """Add two phase-space positions.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc

    >>> w1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                            p=u.Quantity([4, 5, 6], "km/s"),
    ...                            t=u.Quantity(0, "Gyr"))
    >>> w2 = gc.PhaseSpacePosition(q=u.Quantity([-1, -2, -3], "kpc"),
    ...                            p=u.Quantity([-4, -5, -6], "km/s"),
    ...                            t=u.Quantity(0, "Gyr"))
    >>> w3 = w1 + w2
    >>> w3
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
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
    if not isinstance(psp2, type(psp1)):
        msg = f"Cannot add {type(psp1)} and {type(psp2)}"
        raise TypeError(msg)

    # Check the times are the same
    if not jnp.all(psp1.t == psp2.t):
        msg = "Cannot add phase-space positions with different times"
        raise ValueError(msg)

    # Add the fields
    return replace(psp1, q=psp1.q + psp2.q, p=psp1.p + psp2.p)
