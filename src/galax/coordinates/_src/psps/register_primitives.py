"""Register primitives for PSPs."""

__all__: list[str] = []

from dataclasses import replace

import jax
from quax import register

from .core import PhaseSpacePosition


@register(jax.lax.add_p)  # type: ignore[misc]
def add_psps(
    psp1: PhaseSpacePosition, psp2: PhaseSpacePosition, /
) -> PhaseSpacePosition:
    """Add two phase-space positions.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc

    >>> w1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                            p=u.Quantity([4, 5, 6], "km/s"))
    >>> w2 = gc.PhaseSpacePosition(q=u.Quantity([-1, -2, -3], "kpc"),
    ...                            p=u.Quantity([-4, -5, -6], "km/s"))
    >>> w3 = w1 + w2
    >>> w3
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      frame=SimulationFrame()
    )

    >>> w3.q.x.value
    Array(0, dtype=int64)

    """
    return replace(psp1, q=psp1.q + psp2.q, p=psp1.p + psp2.p)
