"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = ["Orbit"]


from galdynamix.potential._potential.base import AbstractPotentialBase

from ._core import PhaseSpacePosition


class Orbit(PhaseSpacePosition):
    """Orbit."""

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""
