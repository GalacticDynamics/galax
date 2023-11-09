"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = ["MockStream"]


import equinox as eqx
import jax.typing as jt


class MockStream(eqx.Module):  # type: ignore[misc]
    """Mock stream object.

    Todo:
    ----
    - units stuff
    - change this to be a collection of sub-objects: progenitor, leading arm,
      trailing arm, 3-body ejecta, etc.
    - GR 4-vector stuff
    """

    q: jt.Array
    """Position of the stream particles (x, y, z) [kpc]."""

    p: jt.Array
    """Position of the stream particles (x, y, z) [kpc/Myr]."""

    release_time: jt.Array
    """Release time of the stream particles [Myr]."""
