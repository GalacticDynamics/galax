"""galdynamix: Galactic Dynamix in Jax"""

from __future__ import annotations

__all__ = ["Orbit"]


import equinox as eqx
import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import AbstractPotentialBase


class Orbit(eqx.Module):  # type: ignore[misc]
    """Orbit.

    TODO:
    - Units stuff
    - GR stuff
    """

    q: jt.Array
    """Position of the stream particles (x, y, z) [kpc]."""

    p: jt.Array
    """Position of the stream particles (x, y, z) [kpc/Myr]."""

    t: jt.Array
    """Release time of the stream particles [Myr]."""

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    def to_w(self) -> jt.Array:
        t = self.t[:, None] if self.t.ndim == 1 else self.t
        out = xp.empty(
            (
                self.q.shape[0],
                self.q.shape[1] + self.p.shape[1] + t.shape[1],
            )
        )
        out = out.at[:, : self.q.shape[1]].set(self.q)
        out = out.at[:, self.q.shape[1] : -1].set(self.p)
        out = out.at[:, -1:].set(t)
        return out  # noqa: RET504
