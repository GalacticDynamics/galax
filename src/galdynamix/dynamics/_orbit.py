"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = ["Orbit"]


import equinox as eqx
import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.utils._jax import partial_jit


class Orbit(eqx.Module):  # type: ignore[misc]
    """Orbit.

    Todo:
    ----
    - Units stuff
    - GR stuff
    """

    q: jt.Array
    """Position of the stream particles (x, y, z) [kpc]."""

    p: jt.Array
    """Position of the stream particles (x, y, z) [kpc/Myr]."""

    t: jt.Array
    """Array of times [Myr]."""

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    @partial_jit()
    def to_w(self) -> jt.Array:
        """Return as a single Array[(N, Q + P + T),]."""
        # Reshape t to (N, 1) if necessary
        t = self.t[:, None] if self.t.ndim == 1 else self.t
        # Determine output shape
        shape = (self.q.shape[0], self.q.shape[1] + self.p.shape[1] + t.shape[1])
        # Create output array (jax will fuse these ops)
        out = xp.empty(shape)
        out = out.at[:, : self.q.shape[1]].set(self.q)
        out = out.at[:, self.q.shape[1] : -t.shape[1]].set(self.p)
        out = out.at[:, -t.shape[1] :].set(t)
        return out  # noqa: RET504
