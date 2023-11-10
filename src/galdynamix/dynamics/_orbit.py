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

    @property
    @partial_jit()
    def qp(self) -> jt.Array:
        """Return as a single Array[(N, Q + P + T),]."""
        # Determine output shape
        qd = self.q.shape[1]  # dimensionality of q
        shape = (self.q.shape[0], qd + self.p.shape[1])
        # Create output array (jax will fuse these ops)
        out = xp.empty(shape)
        out = out.at[:, :qd].set(self.q)
        out = out.at[:, qd:].set(self.p)
        return out  # noqa: RET504

    @property
    @partial_jit()
    def w(self) -> jt.Array:
        """Return as a single Array[(N, Q + P + T),]."""
        qp = self.qp
        qpd = qp.shape[1]  # dimensionality of qp
        # Reshape t to (N, 1) if necessary
        t = self.t[:, None] if self.t.ndim == 1 else self.t
        # Determine output shape
        shape = (qp.shape[0], qpd + t.shape[1])
        # Create output array (jax will fuse these ops)
        out = xp.empty(shape)
        out = out.at[:, :qpd].set(qp)
        out = out.at[:, qpd:].set(t)
        return out  # noqa: RET504
