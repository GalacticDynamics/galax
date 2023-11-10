"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = ["MockStream"]


import equinox as eqx
import jax.numpy as xp
import jax.typing as jt

from galdynamix.utils._jax import partial_jit


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

    @property
    @partial_jit()
    def w(self) -> jt.Array:
        """Return as a single Array[(N, Q + P + T),]."""
        # Reshape t to (N, 1) if necessary
        t = (
            self.release_time[:, None]
            if self.release_time.ndim == 1
            else self.release_time
        )
        # Determine output shape
        shape = (self.q.shape[0], self.q.shape[1] + self.p.shape[1] + t.shape[1])
        # Create output array (jax will fuse these ops)
        out = xp.empty(shape)
        out = out.at[:, : self.q.shape[1]].set(self.q)
        out = out.at[:, self.q.shape[1] : -t.shape[1]].set(self.p)
        out = out.at[:, -t.shape[1] :].set(t)
        return out  # noqa: RET504
