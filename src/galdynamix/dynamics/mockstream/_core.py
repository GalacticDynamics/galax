"""galdynamix: Galactic Dynamix in Jax."""

__all__ = ["MockStream"]


import equinox as eqx
import jax.numpy as xp

from galdynamix.typing import VectorN, VectorN3, VectorN6, VectorN7
from galdynamix.utils._jax import partial_jit


# TODO: make a subclass of AbstractPhaseSpacePosition?
class MockStream(eqx.Module):  # type: ignore[misc]
    """Mock stream object.

    Todo:
    ----
    - units stuff
    - change this to be a collection of sub-objects: progenitor, leading arm,
      trailing arm, 3-body ejecta, etc.
    - GR 4-vector stuff
    """

    q: VectorN3
    """Position of the stream particles (x, y, z) [kpc]."""

    p: VectorN3
    """Position of the stream particles (x, y, z) [kpc/Myr]."""

    release_time: VectorN
    """Release time of the stream particles [Myr]."""

    @property
    @partial_jit()
    def qp(self) -> VectorN6:
        """Return as a single Array[(N, Q + P),]."""
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
    def w(self) -> VectorN7:
        """Return as a single Array[(N, Q + P + T),]."""
        qp = self.qp
        qpd = qp.shape[1]  # dimensionality of qp
        # Reshape t to (N, 1) if necessary
        t = (
            self.release_time[:, None]
            if self.release_time.ndim == 1
            else self.release_time
        )
        # Determine output shape
        shape = (qp.shape[0], qpd + t.shape[1])
        # Create output array (jax will fuse these ops)
        out = xp.empty(shape)
        out = out.at[:, :qpd].set(qp)
        out = out.at[:, qpd:].set(t)
        return out  # noqa: RET504
