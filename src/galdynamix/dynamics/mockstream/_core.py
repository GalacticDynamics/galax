"""galdynamix: Galactic Dynamix in Jax."""

__all__ = ["MockStream"]

import equinox as eqx
import jax.numpy as xp

from galdynamix.dynamics._core import AbstractPhaseSpacePositionBase, converter_batchvec
from galdynamix.typing import BatchFloatScalar, BatchVec7
from galdynamix.utils import partial_jit
from galdynamix.utils._shape import atleast_batched, batched_shape


class MockStream(AbstractPhaseSpacePositionBase):
    """Mock stream object.

    Todo:
    ----
    - units stuff
    - change this to be a collection of sub-objects: progenitor, leading arm,
      trailing arm, 3-body ejecta, etc.
    - GR 4-vector stuff
    """

    release_time: BatchFloatScalar = eqx.field(converter=converter_batchvec)
    """Release time of the stream particles [Myr]."""

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        """Batch ."""
        qbatch, qshape = batched_shape(self.q, expect_scalar=False)
        pbatch, pshape = batched_shape(self.p, expect_scalar=False)
        tbatch, tshape = batched_shape(self.release_time, expect_scalar=True)
        batch_shape = xp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, (qshape, pshape, tshape)

    @property
    @partial_jit()
    def w(self) -> BatchVec7:
        """Return as a single Array[(*batch, Q + P + T),]."""
        batch_shape, component_shapes = self._shape_tuple
        q = xp.broadcast_to(self.q, batch_shape + component_shapes[0:1])
        p = xp.broadcast_to(self.p, batch_shape + component_shapes[1:2])
        t = xp.broadcast_to(
            atleast_batched(self.release_time), batch_shape + component_shapes[2:3]
        )
        return xp.concatenate((q, p, t), axis=-1)
