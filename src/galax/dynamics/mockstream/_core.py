"""galax: Galactic Dynamix in Jax."""

__all__ = ["MockStream"]

import equinox as eqx
import jax.numpy as xp
from jaxtyping import Array, Float

from galax.dynamics._core import AbstractPhaseSpacePositionBase
from galax.typing import BatchVec7, TimeVector
from galax.utils import partial_jit
from galax.utils._shape import atleast_batched, batched_shape
from galax.utils.dataclasses import converter_float_array


class MockStream(AbstractPhaseSpacePositionBase):
    """Mock stream object.

    Todo:
    ----
    - units stuff
    - change this to be a collection of sub-objects: progenitor, leading arm,
      trailing arm, 3-body ejecta, etc.
    - GR 4-vector stuff
    """

    q: Float[Array, "*batch time 3"] = eqx.field(converter=converter_float_array)
    """Positions (x, y, z)."""

    p: Float[Array, "*batch time 3"] = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta (v_x, v_y, v_z)."""

    release_time: TimeVector = eqx.field(converter=converter_float_array)
    """Release time of the stream particles [Myr]."""

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        """Batch ."""
        qbatch, qshape = batched_shape(self.q, expect_ndim=1)
        pbatch, pshape = batched_shape(self.p, expect_ndim=1)
        tbatch, _ = batched_shape(self.release_time, expect_ndim=0)
        batch_shape = xp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, qshape + pshape + (1,)

    @property
    @partial_jit()
    def w(self) -> BatchVec7:
        """Return as a single Array[float, (*batch, Q + P + T)]."""
        batch_shape, component_shapes = self._shape_tuple
        q = xp.broadcast_to(self.q, batch_shape + component_shapes[0:1])
        p = xp.broadcast_to(self.p, batch_shape + component_shapes[1:2])
        t = xp.broadcast_to(
            atleast_batched(self.release_time), batch_shape + component_shapes[2:3]
        )
        return xp.concatenate((q, p, t), axis=-1)
