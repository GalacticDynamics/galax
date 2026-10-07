"""Mock stellar stream arm."""

__all__ = ["MockStreamArm"]

from typing import ClassVar, final

import equinox as eqx

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics.custom_types as gt
from galax.coordinates._src.shape import batched_shape, vector_batched_shape


@final
class MockStreamArm(gc.AbstractBasicPhaseSpaceCoordinate):
    """Component of a mock stream object.

    Parameters
    ----------
    q : Array[float, (*batch, 3)]
        Positions (x, y, z).
    p : Array[float, (*batch, 3)]
        Conjugate momenta (v_x, v_y, v_z).
    t : Array[float, (*batch,)]
        Array of times corresponding to the positions.
    release_time : Array[float, (*batch,)]
        Release time of the stream particles [Myr].
    frame : AbstractReferenceFrame

    """

    q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
    """Positions (x, y, z)."""

    p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
    r"""Conjugate momenta (v_x, v_y, v_z)."""

    t: gt.QuSzTime = eqx.field(converter=u.Quantity["time"].from_)
    """Array of times corresponding to the positions."""

    release_time: gt.QuSzTime = eqx.field(converter=u.Quantity["time"].from_)
    """Release time of the stream particles [Myr]."""

    frame: gc.frames.SimulationFrame  # TODO: support frames
    """The reference frame of the phase-space position."""

    _GETITEM_DYNAMIC_FILTER_SPEC: ClassVar = (True, True, True, True, False)
    _GETITEM_TIME_FILTER_SPEC: ClassVar = (False, False, True, True, False)

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[gt.Shape, gc.ComponentShapeTuple]:
        """Batch ."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, gc.ComponentShapeTuple(q=qshape, p=pshape, t=1)
