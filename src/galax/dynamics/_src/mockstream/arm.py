"""Mock stellar stream arm."""

__all__ = ["MockStreamArm"]

from typing import Any, ClassVar, Protocol, cast, final, runtime_checkable

import equinox as eqx
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.typing as gt
from galax.utils._shape import batched_shape, vector_batched_shape


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


#####################################################################

# =========================================================
# `__getitem__`


@runtime_checkable
class HasShape(Protocol):
    """Protocol for an object with a shape attribute."""

    shape: gt.Shape


@dispatch
def _psc_getitem_time_index(wt: MockStreamArm, index: Any) -> Any:
    """Get the time index from an index."""
    if isinstance(index, tuple):
        if len(index) == 0:  # slice is an empty tuple
            return slice(None)
        if wt.t.ndim == 1:  # slicing a Sz1
            return slice(None)
        if len(index) >= wt.t.ndim:
            msg = f"Index {index} has too many dimensions for time array of shape {wt.t.shape}"  # noqa: E501
            raise IndexError(msg)
    elif isinstance(index, HasShape):
        if wt.t.ndim == 1:  # Sz1
            return cast(HasShape, jnp.asarray([True]))
        if len(index.shape) >= wt.t.ndim:
            msg = f"Index {index} has too many dimensions for time array of shape {wt.t.shape}"  # noqa: E501
            raise IndexError(msg)
    return index
