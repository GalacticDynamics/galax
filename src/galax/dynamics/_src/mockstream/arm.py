"""Mock stellar stream arm."""

__all__ = ["MockStreamArm"]

from typing import Any, ClassVar, Protocol, cast, final, runtime_checkable

import diffrax as dfx
import equinox as eqx
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity

import galax._custom_types as gt
import galax.coordinates as gc
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


@gc.AbstractPhaseSpaceObject.from_.dispatch  # type: ignore[attr-defined,misc]
def from_(
    cls: type[MockStreamArm],
    soln: dfx.Solution,
    /,
    *,
    release_time: gt.BBtQuSz0,
    frame: cx.frames.AbstractReferenceFrame,
    units: u.AbstractUnitSystem,  # not dispatched on, but required
    unbatch_time: bool = True,
) -> MockStreamArm:
    """Create a new instance of the class."""
    # Reshape (*tbatch, T, *ybatch) to (*tbatch, *ybatch, T)
    t = soln.ts  # already in the shape (*tbatch, T)
    n_tbatch = soln.t0.ndim
    q = jnp.moveaxis(soln.ys[0], n_tbatch, -2)
    p = jnp.moveaxis(soln.ys[1], n_tbatch, -2)

    # Reshape (*tbatch, *ybatch, T) to (*tbatch, *ybatch) if T == 1
    if unbatch_time and t.shape[-1] == 1:
        t = t[..., -1]
        q = q[..., -1, :]
        p = p[..., -1, :]

    # Convert the solution to a phase-space position
    return cls(
        q=cx.CartesianPos3D.from_(q, units["length"]),
        p=cx.CartesianVel3D.from_(p, units["speed"]),
        t=BareQuantity(t, units["time"]),
        release_time=release_time,
        frame=frame,
    )


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
