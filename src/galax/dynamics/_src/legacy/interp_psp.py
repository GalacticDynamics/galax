"""galax: Galactic Dynamics in Jax."""

__all__ = ["InterpolatedPhaseSpaceCoordinate"]

from dataclasses import KW_ONLY
from typing import ClassVar, final

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity as FastQ

import galax.coordinates as gc
import galax.typing as gt
from galax.coordinates._src.frames import SimulationFrame
from galax.coordinates._src.pscs.base import ComponentShapeTuple
from galax.dynamics._src.orbit import PhaseSpaceInterpolation
from galax.utils._shape import batched_shape, vector_batched_shape


@final
class InterpolatedPhaseSpaceCoordinate(gc.AbstractBasicPhaseSpaceCoordinate):
    """Interpolated phase-space position."""

    q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
    """Positions, e.g CartesianPos3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
    r"""Conjugate momenta, e.g. CartesianVel3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: gt.BBtQuSz0 | gt.QuSz1 = eqx.field(converter=u.Quantity["time"].from_)
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

    interpolant: gc.PhaseSpaceObjectInterpolant
    """The interpolation function."""

    _: KW_ONLY

    frame: SimulationFrame  # TODO: support frames
    """The reference frame of the phase-space position."""

    _GETITEM_DYNAMIC_FILTER_SPEC: ClassVar = (True, True, True, False, False)
    _GETITEM_TIME_FILTER_SPEC: ClassVar = (False, False, True, False, False)

    def __call__(self, t: gt.BtFloatQuSz0) -> gc.PhaseSpaceCoordinate:
        """Call the interpolation."""
        return self.interpolant(t)

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, ComponentShapeTuple(q=qshape, p=pshape, t=1)


# TODO: support interpolation
@gc.PhaseSpaceCoordinate.from_.dispatch  # type: ignore[misc,attr-defined]
def from_(
    cls: type[InterpolatedPhaseSpaceCoordinate],
    soln: dfx.Solution,
    *,
    frame: cx.frames.AbstractReferenceFrame,  # not dispatched on, but required
    units: u.AbstractUnitSystem,  # not dispatched on, but required
    interpolant: PhaseSpaceInterpolation,  # not dispatched on, but required
    unbatch_time: bool = False,
) -> InterpolatedPhaseSpaceCoordinate:
    """Convert a solution to a phase-space position."""
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
        t=FastQ(soln.ts, units["time"]),
        frame=frame,
        interpolant=interpolant,
    )
