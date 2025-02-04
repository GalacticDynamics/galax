"""galax: Galactic Dynamics in Jax."""

__all__ = ["InterpolatedPhaseSpacePosition"]

from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax.numpy as jnp

import coordinax as cx
import unxt as u

import galax.coordinates as gc
import galax.typing as gt
from galax.coordinates._src.frames import SimulationFrame
from galax.utils._shape import batched_shape, vector_batched_shape


@final
class InterpolatedPhaseSpacePosition(gc.AbstractOnePhaseSpacePosition):
    """Interpolated phase-space position."""

    q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
    """Positions, e.g CartesianPos3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
    r"""Conjugate momenta, e.g. CartesianVel3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: gt.BBtFloatQuSz0 | gt.QuSz1 = eqx.field(converter=u.Quantity["time"].from_)
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

    interpolant: gc.PhaseSpacePositionInterpolant
    """The interpolation function."""

    _: KW_ONLY

    frame: SimulationFrame  # TODO: support frames
    """The reference frame of the phase-space position."""

    def __call__(self, t: gt.BtFloatQuSz0) -> gc.PhaseSpacePosition:
        """Call the interpolation."""
        return self.interpolant(t)

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[gt.Shape, gc.ComponentShapeTuple]:
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, gc.ComponentShapeTuple(q=qshape, p=pshape, t=1)
