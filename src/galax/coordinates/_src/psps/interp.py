"""galax: Galactic Dynamics in Jax."""

__all__ = ["InterpolatedPhaseSpacePosition", "PhaseSpacePositionInterpolant"]

from dataclasses import KW_ONLY
from typing import Protocol, final, runtime_checkable

import equinox as eqx
import jax.numpy as jnp

import coordinax as cx
import unxt as u

import galax.typing as gt
from .base import ComponentShapeTuple
from .base_psp import AbstractPhaseSpacePosition
from .core import PhaseSpacePosition
from galax.utils._shape import batched_shape, vector_batched_shape


@runtime_checkable
class PhaseSpacePositionInterpolant(Protocol):
    """Protocol for interpolating phase-space positions."""

    units: u.AbstractUnitSystem
    """The unit system for the interpolation."""

    def __call__(self, t: gt.QVecTime) -> PhaseSpacePosition:
        """Evaluate the interpolation.

        Parameters
        ----------
        t : Quantity[float, (time,), 'time']
            The times at which to evaluate the interpolation.

        Returns
        -------
        :class:`galax.coordinates.PhaseSpacePosition`
            The interpolated phase-space positions.
        """
        ...


@final
class InterpolatedPhaseSpacePosition(AbstractPhaseSpacePosition):
    """Interpolated phase-space position."""

    q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
    """Positions, e.g CartesianPos3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
    r"""Conjugate momenta, e.g. CartesianVel3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: gt.BatchableFloatQScalar | gt.QVec1 = eqx.field(
        converter=u.Quantity["time"].from_
    )
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

    interpolant: PhaseSpacePositionInterpolant
    """The interpolation function."""

    _: KW_ONLY

    frame: cx.frames.NoFrame  # TODO: support frames
    """The reference frame of the phase-space position."""

    def __call__(self, t: gt.BatchFloatQScalar) -> PhaseSpacePosition:
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
