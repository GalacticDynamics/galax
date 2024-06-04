"""galax: Galactic Dynamics in Jax."""

__all__ = ["InterpolatedPhaseSpacePosition", "PhaseSpacePositionInterpolant"]

from typing import Protocol, final, runtime_checkable

import equinox as eqx
import jax.numpy as jnp

from coordinax import AbstractPosition3D, AbstractVelocity3D
from unxt import AbstractUnitSystem, Quantity

import galax.typing as gt
from .base import ComponentShapeTuple
from .base_psp import AbstractPhaseSpacePosition
from .core import PhaseSpacePosition
from .utils import _p_converter, _q_converter
from galax.utils._shape import batched_shape, expand_batch_dims, vector_batched_shape


@runtime_checkable
class PhaseSpacePositionInterpolant(Protocol):
    """Protocol for interpolating phase-space positions."""

    units: AbstractUnitSystem
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

    q: AbstractPosition3D = eqx.field(converter=_q_converter)
    """Positions, e.g CartesianPosition3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: AbstractVelocity3D = eqx.field(converter=_p_converter)
    r"""Conjugate momenta, e.g. CartesianVelocity3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: gt.BroadBatchFloatQScalar | gt.QVec1 = eqx.field(
        converter=Quantity["time"].constructor
    )
    """The time corresponding to the positions.

    This is a Quantity with the same batch shape as the positions and
    velocities.  If `t` is a scalar it will be broadcast to the same batch shape
    as `q` and `p`.
    """

    interpolant: PhaseSpacePositionInterpolant
    """The interpolation function."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct. Can be Vec0.
        if self.t.ndim in (0, 1):
            t = expand_batch_dims(self.t, ndim=self.q.ndim - self.t.ndim)
            object.__setattr__(self, "t", t)

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
