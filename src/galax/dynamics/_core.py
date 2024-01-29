"""galax: Galactic Dynamix in Jax."""

__all__ = ["PhaseSpacePosition"]

from functools import partial
from typing import TYPE_CHECKING, final

import equinox as eqx
import jax
import jax.numpy as jnp

from galax.typing import BatchFloatScalar, BroadBatchVec1, BroadBatchVec3
from galax.utils._shape import batched_shape, expand_batch_dims
from galax.utils.dataclasses import converter_float_array

from ._base import AbstractPhaseSpacePosition

if TYPE_CHECKING:
    from galax.potential._potential.base import AbstractPotentialBase


@final
class PhaseSpacePosition(AbstractPhaseSpacePosition):
    """Represents a phase-space position."""

    q: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    """Positions (x, y, z).

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta (v_x, v_y, v_z).

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: BroadBatchVec1 = eqx.field(default=(0.0,), converter=converter_float_array)
    """The time corresponding to the positions.

    This is a scalar with the same batch shape as the positions and velocities.
    The default value is a scalar zero.  `t` will be broadcast to the same batch
    shape as `q` and `p`.
    """

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct
        if self.t.ndim == 0:
            t = expand_batch_dims(self.t, ndim=self.q.ndim)
            object.__setattr__(self, "t", t)
        elif self.t.ndim == 1:
            t = expand_batch_dims(self.t, ndim=self.q.ndim - 1)
            object.__setattr__(self, "t", t)

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        """Batch, component shape."""
        qbatch, qshape = batched_shape(self.q, expect_ndim=1)
        pbatch, pshape = batched_shape(self.p, expect_ndim=1)
        tbatch, _ = batched_shape(self.t, expect_ndim=1)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        array_shape = qshape + pshape + (1,)
        return batch_shape, array_shape

    # ==========================================================================
    # Dynamical quantities

    @partial(jax.jit)
    def potential_energy(
        self, potential: "AbstractPotentialBase", /
    ) -> BatchFloatScalar:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.
        """
        return potential.potential_energy(self.q, t=self.t)
