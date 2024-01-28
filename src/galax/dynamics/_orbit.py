"""galax: Galactic Dynamix in Jax."""

__all__ = ["Orbit"]

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import BatchFloatScalar, BroadBatchVec3, VecTime
from galax.utils._shape import batched_shape
from galax.utils.dataclasses import converter_float_array

from ._base import AbstractPhaseSpacePosition


class Orbit(AbstractPhaseSpacePosition):
    """Represents an orbit.

    Represents an orbit: positions and velocities (conjugate momenta) as a
    function of time.

    """

    q: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    """Positions (x, y, z)."""

    p: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta ($v_x$, $v_y$, $v_z$)."""

    # TODO: consider how this should be vectorized
    t: VecTime = eqx.field(converter=converter_float_array)
    """Array of times corresponding to the positions."""

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""

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
        self, potential: AbstractPotentialBase | None = None, /
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
        if potential is None:
            return self.potential.potential_energy(self.q, t=self.t)
        return potential.potential_energy(self.q, t=self.t)

    @partial(jax.jit)
    def energy(
        self, potential: "AbstractPotentialBase | None" = None, /
    ) -> BatchFloatScalar:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential)
