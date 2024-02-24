"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractPhaseSpacePosition", "PhaseSpacePosition"]

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from vector import Abstract3DVector, Abstract3DVectorDifferential

from .base import AbstractPhaseSpacePositionBase
from galax.typing import BatchFloatScalar, BroadBatchVec3, FloatScalar
from galax.utils._shape import batched_shape
from galax.utils.dataclasses import converter_float_array

if TYPE_CHECKING:
    from typing import Self

    from galax.potential._potential.base import AbstractPotentialBase


class AbstractPhaseSpacePosition(AbstractPhaseSpacePositionBase):
    r"""Abstract base class of Phase-Space Positions.

    The phase-space position is a point in the 6-dimensional phase space
    :math:`\mathbb{R}^6` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}` and the conjugate momentum :math:`\boldsymbol{p}`.
    """

    # TODO: hint shape Float[Array, "*#batch #time 3"]
    q: eqx.AbstractVar[Abstract3DVector]
    """Positions."""

    p: eqx.AbstractVar[Abstract3DVectorDifferential]
    """Conjugate momenta at positions ``q``."""

    # ==========================================================================
    # Array properties

    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied."""
        # TODO: make sure the slice is only on the batch, not the component.
        return replace(self, q=self.q[index], p=self.p[index])

    # ==========================================================================
    # Dynamical quantities

    def potential_energy(
        self, potential: "AbstractPotentialBase", /, t: FloatScalar
    ) -> BatchFloatScalar:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : :class:`~galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.
        t : float
            The time at which to compute the potential energy at the given
            positions.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.
        """
        return potential.potential_energy(self.q, t=t)

    @partial(jax.jit)
    def energy(
        self, potential: "AbstractPotentialBase", /, t: FloatScalar
    ) -> BatchFloatScalar:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2 E_\Phi =
            \Phi(\boldsymbol{q}) E = E_K + E_\Phi

        Parameters
        ----------
        potential : :class:`~galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.
        t : float
            The time at which to compute the potential energy at the given
            positions.

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential, t=t)


##############################################################################


@final
class PhaseSpacePosition(AbstractPhaseSpacePosition):
    r"""Represents a phase-space position.

    The phase-space position is a point in the 6-dimensional phase space
    :math:`\\mathbb{R}^6` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}` and the conjugate momentum :math:`\boldsymbol{p}`.

    See Also
    --------
    :class:`~galax.coordinates.PhaseSpaceTimePosition`
        A phase-space position with time.
    """

    q: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    """Positions (x, y, z).

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta (v_x, v_y, v_z).

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int]]:
        """Batch, component shape."""
        qbatch, qshape = batched_shape(self.q, expect_ndim=1)
        pbatch, pshape = batched_shape(self.p, expect_ndim=1)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch)
        return batch_shape, qshape + pshape
