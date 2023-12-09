"""galdynamix: Galactic Dynamix in Jax."""

__all__ = ["AbstractPhaseSpacePosition", "PhaseSpacePosition"]

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as xp

from galdynamix.typing import BatchFloatScalar, BatchVec3, BatchVec6, BatchVec7
from galdynamix.utils import partial_jit
from galdynamix.utils._shape import atleast_batched, batched_shape
from galdynamix.utils.dataclasses import converter_float_array

if TYPE_CHECKING:
    from galdynamix.potential._potential.base import AbstractPotentialBase


class AbstractPhaseSpacePositionBase(eqx.Module):  # type: ignore[misc]
    """Abstract Base Class of Phase-Space Positions.

    Todo:
    ----
    - Units stuff
    - GR stuff (note that then this will include time and can be merged with
      ``AbstractPhaseSpacePosition``)
    """

    q: BatchVec3 = eqx.field(converter=converter_float_array)
    """Positions (x, y, z)."""

    p: BatchVec3 = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta (v_x, v_y, v_z)."""

    @property
    @abstractmethod
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        """Batch, component shape."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the position and velocity arrays."""
        batch_shape, component_shapes = self._shape_tuple
        return (*batch_shape, sum(component_shapes))

    # ==========================================================================
    # Convenience properties

    @property
    @partial_jit()
    def qp(self) -> BatchVec6:
        """Return as a single Array[(*batch, Q + P),]."""
        batch_shape, component_shapes = self._shape_tuple
        q = xp.broadcast_to(self.q, batch_shape + component_shapes[0:1])
        p = xp.broadcast_to(self.p, batch_shape + component_shapes[1:2])
        return xp.concatenate((q, p), axis=-1)

    # ==========================================================================
    # Dynamical quantities

    @partial_jit()
    def kinetic_energy(self) -> BatchFloatScalar:
        r"""Return the specific kinetic energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.
        """
        # TODO: use a ``norm`` function so that this works for non-Cartesian.
        return 0.5 * xp.sum(self.p**2, axis=-1)


class AbstractPhaseSpacePosition(AbstractPhaseSpacePositionBase):
    """Abstract Base Class of Phase-Space Positions."""

    t: BatchFloatScalar = eqx.field(converter=converter_float_array)
    """Array of times."""

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        """Batch ."""
        qbatch, qshape = batched_shape(self.q, expect_ndim=1)
        pbatch, pshape = batched_shape(self.p, expect_ndim=1)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        batch_shape: tuple[int, ...] = xp.broadcast_shapes(qbatch, pbatch, tbatch)
        array_shape: tuple[int, int, int] = qshape + pshape + (1,)
        return batch_shape, array_shape

    # ==========================================================================
    # Convenience properties

    @property
    @partial_jit()
    def w(self) -> BatchVec7:
        """Return as a single Array[(*batch, Q + P + T),]."""
        batch_shape, component_shapes = self._shape_tuple
        q = xp.broadcast_to(self.q, batch_shape + component_shapes[0:1])
        p = xp.broadcast_to(self.p, batch_shape + component_shapes[1:2])
        t = xp.broadcast_to(
            atleast_batched(self.t), batch_shape + component_shapes[2:3]
        )
        return xp.concatenate((q, p, t), axis=-1)

    @partial_jit()
    def angular_momentum(self) -> BatchVec3:
        r"""Compute the angular momentum.

        .. math::

            \boldsymbol{{L}} = \boldsymbol{{q}} \times \boldsymbol{{p}}

        See :ref:`shape-conventions` for more information about the shapes of
        input and output objects.

        Returns
        -------
        L : Array[float, (*batch,3)]
            Array of angular momentum vectors.

        Examples
        --------
            >>> import numpy as np
            >>> import astropy.units as u
            >>> pos = np.array([1., 0, 0]) * u.au
            >>> vel = np.array([0, 2*np.pi, 0]) * u.au/u.yr
            >>> w = PhaseSpacePosition(pos, vel)
            >>> w.angular_momentum() # doctest: +FLOAT_CMP
            <Quantity [0.        ,0.        ,6.28318531] AU2 / yr>
        """
        # TODO: when q, p are not Cartesian.
        return xp.cross(self.q, self.p)

    # ==========================================================================
    # Dynamical quantities

    @partial_jit()
    def potential_energy(
        self, potential: "AbstractPotentialBase", /
    ) -> BatchFloatScalar:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galdynamix.potential.AbstractPotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.
        """
        return potential.potential_energy(self.q, t=self.t)

    @partial_jit()
    def energy(self, potential: "AbstractPotentialBase", /) -> BatchFloatScalar:
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


class PhaseSpacePosition(AbstractPhaseSpacePosition):
    pass
