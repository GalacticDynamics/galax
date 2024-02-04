"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractPhaseSpaceTimePosition", "PhaseSpaceTimePosition"]

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, final

import equinox as eqx
import jax
import jax.experimental.array_api as xp
import jax.numpy as jnp
from jaxtyping import Array, Float

from galax.typing import (
    BatchFloatScalar,
    BatchVec7,
    BroadBatchFloatScalar,
    BroadBatchVec3,
    Vec1,
)
from galax.units import UnitSystem
from galax.utils._shape import atleast_batched, batched_shape, expand_batch_dims
from galax.utils.dataclasses import converter_float_array

from ._base import AbstractPhaseSpacePositionBase
from ._utils import getitem_broadscalartime_index

if TYPE_CHECKING:
    from galax.potential._potential.base import AbstractPotentialBase


class AbstractPhaseSpaceTimePosition(AbstractPhaseSpacePositionBase):
    r"""Abstract base class of Phase-Space Positions with time.

    The phase-space position is a point in the 7-dimensional phase space
    :math:`\mathbb{R}^7` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}`, the conjugate momentum :math:`\boldsymbol{p}`, and
    the time :math:`t`.

    See Also
    --------
    :class:`~galax.coordinates.PhaseSpacePosition`
        A phase-space position without time.
    """

    q: eqx.AbstractVar[Float[Array, "*#batch #time 3"]]
    """Positions."""

    p: eqx.AbstractVar[Float[Array, "*#batch #time 3"]]
    """Conjugate momenta at positions ``q``."""

    t: eqx.AbstractVar[Float[Array, "*#batch #time"]]
    """Time corresponding to the positions and momenta."""

    # ==========================================================================
    # Array methods

    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied."""
        # Compute subindex
        subindex = getitem_broadscalartime_index(index, self.t)
        # Apply slice
        return replace(self, q=self.q[index], p=self.p[index], t=self.t[subindex])

    # ==========================================================================
    # Convenience methods

    def wt(self, *, units: UnitSystem | None = None) -> BatchVec7:
        """Phase-space position as an Array[float, (*batch, Q + P + 1)].

        This is the full phase-space position, including the time.

        Parameters
        ----------
        units : `galax.units.UnitSystem`, optional keyword-only
            The unit system If ``None``, use the current unit system.

        Returns
        -------
        wt : Array[float, (*batch, Q + P + 1)]
            The full phase-space position, including time.
        """
        if units is not None:
            msg = "units not yet implemented."
            raise NotImplementedError(msg)

        batch_shape, comp_shapes = self._shape_tuple
        q = xp.broadcast_to(self.q, batch_shape + comp_shapes[0:1])
        p = xp.broadcast_to(self.p, batch_shape + comp_shapes[1:2])
        t = xp.broadcast_to(atleast_batched(self.t), batch_shape + comp_shapes[2:3])
        return xp.concat((q, p, t), axis=-1)

    # ==========================================================================
    # Dynamical quantities

    def potential_energy(self, potential: "AbstractPotentialBase") -> BatchFloatScalar:
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

    @partial(jax.jit)
    def energy(self, potential: "AbstractPotentialBase") -> BatchFloatScalar:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Parameters
        ----------
        potential : `galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential)


###############################################################################


@final
class PhaseSpaceTimePosition(AbstractPhaseSpaceTimePosition):
    q: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    """Positions (x, y, z).

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta (v_x, v_y, v_z).

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    t: BroadBatchFloatScalar | Vec1 = eqx.field(converter=converter_float_array)
    """The time corresponding to the positions.

    This is a scalar with the same batch shape as the positions and velocities.
    The default value is a scalar zero. If `t` is a scalar it will be broadcast
    to the same batch shape as `q` and `p`.
    """

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct. Can be Vec0.
        if self.t.ndim in (0, 1):
            t = expand_batch_dims(self.t, ndim=self.q.ndim - self.t.ndim - 1)
            object.__setattr__(self, "t", t)

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        """Batch, component shape."""
        qbatch, qshape = batched_shape(self.q, expect_ndim=1)
        pbatch, pshape = batched_shape(self.p, expect_ndim=1)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        array_shape = qshape + pshape + (1,)
        return batch_shape, array_shape

    # ==========================================================================
    # Convenience methods

    def wt(self, *, units: UnitSystem | None = None) -> BatchVec7:
        """Phase-space position as an Array[float, (*batch, Q + P + 1)].

        This is the full phase-space position, including the time.

        Parameters
        ----------
        units : `galax.units.UnitSystem`, optional keyword-only
            The unit system If ``None``, use the current unit system.

        Returns
        -------
        wt : Array[float, (*batch, Q + P + 1)]
            The full phase-space position, including time.
        """
        if units is not None:
            msg = "units not yet implemented."
            raise NotImplementedError(msg)

        batch_shape, comp_shapes = self._shape_tuple
        q = xp.broadcast_to(self.q, batch_shape + comp_shapes[0:1])
        p = xp.broadcast_to(self.p, batch_shape + comp_shapes[1:2])
        return xp.concat((q, p, self.t[..., None]), axis=-1)
