"""galax: Galactic Dynamix in Jax."""

__all__ = ["MockStream"]

from functools import partial
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from galax.dynamics._base import AbstractPhaseSpacePosition
from galax.typing import BatchFloatScalar, VecTime
from galax.utils._shape import batched_shape
from galax.utils.dataclasses import converter_float_array

if TYPE_CHECKING:
    from galax.potential._potential.base import AbstractPotentialBase


class MockStream(AbstractPhaseSpacePosition):
    """Mock stream object.

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

    Todo:
    ----
    - units stuff
    - change this to be a collection of sub-objects: progenitor, leading arm,
      trailing arm, 3-body ejecta, etc.
    - GR 4-vector stuff
    """

    q: Float[Array, "*batch time 3"] = eqx.field(converter=converter_float_array)
    """Positions (x, y, z)."""

    p: Float[Array, "*batch time 3"] = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta (v_x, v_y, v_z)."""

    t: VecTime = eqx.field(converter=converter_float_array)
    """Array of times corresponding to the positions."""

    release_time: VecTime = eqx.field(converter=converter_float_array)
    """Release time of the stream particles [Myr]."""

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        """Batch ."""
        qbatch, qshape = batched_shape(self.q, expect_ndim=1)
        pbatch, pshape = batched_shape(self.p, expect_ndim=1)
        tbatch, _ = batched_shape(self.t, expect_ndim=0)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, qshape + pshape + (1,)

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
        t : float
            Time at which to compute the potential energy.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.
        """
        return potential.potential_energy(self.q, t=self.t)
