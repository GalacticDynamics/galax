"""galax: Galactic Dynamix in Jax."""

__all__ = ["Orbit"]

import equinox as eqx
from typing_extensions import override

from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import BatchFloatScalar, BatchVec3
from galax.utils._jax import partial_jit
from galax.utils.dataclasses import converter_float_array

from ._core import AbstractPhaseSpacePosition


class Orbit(AbstractPhaseSpacePosition):
    """Represents an orbit.

    Represents an orbit: positions and velocities (conjugate momenta) as a
    function of time.

    """

    q: BatchVec3 = eqx.field(converter=converter_float_array)
    """Positions (x, y, z)."""

    p: BatchVec3 = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta (v_x, v_y, v_z)."""

    t: BatchFloatScalar = eqx.field(converter=converter_float_array)
    """Array of times."""

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    # ==========================================================================
    # Dynamical quantities

    @override
    @partial_jit()
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
