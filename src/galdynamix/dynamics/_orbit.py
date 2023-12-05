"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = ["Orbit"]

import jax.typing as jt

from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.utils._jax import partial_jit

from ._core import PhaseSpacePosition


class Orbit(PhaseSpacePosition):
    """Represents an orbit.

    Represents an orbit: positions and velocities (conjugate momenta) as a
    function of time.

    """

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    # ==========================================================================
    # Dynamical quantities

    @partial_jit()
    def potential_energy(
        self, potential: AbstractPotentialBase | None = None, /
    ) -> jt.Array:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galdynamix.potential.AbstractPotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : :class:`~jax.Array`
            The specific potential energy.
        """
        if potential is None:
            return self.potential.potential_energy(self, self.t)
        return potential.potential_energy(self, self.t)

    @partial_jit()
    def energy(self, potential: AbstractPotentialBase | None = None, /) -> jt.Array:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Returns
        -------
        E : :class:`~astropy.units.Quantity`
            The kinetic energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential)
