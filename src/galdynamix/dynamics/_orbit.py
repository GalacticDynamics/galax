"""galdynamix: Galactic Dynamix in Jax."""

__all__ = ["Orbit"]

from typing_extensions import override

from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.typing import BatchFloatScalar
from galdynamix.utils._jax import partial_jit

from ._core import AbstractPhaseSpacePosition


class Orbit(AbstractPhaseSpacePosition):
    """Represents an orbit.

    Represents an orbit: positions and velocities (conjugate momenta) as a
    function of time.

    """

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
        potential : `galdynamix.potential.AbstractPotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.
        """
        if potential is None:
            return self.potential.potential_energy(self.q, t=self.t)
        return potential.potential_energy(self.q, t=self.t)
