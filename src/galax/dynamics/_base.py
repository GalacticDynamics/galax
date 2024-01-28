"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractPhaseSpacePosition"]

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.experimental.array_api as xp
import jax.numpy as jnp
from jaxtyping import Array, Float

from galax.typing import BatchFloatScalar, BatchVec3, BatchVec6, BatchVec7
from galax.units import UnitSystem
from galax.utils._shape import atleast_batched

if TYPE_CHECKING:
    from galax.potential._potential.base import AbstractPotentialBase


class AbstractPhaseSpacePosition(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract Base Class of Phase-Space Positions.

    Todo:
    ----
    - Units stuff
    - GR stuff (note that then this will include time and can be merged with
      ``AbstractPhaseSpacePosition``)
    """

    q: eqx.AbstractVar[Float[Array, "*#batch #time 3"]]
    """Positions."""

    p: eqx.AbstractVar[Float[Array, "*#batch #time 3"]]
    """Conjugate momenta at positions ``q``."""

    t: eqx.AbstractVar[Float[Array, "*#batch #time"]]
    """Time corresponding to the positions and momenta."""

    # ==========================================================================
    # Array properties

    @property
    @abstractmethod
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Batch, component shape."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the position and velocity arrays."""
        return self._shape_tuple[0]

    def __len__(self) -> int:
        """Return the number of particles."""
        return self.shape[0]

    # ==========================================================================

    @property
    def full_shape(self) -> tuple[int, ...]:
        """Shape of the position and velocity arrays."""
        batch_shape, component_shapes = self._shape_tuple
        return (*batch_shape, sum(component_shapes))

    # ==========================================================================
    # Convenience properties

    def w(self, *, units: UnitSystem | None = None) -> BatchVec6:
        """Phase-space position as an Array[float, (*batch, Q + P)].

        This is the full phase-space position, not including the time.

        Parameters
        ----------
        units : `galax.units.UnitSystem`, optional keyword-only
            The unit system If ``None``, use the current unit system.

        Returns
        -------
        w : Array[float, (*batch, Q + P)]
            The phase-space position.
        """
        if units is not None:
            msg = "units not yet implemented."
            raise NotImplementedError(msg)

        batch_shape, component_shapes = self._shape_tuple
        q = xp.broadcast_to(self.q, batch_shape + component_shapes[0:1])
        p = xp.broadcast_to(self.p, batch_shape + component_shapes[1:2])
        return xp.concat((q, p), axis=-1)

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

    @partial(jax.jit)
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

    @abstractmethod
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
        ...

    @partial(jax.jit)
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

    @property
    @partial(jax.jit)
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
        We assume the following imports

            >>> import numpy as np
            >>> import astropy.units as u
            >>> from galax.dynamics import PhaseSpacePosition

        We can compute the angular momentum of a single object

            >>> pos = np.array([1., 0, 0]) * u.au
            >>> vel = np.array([0, 2*np.pi, 0]) * u.au/u.yr
            >>> w = PhaseSpacePosition(pos, vel)
            >>> w.angular_momentum
            Array([0.        , 0.        , 6.28318531], dtype=float64)
        """
        # TODO: when q, p are not Cartesian.
        return jnp.cross(self.q, self.p)
