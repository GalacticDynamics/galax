"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractPhaseSpacePositionBase"]

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.experimental.array_api as xp
import jax.numpy as jnp
from jaxtyping import Array, Float

from galax.typing import BatchFloatScalar, BatchVec3, BatchVec6
from galax.units import UnitSystem

if TYPE_CHECKING:
    from typing import Self


class AbstractPhaseSpacePositionBase(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class for all the types of phase-space positions.

    See Also
    --------
    :class:`~galax.coordinates.AbstractPhaseSpacePosition`
    :class:`~galax.coordinates.AbstractPhaseSpaceTimePosition`
    """

    q: eqx.AbstractVar[Float[Array, "*#batch #time 3"]]
    """Positions."""

    p: eqx.AbstractVar[Float[Array, "*#batch #time 3"]]
    """Conjugate momenta at positions ``q``."""

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

    @property
    def ndim(self) -> int:
        """Number of dimensions, not including component shape."""
        return len(self.shape)

    def __len__(self) -> int:
        """Return the number of particles."""
        return self.shape[0]

    @abstractmethod
    def __getitem__(self, index: Any) -> "Self":
        ...

    # ==========================================================================

    @property
    def full_shape(self) -> tuple[int, ...]:
        """Shape of the position and velocity arrays."""
        batch_shape, component_shapes = self._shape_tuple
        return (*batch_shape, sum(component_shapes))

    # ==========================================================================
    # Convenience methods

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

    # ==========================================================================
    # Dynamical quantities

    # TODO: property?
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

    # TODO: property?
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
            >>> from galax.coordinates import PhaseSpacePosition

        We can compute the angular momentum of a single object

            >>> pos = np.array([1., 0, 0]) * u.au
            >>> vel = np.array([0, 2*np.pi, 0]) * u.au/u.yr
            >>> w = PhaseSpacePosition(pos, vel)
            >>> w.angular_momentum()
            Array([0.        , 0.        , 6.28318531], dtype=float64)
        """
        # TODO: when q, p are not Cartesian.
        return jnp.cross(self.q, self.p)
