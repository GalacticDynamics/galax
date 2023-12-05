"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = ["PhaseSpacePosition"]

from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax.numpy as xp
import jax.typing as jt

from galdynamix.utils._jax import partial_jit

if TYPE_CHECKING:
    from galdynamix.potential._potential.base import AbstractPotentialBase


class PhaseSpacePosition(eqx.Module):  # type: ignore[misc]
    """Orbit.

    Todo:
    ----
    - Units stuff
    - GR stuff
    """

    q: jt.Array
    """Position of the stream particles (x, y, z) [kpc]."""

    p: jt.Array
    """Position of the stream particles (x, y, z) [kpc/Myr]."""

    t: jt.Array
    """Array of times [Myr]."""

    @property
    @partial_jit()
    def qp(self) -> jt.Array:
        """Return as a single Array[(N, Q + P),]."""
        # Determine output shape
        qd = self.q.shape[1]  # dimensionality of q
        shape = (self.q.shape[0], qd + self.p.shape[1])
        # Create output array (jax will fuse these ops)
        out = xp.empty(shape)
        out = out.at[:, :qd].set(self.q)
        out = out.at[:, qd:].set(self.p)
        return out  # noqa: RET504

    @property
    @partial_jit()
    def w(self) -> jt.Array:
        """Return as a single Array[(N, Q + P + T),]."""
        qp = self.qp
        qpd = qp.shape[1]  # dimensionality of qp
        # Reshape t to (N, 1) if necessary
        t = self.t[:, None] if self.t.ndim == 1 else self.t
        # Determine output shape
        shape = (qp.shape[0], qpd + t.shape[1])
        # Create output array (jax will fuse these ops)
        out = xp.empty(shape)
        out = out.at[:, :qpd].set(qp)
        out = out.at[:, qpd:].set(t)
        return out  # noqa: RET504

    # ==========================================================================
    # Array stuff

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the position and velocity arrays."""
        return cast(
            tuple[int, ...],
            xp.broadcast_shapes(self.q.shape, self.p.shape, self.t.shape),
        )

    # ==========================================================================
    # Dynamical quantities

    @partial_jit()
    def kinetic_energy(self) -> jt.Array:
        r"""Return the specific kinetic energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2

        Returns
        -------
        E : :class:`~astropy.units.Quantity`
            The kinetic energy.
        """
        # TODO: use a ``norm`` function
        return 0.5 * xp.sum(self.p**2, axis=-1)

    @partial_jit()
    def potential_energy(self, potential: AbstractPotentialBase, /) -> jt.Array:
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
        return potential.potential_energy(self, self.t)

    @partial_jit()
    def energy(self, potential: AbstractPotentialBase, /) -> jt.Array:
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

    @partial_jit()
    def angular_momentum(self) -> jt.Array:
        r"""Compute the angular momentum.

        .. math::

            \boldsymbol{{L}} = \boldsymbol{{q}} \times \boldsymbol{{p}}

        See :ref:`shape-conventions` for more information about the shapes of
        input and output objects.

        Returns
        -------
        L : :class:`~astropy.units.Quantity`
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
