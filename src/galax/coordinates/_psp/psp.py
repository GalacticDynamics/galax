"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractPhaseSpacePosition", "PhaseSpacePosition"]

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from plum import convert

from jax_quantity import Quantity
from vector import Abstract3DVector, Abstract3DVectorDifferential

from .base import AbstractPhaseSpacePositionBase, _p_converter, _q_converter
from galax.typing import (
    BatchableFloatOrIntScalarLike,
    BatchFloatOrIntQScalar,
)
from galax.utils._shape import vector_batched_shape

if TYPE_CHECKING:
    from typing import Self

    from galax.potential._potential.base import AbstractPotentialBase


class AbstractPhaseSpacePosition(AbstractPhaseSpacePositionBase):
    r"""Abstract base class of Phase-Space Positions.

    The phase-space position is a point in the 6-dimensional phase space
    :math:`\mathbb{R}^6` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}` and the conjugate momentum :math:`\boldsymbol{p}`.

    Parameters
    ----------
    q : :class:`~vector.Abstract3DVector`
        Positions.
    p : :class:`~vector.Abstract3DVectorDifferential`
        Conjugate momenta at positions ``q``.
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
        self,
        potential: "AbstractPotentialBase",
        /,
        t: BatchFloatOrIntQScalar | BatchableFloatOrIntScalarLike,
    ) -> Quantity["specific energy"]:  # TODO: shape hint
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : :class:`~galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.
        t : :class:`jax_quantity.Quantity[float, (*batch,), "time"]`
            The time at which to compute the potential energy at the given
            positions.

        Returns
        -------
        E : Quantity[float, (*batch,), "specific energy"]
            The specific potential energy.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp

        We can construct a phase-space position:

        >>> q = Cartesian3DVector(
        ...     x=Quantity(1, "kpc"),
        ...     y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=Quantity(2, "kpc"))
        >>> p = CartesianDifferential3D(
        ...     d_x=Quantity(0, "km/s"),
        ...     d_y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=Quantity(0, "km/s"))
        >>> w = gc.PhaseSpacePosition(q, p)

        We can compute the kinetic energy:

        >>> pot = gp.MilkyWayPotential()
        >>> w.potential_energy(pot, t=Quantity(0, "Gyr"))
        Quantity['specific energy'](Array(..., dtype=float64), unit='kpc2 / Myr2')
        """
        x = convert(self.q, Quantity).decompose(potential.units).value  # Cartesian
        return potential.potential_energy(x, t=t)

    @partial(jax.jit)
    def energy(
        self,
        potential: "AbstractPotentialBase",
        /,
        t: BatchFloatOrIntQScalar | BatchableFloatOrIntScalarLike,
    ) -> Quantity["specific energy"]:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2 E_\Phi =
            \Phi(\boldsymbol{q}) E = E_K + E_\Phi

        Parameters
        ----------
        potential : :class:`~galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.
        t : Quantity[float, (*batch,), "time"]
            The time at which to compute the potential energy at the given
            positions.

        Returns
        -------
        E : Quantity[float, (*batch,), "specific energy"]
            The kinetic energy.

        Examples
        --------
        We assume the following imports:

        >>> from jax_quantity import Quantity
        >>> from vector import Cartesian3DVector, CartesianDifferential3D
        >>> from galax.coordinates import PhaseSpacePosition
        >>> from galax.potential import MilkyWayPotential

        We can construct a phase-space position:

        >>> q = Cartesian3DVector(
        ...     x=Quantity(1, "kpc"),
        ...     y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "kpc"),
        ...     z=Quantity(2, "kpc"))
        >>> p = CartesianDifferential3D(
        ...     d_x=Quantity(0, "km/s"),
        ...     d_y=Quantity([[1.0, 2, 3, 4], [1.0, 2, 3, 4]], "km/s"),
        ...     d_z=Quantity(0, "km/s"))
        >>> w = PhaseSpacePosition(q, p)

        We can compute the kinetic energy:

        >>> pot = MilkyWayPotential()
        >>> w.energy(pot, t=Quantity(0, "Gyr"))
        Quantity['specific energy'](Array(..., dtype=float64), unit='km2 / s2')
        """
        return self.kinetic_energy() + self.potential_energy(potential, t=t)


##############################################################################


@final
class PhaseSpacePosition(AbstractPhaseSpacePosition):
    r"""Represents a phase-space position.

    The phase-space position is a point in the 6-dimensional phase space
    :math:`\\mathbb{R}^6` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}` and the conjugate momentum :math:`\boldsymbol{p}`.

    Parameters
    ----------
    q : :class:`~vector.Abstract3DVector`
        Positions.
    p : :class:`~vector.Abstract3DVectorDifferential`
        Conjugate momenta at positions ``q``.

    See Also
    --------
    :class:`~galax.coordinates.PhaseSpaceTimePosition`
        A phase-space position with time.

    Examples
    --------
    We assume the following imports:

    >>> from jax_quantity import Quantity
    >>> from vector import Cartesian3DVector, CartesianDifferential3D
    >>> from galax.coordinates import PhaseSpacePosition

    We can create a phase-space position:

    >>> q = Cartesian3DVector(x=Quantity(1, "m"), y=Quantity(2, "m"),
    ...                       z=Quantity(3, "m"))
    >>> p = CartesianDifferential3D(d_x=Quantity(4, "m/s"), d_y=Quantity(5, "m/s"),
    ...                             d_z=Quantity(6, "m/s"))

    >>> pos = PhaseSpacePosition(q=q, p=p)
    >>> pos
    PhaseSpacePosition(
      q=Cartesian3DVector(
        x=Quantity[PhysicalType('length')](value=f64[], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f64[], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f64[], unit=Unit("m"))
      ),
      p=CartesianDifferential3D(
        d_x=Quantity[PhysicalType({'speed', 'velocity'})](
          value=f64[], unit=Unit("m / s")
        ),
        d_y=Quantity[PhysicalType({'speed', 'velocity'})](
          value=f64[], unit=Unit("m / s")
        ),
        d_z=Quantity[PhysicalType({'speed', 'velocity'})](
          value=f64[], unit=Unit("m / s")
        )
      )
    )
    """

    q: Abstract3DVector = eqx.field(converter=_q_converter)
    """Positions (x, y, z).

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: Abstract3DVectorDifferential = eqx.field(converter=_p_converter)
    r"""Conjugate momenta (v_x, v_y, v_z).

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int]]:
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch)
        return batch_shape, qshape + pshape
