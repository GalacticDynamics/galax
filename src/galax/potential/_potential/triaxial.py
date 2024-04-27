"""Wrapper to add frame operations to a potential."""

__all__ = ["TriaxialInThePotential"]


from typing import cast, final

from coordinax.operators import simplify_op
from unxt import AbstractUnitSystem, Quantity

import galax.typing as gt
from galax.potential._potential.base import AbstractPotentialBase
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.utils import ImmutableDict


@final
class TriaxialInThePotential(AbstractPotentialBase):  # TODO: make generic wrt potential
    """Add triaxiality to the potential.

    .. warning::

        This is triaxiality in the potential, not the density!
        The density might be unphysical.

    Examples
    --------
    In this example, we create a triaxial Hernquist potential and apply a few
    coordinate transformations.

    First some imports:

    >>> from unxt import Quantity
    >>> import coordinax.operators as cxo
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp

    TODO

    """

    potential: AbstractPotentialBase  # TODO: make a type parameter
    """The potential to which to add triaxiality."""

    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=Quantity(1, ""),
        dimensions="dimensionless",
    )
    """Scale length in the y direction relative to the x direction.

    - If `q1 = 1`, the potential is spherically symmetric in the xy plane.
    - If `q1 < 1`, the potential is compressed in the y direction.
    - If `q1 > 1`, the potential is stretched in the y direction.
    """

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=Quantity(1, ""),
        dimensions="dimensionless",
    )
    """Scale length in the z direction relative to the x direction.

    - If `q2 = 1`, the potential is spherically symmetric in the xz plane.
    - If `q2 < 1`, the potential is compressed in the z direction.
    - If `q2 > 1`, the potential is stretched in the z direction.
    """

    @property
    def units(self) -> AbstractUnitSystem:
        """The unit system of the potential."""
        return cast(AbstractUnitSystem, self.potential.units)

    @property
    def constants(self) -> ImmutableDict[Quantity]:
        """The constants of the potential."""
        return cast("ImmutableDict[Quantity]", self.potential.constants)

    def _potential_energy(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : Array[float, (3,)]
            The position(s) at which to compute the potential energy.
        t : float
            The time at which to compute the potential energy.

        Returns
        -------
        Array[float, (...)]
            The potential energy at the given position(s).
        """
        q = q.at[..., 1].set(q[..., 1] / self.q1)
        q = q.at[..., 2].set(q[..., 2] / self.q2)
        return self.potential._potential_energy(q, t)  # noqa: SLF001


#####################################################################


@simplify_op.register  # type: ignore[misc]
def _simplify_op(
    pot: TriaxialInThePotential, /
) -> AbstractPotentialBase:  # TODO: better type hint
    """Simplify a TriaxialInThePotential."""
    if pot.q1 == 1 and pot.q2 == 1:
        return pot.potential
    return pot
