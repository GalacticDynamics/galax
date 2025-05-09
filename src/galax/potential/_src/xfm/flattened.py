"""Flattened transformation on a potential."""

__all__ = ["FlattenedInThePotential"]

from functools import partial

import jax

import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from .base import AbstractTransformedPotential
from galax.potential._src.base import AbstractPotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


class FlattenedInThePotential(AbstractTransformedPotential):
    r"""Flattening in the Potential (not density).

    This is a transformation of the potential, not the density. The
    transformation is done by scaling the position coordinates then evaluating
    the potential at the transformed coordinates. This potential flattens the z-axis of
    the potential.

    - `q_z`: the scaling of the z-axis

    This is applied as

    $$
        \Phi'(x, y, z) = \Phi(x, y, z / a)
    $$

    So a `q_z` of 2 means that the z-axis is twice as long as the x- or y-axis.

    See also `galax.potential.TriaxialInThePotential` for a triaxial equivalent of this
    wrapper class.

    Example
    -------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp

    >>> opot = gp.HernquistPotential(1e12, 10.0, units="galactic")
    >>> xpot = gp.FlattenedInThePotential(opot, q_z=2.0)

    >>> x1 = u.Quantity([1, 0, 0], "kpc")
    >>> x2 = u.Quantity([0, 0, 1], "kpc")
    >>> t = u.Quantity(0, "Gyr")

    >>> opot.potential(x1, t) == xpot.potential(x1, t)
    Array(True, dtype=bool)

    >>> opot.potential(x2, t)
    Quantity[...](Array(-0.40895474, dtype=float64), unit='kpc2 / Myr2')

    >>> xpot.potential(x2, t)
    Quantity[...](Array(-0.42842878, dtype=float64), unit='kpc2 / Myr2')

    >>> opot.potential(x2 / 2.0, t) == xpot.potential(x2, t)
    Array(True, dtype=bool)

    """

    #: the original potential
    base_potential: AbstractPotential

    q_z: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(1, ""),
        dimensions="dimensionless",
        doc="ratio of the z-axis to the x- or y-axis (or cylindrical radius)",
    )

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz.astype(float))
        # Transform the position
        u1 = self.units["dimensionless"]
        xyz = xyz.at[..., 2].divide(self.q_z(t, ustrip=u1))

        # Evaluate the potential energy at the transformed position, time.
        return self.base_potential._potential(xyz, t)  # noqa: SLF001
