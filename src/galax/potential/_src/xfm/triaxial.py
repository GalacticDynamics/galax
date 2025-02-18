"""Triaxial transformation on a potential."""

__all__ = ["TriaxialInThePotential"]

from functools import partial

import jax

import unxt as u

import galax.typing as gt
from .base import AbstractTransformedPotential
from galax.potential._src.base import AbstractPotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.utils._unxt import AllowValue


class TriaxialInThePotential(AbstractTransformedPotential):
    r"""Triaxiality in the Potential (not density).

    This is a transformation of the potential, not the density. The
    transformation is done by scaling the position coordinates then evaluating
    the potential at the transformed coordinates. Two ratios are used to scale
    the coordinates:

    - `y_over_x`: the ratio of the y-axis to the x-axis.
    - `z_over_x`: the ratio of the z-axis to the x-axis.

    This is applied as

    $$
        \Phi'(x, y, z) = \Phi(x, y / a_{y/x}, z / a_{z/x})
    $$

    So a `y_over_x` of 2 means that the y-axis is twice as long as the x-axis,
    which is achieved by shifting the y coordinate to twice its value.

    Note that as with all parameters in `galax`, the triaxiality ratios can be
    time dependent.

    Example
    -------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp

    >>> opot = gp.KeplerPotential(1e12, units="galactic")
    >>> xpot = gp.TriaxialInThePotential(opot, y_over_x=2.0)

    >>> x = u.Quantity([1, 0, 0], "kpc")
    >>> y = u.Quantity([0, 1, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")

    >>> opot.potential(x, t) == xpot.potential(x, t)
    Array(True, dtype=bool)

    >>> opot.potential(y, t)
    Quantity[...](Array(-4.49850215, dtype=float64), unit='kpc2 / Myr2')

    >>> xpot.potential(y, t)
    Quantity[...](Array(-8.9970043, dtype=float64), unit='kpc2 / Myr2')

    >>> opot.potential(y / 2, t) == xpot.potential(y, t)
    Array(True, dtype=bool)

    """

    #: the original potential
    original_potential: AbstractPotential

    #: ratio of the y-axis to the x-axis
    y_over_x: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(1, ""), dimensions="dimensionless"
    )

    #: ratio of the z-axis to the x-axis
    z_over_x: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(1, ""), dimensions="dimensionless"
    )

    @partial(jax.jit)
    def _potential(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtRealQuSz0 | gt.BBtRealSz0, /
    ) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz.astype(float))
        # Transform the position
        xyz = xyz.at[..., 1].divide(self.y_over_x(t).value)
        xyz = xyz.at[..., 2].divide(self.z_over_x(t).value)

        # Evaluate the potential energy at the transformed position, time.
        return self.original_potential._potential(xyz, t)  # noqa: SLF001
