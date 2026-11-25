"""Private module that defines `TriaxialHernquistPotential` and its functions.

The public API is `galax.potential.triaxialhernquist`.
"""

__all__ = [
    # class
    "TriaxialHernquistPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax

import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .hernquist import potential as hernquist_potential
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
    cartesian_laplacian,
    poisson_density,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import safe_sqrt


@final
class TriaxialHernquistPotential(AbstractSinglePotential):
    r"""Triaxial Hernquist Potential.

    The `~galax.potential.HernquistPotential` evaluated on the ellipsoidal
    radius

    .. math::

        r' = \sqrt{x^2 + (y / q_1)^2 + (z / q_2)^2},

    so that the surfaces of constant potential are ellipsoids with axes in the
    ratio :math:`1 : q_1 : q_2`. The potential is

    .. math::

        \Phi = -\frac{G m_{\mathrm{tot}}}{r' + r_s}.

    This is not the potential of a Hernquist density with ellipsoidal
    isodensity surfaces. The density is :math:`\nabla^2 \Phi / (4 \pi G)` of the
    expression above, found by automatic differentiation. At exactly the origin
    the Hessian is a large number set by the floor in `safe_sqrt`, not a
    physical value, so evaluate it away from the origin.

    Parameters
    ----------
    m_tot : :class:`~galax.potential.AbstractParameter`['mass']
        Mass parameter. This can be a
        :class:`~galax.potential.AbstractParameter` or an appropriate callable
        or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    r_s : :class:`~galax.potential.AbstractParameter`['length']
        A scale length that determines the concentration of the system.  This
        can be a :class:`~galax.potential.AbstractParameter` or an appropriate
        callable or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    q1 : :class:`~galax.potential.AbstractParameter`['dimensionless']
        Axis ratio of the y axis to the x axis.
    q2 : :class:`~galax.potential.AbstractParameter`['dimensionless']
        Axis ratio of the z axis to the x axis.

    units : :class:`~unxt.AbstractUnitSystem`, keyword-only
        The unit system to use for the potential.  This parameter accepts a
        :class:`~unxt.AbstractUnitSystem` or anything that can be converted to a
        :class:`~unxt.AbstractUnitSystem` using :func:`~unxt.unitsystem`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.TriaxialHernquistPotential(m_tot=1e12, r_s=8, q1=1, q2=0.5,
    ...                                     units="galactic")

    >>> q = u.Q([1, 0, 0], "kpc")
    >>> t = u.Q(0, "Gyr")
    >>> pot.potential(q, t)
    Q(-0.49983357, 'kpc2 / Myr2')
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length",
        doc="Scale a scale length that determines the concentration of the system.",
    )

    # TODO: move to a triaxial wrapper
    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(1.0, ""),
        dimensions="dimensionless",
        doc="Axis ratio of the y axis to the x axis.",
    )

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(1.0, ""),
        dimensions="dimensionless",
        doc="Axis ratio of the z axis to the x axis.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        converter=ImmutableMap, default=default_constants
    )

    symmetry = Symmetry.PLANE_REFLECTION
    """`Symmetry.PLANE_REFLECTION`: the potential is even in $x$, $y$ and $z$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute
        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]


# ====================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtSz0:
    r"""Potential for the triaxial Hernquist potential.

    .. math::

        \Phi = -\frac{G m_{\mathrm{tot}}}{r' + r_s}, \qquad
        r' = \sqrt{x^2 + (y / q_1)^2 + (z / q_2)^2}

    where :math:`m_{\mathrm{tot}}` is the total mass, :math:`r_s` the scale
    radius and :math:`q_1`, :math:`q_2` the axis ratios. The square root is
    `safe_sqrt`, which keeps the value finite at the origin.
    """
    rprime = safe_sqrt(
        xyz[..., 0] ** 2 + (xyz[..., 1] / p["q1"]) ** 2 + (xyz[..., 2] / p["q2"]) ** 2
    )
    return hernquist_potential(p, rprime)  # type: ignore[no-any-return]


# Derived quantities, built from the kernel above.
laplacian = cartesian_laplacian(potential)
density = poisson_density(laplacian)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
