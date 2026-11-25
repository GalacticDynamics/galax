"""Private module that defines `Vogelsberger08TriaxialNFWPotential` and its functions.

The public API is `galax.potential.vogelsberger08triaxialnfw`.
"""

__all__ = [
    # class
    "Vogelsberger08TriaxialNFWPotential",
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
from .base import potential as nfw_potential
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
from galax.potential._src.utils import safe_sqrt, safe_vector_norm


@final
class Vogelsberger08TriaxialNFWPotential(AbstractSinglePotential):
    r"""Triaxial NFW Potential from DOI 10.1111/j.1365-2966.2007.12746.x.

    The NFW potential evaluated at the radius

    .. math::

        \tilde{r} = \frac{(r_a + r)\, r_e}{r_a + r_e}, \qquad
        r_e = \sqrt{x^2 + y^2/q_1^2 + z^2/q_2^2}, \qquad r_a = a_r r_s,

    with :math:`q_2^2 = 3 - q_1^2` and :math:`r` the spherical radius. The
    isopotential surfaces are ellipsoids inside :math:`r_a` and near spheres
    outside it. The density is :math:`\nabla^2 \Phi / (4 \pi G)` of the
    expression above, found by automatic differentiation. At exactly the
    origin the Hessian and the Laplacian are 0, set by the floor in
    `safe_sqrt`, and not the cusp of the NFW profile, so evaluate it away from
    the origin.
    """

    m: AbstractParameter = ParameterField(dimensions="mass", doc="Scale mass.")  # type: ignore[assignment]
    # TODO: note the different definitions of m.

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.StaticQuantity(1.0, ""),
        doc="""y/x axis ratio.

    The z/x axis ratio is defined as :math:`q_2^2 = 3 - q_1^2`
    """,
    )

    a_r: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.StaticQuantity(1.0, ""),
        doc="""Transition radius relative to :math:`r_s`.

    :math:`r_a = a_r r_s  is a transition scale where the potential shape
    changes from ellipsoidal to near spherical.
    """,
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
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
    r"""Potential for the Vogelsberger et al. (2008) triaxial NFW potential.

    .. math::

        \Phi(\mathbf{x}) = \Phi_{\mathrm{NFW}}\left(\tilde{r}\right),
        \qquad
        \tilde{r} = \frac{(r_a + r)\, r_e}{r_a + r_e},
        \qquad r_a = a_r r_s,

    where :math:`\Phi_{\mathrm{NFW}}` is `galax.potential.nfw.potential`,
    :math:`r` is the spherical radius and :math:`r_e = \sqrt{x^2 + y^2/q_1^2 +
    z^2/q_2^2}` with :math:`q_2^2 = 3 - q_1^2`; the square root is `safe_sqrt`.
    The radius :math:`\tilde{r}` tends to :math:`r_e` well inside :math:`r_a` and
    to :math:`r` well outside it.
    """
    q1sq = p["q1"] ** 2
    q2sq = 3 - q1sq
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r_e = safe_sqrt(x**2 + y**2 / q1sq + z**2 / q2sq)
    r_a = p["a_r"] * p["r_s"]
    r_tilde = (r_a + safe_vector_norm(xyz)) * r_e / (r_a + r_e)
    # NFW's radial function of r~, whose derivatives do not cancel.
    return nfw_potential(p, r_tilde)  # type: ignore[no-any-return]


# Derived quantities, built from the kernel above.
laplacian = cartesian_laplacian(potential)
density = poisson_density(laplacian)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
