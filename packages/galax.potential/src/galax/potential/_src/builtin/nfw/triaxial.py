"""Private module that defines `TriaxialNFWPotential` and its functions.

The public API is `galax.potential.triaxialnfw`.
"""

__all__ = [
    # class
    "TriaxialNFWPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
    "rho0_of_m",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.builtin.nfw.base import rho0_of_m
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
    poisson_laplacian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import chandrasekhar_integral


@final
class TriaxialNFWPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Triaxial NFW potential.

    The NFW density law on ellipsoids: the density is the spherical NFW profile
    of the ellipsoidal radius :math:`\xi`, with different axis lengths along the
    Cartesian axes. The parameter ``m`` is the mass scale (it is the mass
    parameter of :class:`~galax.potential.NFWPotential`), ``r_s`` is the scale
    radius along the :math:`x` axis, and ``q1`` and ``q2`` are the axis ratios
    :math:`y / x` and :math:`z / x`. For :math:`q_1 = q_2 = 1` it is
    :class:`~galax.potential.NFWPotential`.

    The density is

    .. math::

        \rho(\mathbf{x}) = \frac{\rho_0}{(\xi / r_s) (1 + \xi / r_s)^2},
        \qquad \rho_0 = \frac{m}{4 \pi r_s^3},

    where :math:`\xi^2 = x^2 + y^2 / q_1^2 + z^2 / q_2^2`. The density diverges
    at the centre as :math:`\rho_0 r_s / \xi`, and falls as :math:`\xi^{-3}`
    far out, so the total mass diverges logarithmically, as for the spherical
    NFW. The potential has no closed form. With :math:`\Phi \to 0` at infinity
    it is the one-dimensional integral

    .. math::

        \Phi = -2 \pi G \rho_0 r_s^2 q_1 q_2 \int_0^1
            \frac{2 / (1 + \xi(s) / r_s)}
                 {\sqrt{\left[(q_1^2 - 1) s^2 + 1\right]
                        \left[(q_2^2 - 1) s^2 + 1\right]}} \, ds,

    with

    .. math::

        \xi^2(s) = s^2 \left[x^2 + \frac{y^2}{1 + (q_1^2 - 1) s^2}
                                 + \frac{z^2}{1 + (q_2^2 - 1) s^2}\right],

    where :math:`G` is the gravitational constant. It is evaluated by
    Gauss-Legendre quadrature of order ``integration_order``.

    Notes
    -----
    The integral is Chandrasekhar's (1969) [Chandrasekhar1969TriNFW]_ potential
    of a heterogeneous ellipsoid, in the form of Merritt & Fridman (1996)
    [MerrittFridman1996TriNFW]_, after the change of variable :math:`\tau = 1 /
    s^2 - 1`. The potential at the centre is finite, and for :math:`q_1 = q_2 =
    1` it is :math:`-G m / r_s`. Far from the centre it is :math:`-G m q_1 q_2
    \ln(r / r_s) / r` to leading order in the logarithm.

    The density diverges at exactly :math:`\xi = 0`, and the gradient and the
    Hessian of the potential there are NaN, as they come from automatic
    differentiation of a square root of zero. Evaluate away from the origin.

    The quadrature has a fixed order, so its error grows where the integrand
    narrows, at large :math:`r / r_s`. Against a 30-digit mpmath integration,
    for axis ratios from 0.3 to 1.5, the relative error of the order-50
    potential is at most 1e-14 from :math:`10^{-3}` to :math:`30 r_s`, 3e-9 at
    :math:`100 r_s`, 1e-5 at :math:`300 r_s` and 1.6e-3 at :math:`10^3 r_s`, so
    raise ``integration_order`` for the far field. The gradient and Hessian,
    which are derivatives of the quadrature, lose accuracy faster: for
    :math:`q_1 = 0.9`, :math:`q_2 = 0.6` the largest gradient error over six
    directions is below 4e-15 at :math:`30 r_s`, 3e-8 at :math:`100 r_s`, 5.3e-5
    at :math:`300 r_s` and 3.5e-3 at :math:`10^3 r_s`. The functions are in
    `galax.potential.triaxialnfw`. The gradient and Hessian of this class come
    from automatic differentiation of the potential.

    References
    ----------
    .. [Chandrasekhar1969TriNFW] Chandrasekhar, S. (1969). Ellipsoidal Figures of
       Equilibrium. Yale University Press. Theorem 12.
       `ADS <https://ui.adsabs.harvard.edu/abs/1969efe..book.....C>`__.
    .. [MerrittFridman1996TriNFW] Merritt, D., & Fridman, T. (1996). Triaxial
       Galaxies with Cusps. The Astrophysical Journal, 460, 136.
       `ADS <https://ui.adsabs.harvard.edu/abs/1996ApJ...460..136M>`__.
       The potential is their eq. 2a.

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="mass scale of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius of the potential."
    )

    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(1.0, ""),
        dimensions="dimensionless",
        doc="Scale length in the y/x direction.",
    )

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(1.0, ""),
        dimensions="dimensionless",
        doc="Scale length in the z/x direction.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    integration_order: int = eqx.field(default=50, static=True)
    """Order of the Gauss-Legendre quadrature.

    See :func:`numpy.polynomial.legendre.leggauss` for details.
    """
    # ==========================================================================

    symmetry = Symmetry.PLANE_REFLECTION
    """`Symmetry.PLANE_REFLECTION`: the potential is even in $x$, $y$ and $z$."""

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def rho0(
        self, t: gt.BBtQuSz0, /, *, ustrip: bool = False
    ) -> gt.BtFloatQuSz0 | gt.BtFloatSz0:
        r"""Characteristic density.

        $$ \rho_0 = \frac{m}{4 \pi r_s^3} $$

        """
        return rho0_of_m(self.params(t, ustrip=ustrip))

    # ==========================================================================

    @ft.partial(jax.jit, inline=True)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute parameters
        p = self.params(t)
        return potential(p, xyz, integration_order=self.integration_order)  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute parameters
        p = self.params(t)
        return density(p, xyz)  # type: ignore[no-any-return]


# ===================================================================
# Functions


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz0:
    r"""Density of the triaxial NFW model.

    .. math::

        \rho(\mathbf{x}) = \frac{\rho_0}{(\xi / r_s) (1 + \xi / r_s)^2},
        \qquad \xi^2 = x^2 + \frac{y^2}{q_1^2} + \frac{z^2}{q_2^2},

    with :math:`\rho_0` from :func:`~galax.potential.triaxialnfw.rho0_of_m`.
    The density is infinite at exactly :math:`\xi = 0`.

    """
    xi2 = (
        xyz[..., 0] ** 2
        + xyz[..., 1] ** 2 / p["q1"] ** 2
        + xyz[..., 2] ** 2 / p["q2"] ** 2
    )
    x = jnp.sqrt(xi2) / p["r_s"]
    return rho0_of_m(p) / x / (1.0 + x) ** 2  # type: ignore[no-any-return]


@ft.partial(jax.jit, static_argnames=("integration_order",))
def potential(
    p: gt.Params, xyz: gt.BBtSz3, *, integration_order: int = 50
) -> gt.BBtSz0:
    r"""Potential of the triaxial NFW model.

    .. math::

        \Phi = -2 \pi G \rho_0 r_s^2 q_1 q_2 \int_0^1
            \frac{2 / (1 + \xi(s) / r_s)}
                 {\sqrt{\left[(q_1^2 - 1) s^2 + 1\right]
                        \left[(q_2^2 - 1) s^2 + 1\right]}} \, ds,

    with :math:`\xi^2(s) = s^2 \left[x^2 + y^2 / (1 + (q_1^2 - 1) s^2) + z^2 /
    (1 + (q_2^2 - 1) s^2)\right]`, by Gauss-Legendre quadrature of order
    ``integration_order``, which is a static argument. The positions may carry
    leading batch axes.

    """
    r_s, q1, q2 = p["r_s"], p["q1"], p["q2"]
    integral = chandrasekhar_integral(
        lambda w: 2.0 / (1.0 + jnp.sqrt(w) / r_s),
        xyz,
        q1,
        q2,
        integration_order,
    )
    return (-2.0 * jnp.pi * p["G"] * rho0_of_m(p) * r_s**2 * q1 * q2) * integral  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
