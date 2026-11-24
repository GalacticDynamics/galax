"""Private module that defines `TriaxialGaussianPotential` and its functions.

The public API is `galax.potential.triaxialgaussian`.
"""

__all__ = [
    # class
    "TriaxialGaussianPotential",
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
class TriaxialGaussianPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Triaxial Gaussian potential.

    A mass distribution of finite total mass whose density is a Gaussian in the
    ellipsoidal radius :math:`\xi`, with three different widths along the
    Cartesian axes. The parameter ``m_tot`` is the total mass, ``r_s`` the
    standard deviation along the :math:`x` axis, and ``q1`` and ``q2`` the axis
    ratios :math:`y / x` and :math:`z / x`. The density is finite and positive
    everywhere. For :math:`q_1 = q_2 = 1` it is
    :class:`~galax.potential.GaussianPotential`, and for :math:`q_1 = 1` it is
    :class:`~galax.potential.AxisymmetricGaussianPotential`.

    The density is

    .. math::

        \rho(\mathbf{x}) = \rho_0 \exp\left(-\frac{\xi^2}{2 r_s^2}\right),
        \qquad \rho_0 = \frac{m_{\mathrm{tot}}}{q_1 q_2 (2 \pi)^{3/2} r_s^3},

    where :math:`\xi^2 = x^2 + y^2 / q_1^2 + z^2 / q_2^2`. The factor
    :math:`q_1 q_2` in :math:`\rho_0` keeps :math:`m_{\mathrm{tot}}` equal to the
    total mass for every axis ratio. The potential has no closed form. With
    :math:`\Phi \to 0` at infinity it is the one-dimensional integral

    .. math::

        \Phi = -2 \pi G \rho_0 r_s^2 q_1 q_2 \int_0^1
            \frac{2 \exp\left(-\xi^2(s) / 2 r_s^2\right)}
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
    The integral is Chandrasekhar's (1969) potential of a heterogeneous
    ellipsoid, in the form of Merritt & Fridman (1996), after the change of
    variable :math:`\tau = 1 / s^2 - 1`. The potential at the centre is finite,
    and far from the centre :math:`\Phi \to -G m_{\mathrm{tot}} / r`. The density
    at the centre is :math:`\rho_0`.

    The quadrature has a fixed order, so its error grows where the integrand
    narrows, at large :math:`r / r_s`. Against a 30-digit mpmath integration,
    for axis ratios from 0.3 to 1.5, the relative error of the order-50
    potential is at most 3e-14 from :math:`10^{-3}` to :math:`30 r_s`, 2e-8 at
    :math:`100 r_s`, 1.1e-3 at :math:`300 r_s` and 2e-2 at :math:`10^3 r_s`, so
    raise ``integration_order`` for the far field. The functions are in
    `galax.potential.triaxialgaussian`. The gradient and Hessian of this class
    come from automatic differentiation of the potential.

    References
    ----------
    .. [Chandrasekhar1969TriGaussian] Chandrasekhar, S. (1969). Ellipsoidal Figures of
       Equilibrium. Yale University Press. Theorem 12.
       `ADS <https://ui.adsabs.harvard.edu/abs/1969efe..book.....C>`__.
    .. [MerrittFridman1996TriGaussian] Merritt, D., & Fridman, T. (1996). Triaxial
       Galaxies with Cusps. The Astrophysical Journal, 460, 136.
       `ADS <https://ui.adsabs.harvard.edu/abs/1996ApJ...460..136M>`__.
       The potential is their eq. 2a.

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
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
        r"""Central density.

        $$ \rho_0 = \frac{m_\mathrm{tot}}{q_1 q_2 (2 \pi)^{3/2} r_s^3} $$

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
def rho0_of_m(p: gt.Params, /) -> gt.Sz0:
    r"""Central density for the triaxial Gaussian model.

    .. math::

        \rho_0 = \frac{m_{\mathrm{tot}}}{q_1 q_2 (2 \pi)^{3/2} r_s^3}.

    """
    return p["m_tot"] / (p["q1"] * p["q2"] * (2 * jnp.pi) ** 1.5 * p["r_s"] ** 3)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz0:
    r"""Density of the triaxial Gaussian model.

    .. math::

        \rho(\mathbf{x}) = \rho_0 \exp\left(-\frac{\xi^2}{2 r_s^2}\right),
        \qquad \xi^2 = x^2 + \frac{y^2}{q_1^2} + \frac{z^2}{q_2^2},

    with :math:`\rho_0` from :func:`~galax.potential.triaxialgaussian.rho0_of_m`.

    """
    xi2 = (
        xyz[..., 0] ** 2
        + xyz[..., 1] ** 2 / p["q1"] ** 2
        + xyz[..., 2] ** 2 / p["q2"] ** 2
    ) / p["r_s"] ** 2
    return rho0_of_m(p) * jnp.exp(-xi2 / 2)  # type: ignore[no-any-return]


@ft.partial(jax.jit, static_argnames=("integration_order",))
def potential(
    p: gt.Params, xyz: gt.BBtSz3, *, integration_order: int = 50
) -> gt.BBtSz0:
    r"""Potential of the triaxial Gaussian model.

    .. math::

        \Phi = -2 \pi G \rho_0 r_s^2 q_1 q_2 \int_0^1
            \frac{2 \exp\left(-\xi^2(s) / 2 r_s^2\right)}
                 {\sqrt{\left[(q_1^2 - 1) s^2 + 1\right]
                        \left[(q_2^2 - 1) s^2 + 1\right]}} \, ds,

    with :math:`\xi^2(s) = s^2 \left[x^2 + y^2 / (1 + (q_1^2 - 1) s^2) + z^2 /
    (1 + (q_2^2 - 1) s^2)\right]`, by Gauss-Legendre quadrature of order
    ``integration_order``, which is a static argument. The positions may carry
    leading batch axes.

    """
    r_s, q1, q2 = p["r_s"], p["q1"], p["q2"]
    integral = chandrasekhar_integral(
        lambda w, r_s: 2.0 * jnp.exp(-(w / r_s**2) / 2),
        xyz,
        r_s,
        q1,
        q2,
        integration_order,
    )
    return (-2.0 * jnp.pi * p["G"] * rho0_of_m(p) * r_s**2 * q1 * q2) * integral  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
