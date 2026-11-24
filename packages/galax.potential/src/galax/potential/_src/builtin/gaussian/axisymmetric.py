"""Private module that defines `AxisymmetricGaussianPotential` and its functions.

The public API is `galax.potential.axisymmetricgaussian`.
"""

__all__ = [
    # class
    "AxisymmetricGaussianPotential",
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

from jaxtyping import Array
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
from galax.potential._src.utils import gauss_legendre


@final
class AxisymmetricGaussianPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Axisymmetric (oblate or prolate) Gaussian potential.

    A flattened or elongated mass distribution of finite total mass whose
    density is a Gaussian in the spheroidal radius :math:`\xi`. The parameter
    ``m_tot`` is the total mass, ``r_s`` the standard deviation of the Gaussian
    along the :math:`x` and :math:`y` axes, and ``q2`` the axis ratio
    :math:`z / R`: :math:`q_2 < 1` is oblate and :math:`q_2 > 1` is prolate. The
    density is finite and positive everywhere. It is the :math:`q_1 = 1` case of
    :class:`~galax.potential.TriaxialGaussianPotential`, and for :math:`q_2 = 1`
    it is :class:`~galax.potential.GaussianPotential`.

    The density is

    .. math::

        \rho(R, z) = \rho_0 \exp\left(-\frac{\xi^2}{2 r_s^2}\right),
        \qquad \rho_0 = \frac{m_{\mathrm{tot}}}{q_2 (2 \pi)^{3/2} r_s^3},

    where :math:`\xi^2 = R^2 + z^2 / q_2^2` and :math:`R^2 = x^2 + y^2`. The
    factor :math:`q_2` in :math:`\rho_0` keeps :math:`m_{\mathrm{tot}}` equal to
    the total mass for every flattening. The potential has no closed form. With
    :math:`\Phi \to 0` at infinity it is the one-dimensional integral

    .. math::

        \Phi(R, z) = -2 \pi G \rho_0 r_s^2 q_2 \int_0^1
            \frac{2 \exp\left(-\xi^2(s) / 2 r_s^2\right)}
                 {\sqrt{(q_2^2 - 1) s^2 + 1}} \, ds,

    with :math:`\xi^2(s) = s^2 \left[R^2 + z^2 / (1 + (q_2^2 - 1) s^2)\right]`,
    where :math:`G` is the gravitational constant. It is evaluated by
    Gauss-Legendre quadrature of order ``integration_order``.

    Notes
    -----
    The integral is Chandrasekhar's (1969) potential of a heterogeneous
    spheroid, in the form of Merritt & Fridman (1996), after the change of
    variable :math:`\tau = 1 / s^2 - 1`. At the centre the potential is finite,
    :math:`\Phi(0) = -\sqrt{2 / \pi}\, (G m_{\mathrm{tot}} / r_s)\, f(q_2)` with
    :math:`f = \arcsin(e) / e` and :math:`e = \sqrt{1 - q_2^2}` for
    :math:`q_2 < 1`, and :math:`f = \operatorname{arcsinh}(e) / e` with
    :math:`e = \sqrt{q_2^2 - 1}` for :math:`q_2 > 1`. Far from the centre
    :math:`\Phi \to -G m_{\mathrm{tot}} / r`. The density at the centre is
    :math:`\rho_0`.

    The quadrature has a fixed order, so its error grows where the integrand
    narrows, at large :math:`r / r_s`. Against a 30-digit mpmath integration the
    relative error of the order-50 potential is below 3e-14 from :math:`10^{-3}`
    to :math:`30 r_s`, 2e-8 at :math:`100 r_s`, 1e-3 at :math:`300 r_s` and 2e-2
    at :math:`10^3 r_s`, so raise ``integration_order`` for the far field. The
    functions are in `galax.potential.axisymmetricgaussian`. The gradient and
    Hessian of this class come from automatic differentiation of the potential.

    References
    ----------
    .. [Chandrasekhar1969] Chandrasekhar, S. (1969). Ellipsoidal Figures of
       Equilibrium. Yale University Press. Theorem 12.
       `ADS <https://ui.adsabs.harvard.edu/abs/1969efe..book.....C>`__.
    .. [MerrittFridman1996] Merritt, D., & Fridman, T. (1996). Triaxial
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

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(1.0, ""),
        dimensions="dimensionless",
        doc="Axis ratio z/R. q2 < 1 is oblate, q2 > 1 is prolate.",
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

    symmetry = Symmetry.ZROTATION_ZREFLECTION
    """`Symmetry.ZROTATION_ZREFLECTION`: the density depends on $R$ and $|z|$."""

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def rho0(
        self, t: gt.BBtQuSz0, /, *, ustrip: bool = False
    ) -> gt.BtFloatQuSz0 | gt.BtFloatSz0:
        r"""Central density.

        $$ \rho_0 = \frac{m_\mathrm{tot}}{q_2 (2 \pi)^{3/2} r_s^3} $$

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
    r"""Central density for the axisymmetric Gaussian model.

    .. math::

        \rho_0 = \frac{m_{\mathrm{tot}}}{q_2 (2 \pi)^{3/2} r_s^3}.

    """
    return p["m_tot"] / (p["q2"] * (2 * jnp.pi) ** 1.5 * p["r_s"] ** 3)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz0:
    r"""Density of the axisymmetric Gaussian model.

    .. math::

        \rho(R, z) = \rho_0 \exp\left(-\frac{\xi^2}{2 r_s^2}\right),
        \qquad \xi^2 = R^2 + \frac{z^2}{q_2^2},

    with :math:`\rho_0` from :func:`~galax.potential.axisymmetricgaussian.rho0_of_m`.

    """
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    xi2 = (R2 + xyz[..., 2] ** 2 / p["q2"] ** 2) / p["r_s"] ** 2
    return rho0_of_m(p) * jnp.exp(-xi2 / 2)  # type: ignore[no-any-return]


@ft.partial(jax.jit, static_argnames=("integration_order",))
def potential(
    p: gt.Params, xyz: gt.BBtSz3, *, integration_order: int = 50
) -> gt.BBtSz0:
    r"""Potential of the axisymmetric Gaussian model.

    .. math::

        \Phi = -2 \pi G \rho_0 r_s^2 q_2 \int_0^1
            \frac{2 \exp\left(-\xi^2(s) / 2 r_s^2\right)}
                 {\sqrt{(q_2^2 - 1) s^2 + 1}} \, ds,
        \qquad
        \xi^2(s) = s^2 \left[R^2 + \frac{z^2}{1 + (q_2^2 - 1) s^2}\right],

    by Gauss-Legendre quadrature of order ``integration_order``, which is a
    static argument. The positions may carry leading batch axes.

    """
    r_s, q2 = p["r_s"], p["q2"]
    qsq = q2**2
    # The node axis comes first: (N, 1, ..., 1) against the batch of `xyz`.
    batchdims = xyz.ndim - 1
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z2 = xyz[..., 2] ** 2

    def integrand(s: Array) -> Array:
        s2 = s.reshape(s.shape + (1,) * batchdims) ** 2
        xi2 = s2 * (R2 + z2 / (1 + (qsq - 1) * s2)) / r_s**2
        return 2.0 * jnp.exp(-xi2 / 2) / jnp.sqrt((qsq - 1) * s2 + 1)  # type: ignore[no-any-return]

    integral = gauss_legendre(integrand, integration_order)
    return (-2.0 * jnp.pi * p["G"] * rho0_of_m(p) * r_s**2 * q2) * integral  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
