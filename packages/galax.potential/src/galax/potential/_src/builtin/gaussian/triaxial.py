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
from galax.potential._src.utils import gauss_legendre


@final
class TriaxialGaussianPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Triaxial (density) Gaussian Potential.

    .. math::

        \rho(q) = \frac{m_\mathrm{tot}}{q_1 q_2 (2\pi)^{3/2} r_s^3}
        \exp\left(-\frac{\xi^2}{2 r_s^2}\right)

    where

    .. math::

        \xi^2 = x^2 + \frac{y^2}{q_1^2} + \frac{z^2}{q_2^2}

    The extra :math:`q_1 q_2` in the normalization (relative to the spherical
    `~galax.potential.GaussianPotential`) keeps :math:`m_\mathrm{tot}` equal
    to the true total mass for any flattening: squashing the density by
    :math:`q_1 q_2` in volume must be compensated by boosting the central
    density by the same factor to conserve mass.
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
        p = self.params(t, ustrip=ustrip)
        return rho0_of_m(p)

    # ==========================================================================
    # Potential energy

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute
        params = self.params(t)
        return potential(  # type: ignore[no-any-return]
            params, xyz, integration_order=self.integration_order
        )

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute
        params = self.params(t)
        return density(params, xyz)  # type: ignore[no-any-return]


# ====================================================================


@ft.partial(jax.jit)
def rho0_of_m(p: gt.Params, /) -> gt.Sz0:
    r"""Central density from the total mass.

    .. math::

        \rho_0 = \frac{m_\mathrm{tot}}{q_1 q_2 (2 \pi)^{3/2} r_s^3}
    """
    return p["m_tot"] / (p["q1"] * p["q2"] * (2 * jnp.pi) ** 1.5 * p["r_s"] ** 3)  # type: ignore[no-any-return]


@ft.partial(jax.jit, static_argnames=("integration_order",))
def potential(
    p: gt.Params, xyz: gt.BBtSz3, /, *, integration_order: int = 50
) -> gt.BBtSz0:
    r"""Potential energy for the triaxial Gaussian.

    The Gaussian potential is spherically symmetric. For the triaxial
    (density) case, we define ellipsoidal (and dimensionless) coordinates

    .. math::

        r_s^2 \xi^2 = x^2 + y^2/q_1^2 + z^2/q_2^2

    Which maps ellipsoids to spheres. The spherical potential is then
    evaluated under this mapping.

    Chandrasekhar (1969) [1]_ Theorem 12 (though more clearly written in
    Merritt & Fridman (1996) [2]_ eq 2a) gives the form of the
    gravitational potential.

    .. math::

        \Phi = -\pi G q_1 q_2 \int_{0}^{\infty}
            \frac{\Delta \psi(\xi(\tau))}{\sqrt{(1+\tau)(q_1^2 + \tau)(q_2^2
            + \tau)}} d\tau

    with:

    .. math::

        \Delta \psi(\xi) = \psi(\infty) - \psi(\xi)
                         = \int_{\xi^2}^{\infty} \rho(\xi^2) d\xi^2 = \rho_0
                         r_s^2 \, 2 \exp\left(-\frac{\xi^2}{2}\right)

    and

    .. math::

        r_s^2 \xi^2(\tau) = x^2 + \frac{y^2}{q_1^2+\tau} +
        \frac{z^2}{q_2^2+\tau}

    We apply the change of variables :math:`\tau = 1/s^2 - 1` to map the
    integral to the interval [0, 1].

    .. math::

        \Phi = -2 \pi G q_1 q_2 \int_{s=0}^{1} \frac{
            (1+\tau) \Delta\psi(\xi(s))}{\sqrt{(q_1^2+\tau)(q_2^2+\tau)}} ds
             = -2 \pi G q_1 q_2 \int_{s=0}^{1} \frac{\Delta\psi(\xi(s))}
                {\sqrt{((q_1^2-1)s^2 + 1)((q_2^2-1)s^2 + 1)}} ds

    The integral is evaluated by Gauss-Legendre quadrature of order
    ``integration_order``, which is static under `jax.jit`. At the default
    order of 50 the relative error against a quadrature in mpmath is below
    1e-13 out to 30 scale radii, about 1e-8 at 100 scale radii and 1e-3 at
    300; raise ``integration_order`` to evaluate further out.

    References
    ----------
    .. [1] Chandrasekhar, S. (1969). Ellipsoidal figures of equilibrium.
           https://ui.adsabs.harvard.edu/abs/1969efe..book.....C
    .. [2] Merritt, D., & Fridman, T. (1996). Triaxial Galaxies with Cusps.
        Astrophysical Journal, 460, 136.
    """
    xyz = xyz[None]  # a leading axis of size 1, against the nodes
    batchdims = xyz.ndim - 2
    q1sq, q2sq = p["q1"] ** 2, p["q2"] ** 2

    def integrand(s: jax.Array, /) -> jax.Array:
        s2 = s.reshape(s.shape + (1,) * batchdims) ** 2
        surface = s2 * (
            xyz[..., 0] ** 2
            + xyz[..., 1] ** 2 / (1 + (q1sq - 1) * s2)
            + xyz[..., 2] ** 2 / (1 + (q2sq - 1) * s2)
        )
        xi2 = surface / p["r_s"] ** 2
        denom = jnp.sqrt(((q1sq - 1) * s2 + 1) * ((q2sq - 1) * s2 + 1))
        return 2.0 * jnp.exp(-xi2 / 2) / denom  # type: ignore[no-any-return]

    integral = gauss_legendre(integrand, integration_order)
    prefactor = (
        -2.0 * jnp.pi * p["G"] * rho0_of_m(p) * p["r_s"] ** 2 * p["q1"] * p["q2"]
    )
    return prefactor * integral  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    r"""Density for the triaxial Gaussian.

    .. math::

        \rho(q) = \rho_0 \exp\left(-\frac{\xi^2}{2 r_s^2}\right),
        \qquad \xi^2 = x^2 + \frac{y^2}{q_1^2} + \frac{z^2}{q_2^2}
    """
    xi2 = (
        xyz[..., 0] ** 2
        + xyz[..., 1] ** 2 / p["q1"] ** 2
        + xyz[..., 2] ** 2 / p["q2"] ** 2
    ) / p["r_s"] ** 2
    return rho0_of_m(p) * jnp.exp(-xi2 / 2)  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
