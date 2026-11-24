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
    r"""Axisymmetric (oblate/prolate) Gaussian Potential.

    .. math::

        \rho(R, z) = \frac{m_\mathrm{tot}}{q_2 (2\pi)^{3/2} r_s^3}
        \exp\left(-\frac{\xi^2}{2 r_s^2}\right)

    where

    .. math::

        \xi^2 = R^2 + \frac{z^2}{q_2^2}, \qquad R^2 = x^2 + y^2

    The extra :math:`q_2` in the normalization (relative to the spherical
    `~galax.potential.GaussianPotential`) keeps :math:`m_\mathrm{tot}` equal
    to the true total mass for any flattening, as in
    `~galax.potential.TriaxialGaussianPotential`.

    This is the axisymmetric (:math:`q_1 = 1`) special case of
    `~galax.potential.TriaxialGaussianPotential` -- ``q2`` is the same z/x
    axis ratio as on that class, with ``q1`` fixed to 1. It is kept as its
    own class for API clarity in the common oblate/prolate case (one shape
    parameter instead of two), not for performance: profiling shows no
    measurable speedup over `~galax.potential.TriaxialGaussianPotential`
    with ``q1=1`` -- the one fewer multiply-add per quadrature node this
    class's simpler integrand saves is negligible next to the ``exp`` calls
    and dispatch overhead that actually dominate the cost.

    Setting up the potential still requires a 1D quadrature: unlike the
    spherical case (`~galax.potential.GaussianPotential`), a non-spherical
    density does not obey Newton's shell theorem, so there is no simple
    closed form here either -- only a handful of specific density profiles
    (Miyamoto-Nagai, Kuzmin, Satoh, ...) were designed to admit one, and the
    Gaussian is not among them.
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

        \rho_0 = \frac{m_\mathrm{tot}}{q_2 (2 \pi)^{3/2} r_s^3}
    """
    return p["m_tot"] / (p["q2"] * (2 * jnp.pi) ** 1.5 * p["r_s"] ** 3)  # type: ignore[no-any-return]


@ft.partial(jax.jit, static_argnames=("integration_order",))
def potential(
    p: gt.Params, xyz: gt.BBtSz3, /, *, integration_order: int = 50
) -> gt.BBtSz0:
    r"""Potential energy for the axisymmetric Gaussian.

    This is the :math:`q_1 = 1` special case of the general triaxial
    result (see `galax.potential.triaxialgaussian.potential`), for which the
    denominator under the integral collapses from two square-root factors to
    one:

    .. math::

        \Phi = -2 \pi G \rho_0 r_s^2 q_2 \int_{s=0}^{1}
            \frac{2 \exp(-\xi^2(s)/2)}{\sqrt{(q_2^2-1)s^2 + 1}} ds

    with :math:`\rho_0` from `rho0_of_m` and

    .. math::

        r_s^2 \xi^2(s) = s^2 \left[x^2 + y^2 +
            \frac{z^2}{1 + (q_2^2 - 1) s^2}\right].

    The integral is evaluated by Gauss-Legendre quadrature of order
    ``integration_order``, which is static under `jax.jit`. At the default
    order of 50 the relative error against a quadrature in mpmath is below
    1e-13 out to 30 scale radii, about 1e-8 at 100 scale radii and 1e-3 at
    300; raise ``integration_order`` to evaluate further out.
    """
    xyz = xyz[None]  # a leading axis of size 1, against the nodes
    batchdims = xyz.ndim - 2
    qsq = p["q2"] ** 2

    def integrand(s: jax.Array, /) -> jax.Array:
        s2 = s.reshape(s.shape + (1,) * batchdims) ** 2
        surface = s2 * (
            xyz[..., 0] ** 2
            + xyz[..., 1] ** 2
            + xyz[..., 2] ** 2 / (1 + (qsq - 1) * s2)
        )
        xi2 = surface / p["r_s"] ** 2
        return 2.0 * jnp.exp(-xi2 / 2) / jnp.sqrt((qsq - 1) * s2 + 1)  # type: ignore[no-any-return]

    integral = gauss_legendre(integrand, integration_order)
    prefactor = -2.0 * jnp.pi * p["G"] * rho0_of_m(p) * p["r_s"] ** 2 * p["q2"]
    return prefactor * integral  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    r"""Density for the axisymmetric Gaussian.

    .. math::

        \rho(R, z) = \rho_0 \exp\left(-\frac{\xi^2}{2 r_s^2}\right),
        \qquad \xi^2 = R^2 + \frac{z^2}{q_2^2}
    """
    xi2 = (xyz[..., 0] ** 2 + xyz[..., 1] ** 2 + xyz[..., 2] ** 2 / p["q2"] ** 2) / p[
        "r_s"
    ] ** 2
    return rho0_of_m(p) * jnp.exp(-xi2 / 2)  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
