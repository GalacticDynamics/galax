"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "TriaxialGaussianPotential",
]

import functools as ft
from dataclasses import KW_ONLY

from jaxtyping import Array, Float, Shaped
from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import GaussLegendreIntegrator


@final
class TriaxialGaussianPotential(AbstractSinglePotential):
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
        default=u.Q(1.0, ""),
        dimensions="dimensionless",
        doc="Scale length in the y/x direction.",
    )

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Q(1.0, ""),
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
    _integrator: GaussLegendreIntegrator = eqx.field(default=None)

    def __post_init__(self) -> None:
        integrator = GaussLegendreIntegrator.for_order(self.integration_order)
        object.__setattr__(self, "_integrator", integrator)

    # ==========================================================================

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def rho0(
        self, t: gt.BBtQuSz0, /, *, ustrip: bool = False
    ) -> gt.BtFloatQuSz0 | gt.BtFloatSz0:
        r"""Central density.

        $$ \rho_0 = \frac{m_\mathrm{tot}}{q_1 q_2 (2 \pi)^{3/2} r_s^3} $$

        """
        u1 = self.units["dimensionless"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"] if ustrip else None)
        r_s = self.r_s(t, ustrip=self.units["length"] if ustrip else None)
        q1 = self.q1(t, ustrip=u1 if ustrip else None)
        q2 = self.q2(t, ustrip=u1 if ustrip else None)
        return m_tot / (q1 * q2 * (2 * jnp.pi) ** 1.5 * r_s**3)

    # ==========================================================================
    # Potential energy

    @ft.partial(jax.jit, inline=True)
    def _ellipsoid_surface(
        self,
        q: Shaped[Array, "1 *batch 3"],
        q1sq: Shaped[Array, ""],
        q2sq: Shaped[Array, ""],
        s2: Shaped[Array, "N *#batch"],
    ) -> Shaped[u.Quantity["area"], "N *batch"]:
        r"""Compute coordinates on the ellipse.

        .. math::

            r_s^2 \xi^2(\tau) = \frac{x^2}{1 + \tau} + \frac{y^2}{q_1^2 + \tau}
            + \frac{z^2}{q_2^2 + \tau}

        """
        return s2 * (
            q[..., 0] ** 2
            + q[..., 1] ** 2 / (1 + (q1sq - 1) * s2)
            + q[..., 2] ** 2 / (1 + (q2sq - 1) * s2)
        )

    # TODO: fix this to enable non-Quantity mode.
    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
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

        References
        ----------
        .. [1] Chandrasekhar, S. (1969). Ellipsoidal figures of equilibrium.
               https://ui.adsabs.harvard.edu/abs/1969efe..book.....C
        .. [2] Merritt, D., & Fridman, T. (1996). Triaxial Galaxies with Cusps.
            Astrophysical Journal, 460, 136.

        """
        # Parse inputs
        xyz = u.Q.from_(xyz, self.units["length"])
        t = u.Q.from_(t, self.units["time"])

        # Compute parameters
        r_s = self.r_s(t)
        rho0 = self.rho0(t)
        q1, q2 = self.q1(t), self.q2(t)

        # A batch dimension is added here and below for the integration.
        xyz = xyz[None]
        batchdims: int = xyz.ndim - 2

        q1sq, q2sq = q1**2, q2**2

        # Delta(ψ) = ψ(∞) - ψ(ξ)
        # This factors out the rho0 * r_s^2, moving it to the end
        def delta_psi_factor(
            s2: Float[Array | u.AbstractQuantity, "N *#batch"],
        ) -> Float[Array | u.AbstractQuantity, "N *batch"]:
            xi2 = self._ellipsoid_surface(xyz, q1sq, q2sq, s2) / r_s**2
            return 2.0 * jnp.exp(-xi2 / 2)

        def integrand(s: Float[Array, "N"]) -> Float[Array, "N *batch"]:
            s2 = s.reshape(s.shape + (1,) * batchdims) ** 2
            denom = jnp.sqrt(((q1sq - 1) * s2 + 1) * ((q2sq - 1) * s2 + 1))
            return delta_psi_factor(s2) / denom

        # TODO: option to do integrate.quad
        integral = self._integrator(integrand)

        out = (-2.0 * jnp.pi * self.constants["G"] * rho0 * r_s**2 * q1 * q2) * integral
        return out.ustrip(self.units["specific energy"])

    # ==========================================================================

    # TODO: make this work w/out units
    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        # Parse inputs  # TODO: work w/out units
        xyz = u.Q.from_(xyz, self.units["length"])
        t = u.Q.from_(t, self.units["time"])

        # Compute parameters
        rho0 = self.rho0(t)
        r_s = self.r_s(t)
        q1sq, q2sq = self.q1(t) ** 2, self.q2(t) ** 2

        s2 = jnp.asarray([1])
        xi2 = self._ellipsoid_surface(xyz[None], q1sq, q2sq, s2)[0] / r_s**2

        dens = rho0 * jnp.exp(-xi2 / 2)
        return dens.ustrip(self.units["mass density"])
