"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "AxisymmetricGaussianPotential",
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
class AxisymmetricGaussianPotential(AbstractSinglePotential):
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
        default=u.Q(1.0, ""),
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

        $$ \rho_0 = \frac{m_\mathrm{tot}}{q_2 (2 \pi)^{3/2} r_s^3} $$

        """
        u1 = self.units["dimensionless"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"] if ustrip else None)
        r_s = self.r_s(t, ustrip=self.units["length"] if ustrip else None)
        q2 = self.q2(t, ustrip=u1 if ustrip else None)
        return m_tot / (q2 * (2 * jnp.pi) ** 1.5 * r_s**3)

    # ==========================================================================
    # Potential energy

    @ft.partial(jax.jit, inline=True)
    def _spheroid_surface(
        self,
        q: Shaped[Array, "1 *batch 3"],
        qsq: Shaped[Array, ""],
        s2: Shaped[Array, "N *#batch"],
    ) -> Shaped[u.Quantity["area"], "N *batch"]:
        r"""Compute coordinates on the spheroid.

        .. math::

            r_s^2 \xi^2(\tau) = \frac{x^2 + y^2}{1 + \tau} + \frac{z^2}{q_2^2
            + \tau}

        """
        return s2 * (
            q[..., 0] ** 2 + q[..., 1] ** 2 + q[..., 2] ** 2 / (1 + (qsq - 1) * s2)
        )

    # TODO: fix this to enable non-Quantity mode.
    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        r"""Potential energy for the axisymmetric Gaussian.

        This is the :math:`q_1 = 1` special case of the general triaxial
        result (see `TriaxialGaussianPotential._potential`), for which the
        denominator under the integral collapses from two square-root
        factors to one:

        .. math::

            \Phi = -2 \pi G q_2 \int_{s=0}^{1} \frac{\Delta\psi(\xi(s))}
                {\sqrt{(q_2^2-1)s^2 + 1}} ds

        with :math:`\Delta \psi(\xi) = 2 \rho_0 r_s^2 \exp(-\xi^2/2)` as in
        the triaxial case.
        """
        # Parse inputs
        xyz = u.Q.from_(xyz, self.units["length"])
        t = u.Q.from_(t, self.units["time"])

        # Compute parameters
        r_s = self.r_s(t)
        rho0 = self.rho0(t)
        q2 = self.q2(t)

        # A batch dimension is added here and below for the integration.
        xyz = xyz[None]
        batchdims: int = xyz.ndim - 2

        qsq = q2**2

        # Delta(ψ) = ψ(∞) - ψ(ξ)
        # This factors out the rho0 * r_s^2, moving it to the end
        def delta_psi_factor(
            s2: Float[Array | u.AbstractQuantity, "N *#batch"],
        ) -> Float[Array | u.AbstractQuantity, "N *batch"]:
            xi2 = self._spheroid_surface(xyz, qsq, s2) / r_s**2
            return 2.0 * jnp.exp(-xi2 / 2)

        def integrand(s: Float[Array, "N"]) -> Float[Array, "N *batch"]:
            s2 = s.reshape(s.shape + (1,) * batchdims) ** 2
            denom = jnp.sqrt((qsq - 1) * s2 + 1)
            return delta_psi_factor(s2) / denom  # type: ignore[no-any-return]

        # TODO: option to do integrate.quad
        integral = self._integrator(integrand)

        out = (-2.0 * jnp.pi * self.constants["G"] * rho0 * r_s**2 * q2) * integral
        return out.ustrip(self.units["specific energy"])  # type: ignore[no-any-return]

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
        qsq = self.q2(t) ** 2

        s2 = jnp.asarray([1])
        xi2 = self._spheroid_surface(xyz[None], qsq, s2)[0] / r_s**2

        dens = rho0 * jnp.exp(-xi2 / 2)
        return dens.ustrip(self.units["mass density"])  # type: ignore[no-any-return]
