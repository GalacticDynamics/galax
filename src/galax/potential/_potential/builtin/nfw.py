"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "NFWPotential",
    "LeeSutoTriaxialNFWPotential",
    "TriaxialNFWPotential",
    "Vogelsberger08TriaxialNFWPotential",
]

from collections.abc import Callable
from dataclasses import KW_ONLY
from functools import partial
from typing import final

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float, Shaped

import quaxed.array_api as xp
import quaxed.lax as qlax
from unxt import AbstractUnitSystem, Quantity, unitsystem

import galax.typing as gt
from galax.potential._potential.base import default_constants
from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.utils import ImmutableDict

_log2 = xp.log(xp.asarray(2.0))

# -------------------------------------------------------------------


@final
class NFWPotential(AbstractPotential):
    r"""NFW Potential.

    .. math::

        \rho(r) = -\frac{G M}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s})
        \Phi(r) = -\frac{G M}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s})

    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Mass parameter. This is NOT the total mass."""

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale radius of the potential."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        r"""Potential energy.

        .. math::

            \Phi(r) = -\frac{G M}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s})
        """
        r = xp.linalg.vector_norm(q, axis=-1)
        r_s = self.r_s(t)
        u = r / r_s
        v_h2 = self.constants["G"] * self.m(t) / r_s
        return -v_h2 * xp.log(1.0 + u) / u

    @partial(jax.jit)
    def _density(
        self, q: gt.BatchQVec3, /, t: gt.BatchRealQScalar | gt.RealQScalar
    ) -> gt.BatchFloatQScalar:
        r"""Density.

        .. math::

            v_{h2} = -\frac{G M}{r_s}
            \rho_0 = \frac{v_{h2}}{4 \pi G r_s^2}
            \rho(r) = \frac{\rho_0}{u (1 + u)^2}
        """
        r = xp.linalg.vector_norm(q, axis=-1)
        r_s = self.r_s(t)
        rho0 = self.m(t) / (4 * xp.pi * r_s**3)
        u = r / r_s
        return rho0 / u / (1 + u) ** 2


# -------------------------------------------------------------------


@final
class LeeSutoTriaxialNFWPotential(AbstractPotential):
    """Approximate triaxial (in the density) NFW potential.

    Approximation of a Triaxial NFW Potential with the flattening in the
    density, not the potential. See Lee & Suto (2003) for details.

    .. warning::

        This potential is only physical for `a1 >= a2 >= a3`.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.LeeSutoTriaxialNFWPotential(
    ...    m=Quantity(1e11, "Msun"), r_s=Quantity(15, "kpc"),
    ...    a1=1, a2=0.9, a3=0.8, units="galactic")

    >>> q = Quantity([1, 0, 0], "kpc")
    >>> t = Quantity(0, "Gyr")
    >>> pot.potential(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-0.14620419, dtype=float64), unit='kpc2 / Myr2')

    >>> q = Quantity([0, 1, 0], "kpc")
    >>> pot.potential(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-0.14593972, dtype=float64), unit='kpc2 / Myr2')

    >>> q = Quantity([0, 0, 1], "kpc")
    >>> pot.potential(q, t).decompose(pot.units)
    Quantity['specific energy'](Array(-0.14570309, dtype=float64), unit='kpc2 / Myr2')
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r"""Scall mass.

    This is the mass corresponding to the circular velocity at the scale radius.
    :math:`v_c = \sqrt{G M / r_s}`
    """

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale radius."""

    a1: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """Major axis."""

    a2: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """Intermediate axis."""

    a3: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """Minor axis."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    def __check_init__(self) -> None:
        t = Quantity(0.0, "Myr")
        _ = eqx.error_if(
            t,
            (self.a1(t) < self.a2(t)) or (self.a2(t) < self.a3(t)),
            "a1 >= a2 >= a3 is required",
        )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        # https://github.com/adrn/gala/blob/2067009de41518a71c674d0252bc74a7b2d78a36/gala/potential/potential/builtin/builtin_potentials.c#L1472
        # Evaluate the parameters
        r_s = self.r_s(t)
        v_c2 = self.constants["G"] * self.m(t) / r_s
        a1, a2, a3 = self.a1(t), self.a2(t), self.a3(t)

        # 1- eccentricities
        e_b2 = 1 - xp.square(a2 / a1)
        e_c2 = 1 - xp.square(a3 / a1)

        # The potential at the origin
        phi0 = v_c2 / (_log2 - 0.5 + (_log2 - 0.75) * (e_b2 + e_c2))

        # The potential at the given position
        r = xp.linalg.vector_norm(q, axis=-1)
        u = r / r_s

        # The functions F1, F2, and F3 and some useful quantities
        log1pu = xp.log(1 + u)
        u2 = u**2
        um3 = u ** (-3)
        costh2 = q[..., 2] ** 2 / r**2  # z^2 / r^2
        sinthsinphi2 = q[..., 1] ** 2 / r**2  # (sin(theta) * sin(phi))^2
        # Note that ꜛ is safer than computing the separate pieces, as it avoids
        # x=y=0, z!=0, which would result in a NaN.

        F1 = -log1pu / u
        F2 = -1.0 / 3 + (2 * u2 - 3 * u + 6) / (6 * u2) + (1 / u - um3) * log1pu
        F3 = (u2 - 3 * u - 6) / (2 * u2 * (1 + u)) + 3 * um3 * log1pu

        # Select the output, r=0 is a special case.
        out: gt.BatchFloatQScalar = phi0 * qlax.select(
            u == 0,
            xp.ones_like(u),
            (
                F1
                + (e_b2 + e_c2) / 2 * F2
                + (e_b2 * sinthsinphi2 + e_c2 * costh2) / 2 * F3
            ),
        )
        return out


# -------------------------------------------------------------------


class GaussLegendreIntegrator(eqx.Module):  # type: ignore[misc]
    """Gauss-Legendre quadrature integrator."""

    x: Shaped[Array, "O"]
    w: Shaped[Array, "O"]

    @partial(jax.jit, static_argnums=(1,))
    def __call__(
        self, f: Callable[[Shaped[Array, "N"]], Shaped[Array, "N"]], /
    ) -> Shaped[Array, ""]:
        y = f(self.x)
        w = self.w.reshape(self.w.shape + (1,) * (y.ndim - 1))
        return xp.sum(y * w, axis=0)


@final
class TriaxialNFWPotential(AbstractPotential):
    r"""Triaxial (density) NFW Potential.

    .. math::

        \rho(q) = \frac{G M}{4\pi r_s^3} \frac{1}{(\xi/r_s) (1 + \xi/r_s)^2}

    where

    .. math::

        \xi^2 = x^2 + \frac{y^2}{q_1^2} + \frac{z^2}{q_2^2}
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """mass scale of the potential."""

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale radius of the potential."""

    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=Quantity(1, ""),
        dimensions="dimensionless",
    )
    """Scale length in the y/x direction."""

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=Quantity(1, ""),
        dimensions="dimensionless",
    )
    """Scale length in the z/x direction."""

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    integration_order: int = eqx.field(default=50, static=True)
    """Order of the Gauss-Legendre quadrature.

    See :func:`numpy.polynomial.legendre.leggauss` for details.
    """
    _integrator: GaussLegendreIntegrator = eqx.field(init=False)

    def __post_init__(self) -> None:
        # Gauss-Legendre quadrature
        x, w = np.polynomial.legendre.leggauss(self.integration_order)
        # Interval change from [-1, 1] to [0, 1]
        x = 0.5 * (x + 1)
        w = 0.5 * w
        object.__setattr__(self, "_integrator", GaussLegendreIntegrator(x, w))

    # ==========================================================================

    @partial(jax.jit, inline=True)
    def rho0(self, t: gt.BatchableRealQScalar, /) -> gt.BatchFloatQScalar:
        r"""Central density.

        .. math::

            \rho_0 = \frac{M}{4 \pi r_s^3}
        """
        return self.m(t) / (4 * xp.pi * self.r_s(t) ** 3)

    # ==========================================================================
    # Potential energy

    @partial(jax.jit, inline=True)
    def _ellipsoid_surface(
        self,
        q: Shaped[Quantity["length"], "*batch 3"],
        q1: Shaped[Quantity["dimensionless"], "*#batch"],
        q2: Shaped[Quantity["dimensionless"], "*#batch"],
        s2: Shaped[Quantity["dimensionless"], "*#batch"],
    ) -> Shaped[Quantity["length"], "*batch"]:
        r"""Compute coordinates on the ellipse.

        .. math::

            r_s^2 \xi^2(\tau) = \frac{x^2}{1 + \tau} + \frac{y^2}{q_1^2 + \tau}
            + \frac{z^2}{q_2^2 + \tau}

        """
        return s2 * (
            q[..., 0] ** 2
            + q[..., 1] ** 2 / (1 + (q1**2 - 1) * s2)
            + q[..., 2] ** 2 / (1 + (q2**2 - 1) * s2)
        )

    @partial(jax.jit)
    def _potential(
        self,
        q: gt.BatchQVec3,
        t: gt.BatchableRealQScalar,
        /,
    ) -> gt.BatchFloatQScalar:
        r"""Potential energy for the triaxial NFW.

        The NFW potential is spherically symmetric. For the triaxial (density)
        case, we define ellipsoidal (and dimensionless) coordinates

        .. math::

            r_s^2 \xi^2 = x^2 + y^2/q_1^2 + z^2/q_2^2

        Which maps ellipsoids to spheres. The spherical potential is then
        evaluated under this mapping.

        Chandrasekhar (1969) [1]_ Theorem 12 (though more clearly written in
        Merritt & Fridman (1996) [2]_ eq 2a) gives the form of the gravitational
        potential.

        .. math::

            \Phi = -\pi G q_1 q_2 \int_{0}^{\infty}
                \frac{\Delta \psi(\xi(\tau))}{\sqrt{(1+\tau)(q_1^2 + \tau)(q_2^2
                + \tau)}} d\tau

        with:

        .. math::

            \Delta \psi(\xi) = \psi(\infty) - \psi(\xi)
                             = \int_{\xi^2}^{\infty} \rho(\xi^2) d\xi^2 = \rho_0
                             r_s^2 \frac{2}{1 + \xi}

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
        # A batch dimension is added here and below for the integration.
        q = q[None]

        # Compute the parameters.
        r_s, q1, q2 = self.r_s(t), self.q1(t), self.q2(t)
        q1sq, q2sq = q1**2, q2**2
        rho0 = self.rho0(t)

        # Delta(ψ) = ψ(∞) - ψ(ξ)
        # This factors out the rho0 * r_s^2, moving it to the end
        def delta_psi_factor(s2: gt.FloatScalar) -> gt.FloatScalar:
            xi = xp.sqrt(self._ellipsoid_surface(q, q1, q2, s2)) / r_s
            return 2.0 / (1.0 + xi)

        def integrand(s: Float[Array, "N"]) -> Float[Array, "N *batch"]:
            s2 = s.reshape(s.shape + (1,) * (q.ndim - 2)) ** 2
            denom = xp.sqrt(((q1sq - 1) * s2 + 1) * ((q2sq - 1) * s2 + 1))
            return delta_psi_factor(s2) / denom

        # TODO: option to do integrate.quad
        integral = self._integrator(integrand)

        return (-2.0 * xp.pi * self.constants["G"] * rho0 * r_s**2 * q1 * q2) * integral

    # ==========================================================================

    @partial(jax.jit)
    def _density(
        self, q: gt.BatchQVec3, /, t: gt.BatchRealQScalar | gt.RealQScalar
    ) -> gt.BatchFloatQScalar:
        r_s, q1, q2 = self.r_s(t), self.q1(t), self.q2(t)
        xi = xp.sqrt(self._ellipsoid_surface(q, q1, q2, 1)) / r_s
        return self.rho0(t) / xi / (1.0 + xi) ** 2


# -------------------------------------------------------------------


class Vogelsberger08TriaxialNFWPotential(AbstractPotential):
    """Triaxial NFW Potential from DOI 10.1111/j.1365-2966.2007.12746.x."""

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r"""Scale mass."""
    # TODO: note the different definitions of m.

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale radius."""

    q1: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """y/x axis ratio.

    The z/x axis ratio is defined as :math:`q_2^2 = 3 - q_1^2`
    """

    a_r: AbstractParameter = ParameterField(
        dimensions="dimensionless",
        default=Quantity(1.0, ""),  # type: ignore[assignment]
    )
    """Transition radius relative to :math:`r_s`.

    :math:`r_a = a_r r_s  is a transition scale where the potential shape
    changes from ellipsoidal to near spherical.
    """

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    @partial(jax.jit, inline=True)
    def _r_e(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar
    ) -> gt.BatchFloatQScalar:
        q1sq = self.q1(t) ** 2
        q2sq = 3 - q1sq
        return xp.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 / q1sq + q[..., 2] ** 2 / q2sq)

    @partial(jax.jit, inline=True)
    def _r_tilde(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar
    ) -> gt.BatchFloatQScalar:
        r_a = self.a_r(t) * self.r_s(t)
        r_e = self._r_e(q, t)
        r = xp.linalg.vector_norm(q, axis=-1)
        return (r_a + r) * r_e / (r_a + r_e)

    @partial(jax.jit)
    def _potential(
        self: "Vogelsberger08TriaxialNFWPotential", q: gt.QVec3, t: gt.RealQScalar, /
    ) -> gt.FloatQScalar:
        r = self._r_tilde(q, t)
        return -self.constants["G"] * self.m(t) * xp.log(1.0 + r / self.r_s(t)) / r
