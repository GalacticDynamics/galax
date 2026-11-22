"""Private module that defines `gNFWPotential` and its functions.

The public API is `galax.potential.gnfw`.
"""

__all__ = [
    # class
    "gNFWPotential",
    # functions
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .base import rho0_of_m
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.jax import vectorize_method
from galax.potential._src.kernels import (
    poisson_laplacian,
    radial_second_derivative,
    spherical_gradient,
    spherical_hessian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import (
    incomplete_beta_zw,
    r_spherical,
    radial_derivatives,
    shell_dphi_dr,
)

DimL = u.dimension("length")
DimT = u.dimension("time")


@final
class gNFWPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Generalized NFW potential.

    The model is a spherical double power-law density profile with the NFW outer slope
    and a free inner slope. The density falls as :math:`r^{-\gamma}` inside the scale
    radius ``r_s`` and as :math:`r^{-3}` outside it, so the enclosed mass grows
    logarithmically without bound and the total mass is infinite. The parameter
    ``gamma`` is the inner slope: ``gamma = 0`` is a cored profile, ``gamma = 1`` is the
    NFW profile [NFW1996gNFW]_ and larger values are steeper cusps. The parameter ``m``
    is a mass scale, :math:`m = 4 \pi \rho_s r_s^3`, and it is not the total mass. The
    potential at the centre is finite for :math:`0 \le \gamma < 2`, which is the range
    these functions are written for.

    With :math:`x = r / r_s` the density is

    .. math::

        \rho(r) = \rho_s\, x^{-\gamma} (1 + x)^{\gamma - 3},
        \qquad \rho_s = \frac{m}{4 \pi r_s^3},

    the enclosed mass is

    .. math::

        M(<r) = m \int_0^x s^{2-\gamma} (1 + s)^{\gamma - 3}\, ds
              = m\, B_{x/(1+x)}(3 - \gamma, 0)
              = \frac{m\, x^{3-\gamma}}{3 - \gamma}\,
                {}_2F_1(3 - \gamma, 3 - \gamma; 4 - \gamma; -x),

    and the potential, with :math:`\Phi(r) \to 0` as :math:`r \to \infty`, is

    .. math::

        \Phi(r) = -G \left[ \frac{M(<r)}{r}
            + \frac{m}{r_s} B_{1/(1+x)}(1, 2 - \gamma) \right]
          = -\frac{G M(<r)}{r}
            - \frac{G m}{(2 - \gamma) r_s}
              \left[ 1 - \left(\frac{x}{1 + x}\right)^{2-\gamma} \right],

    where :math:`G` is the gravitational constant,
    :math:`B_z(a, b) = \int_0^z t^{a-1} (1 - t)^{b-1}\, dt` is the incomplete
    beta function and :math:`{}_2F_1` is the Gauss hypergeometric function. The
    central potential is :math:`\Phi(0) = -G m / [(2 - \gamma) r_s]`, and the
    circular velocity is :math:`v_c^2(r) = G M(<r) / r`.

    Notes
    -----
    This is the member :math:`(\alpha, \beta, \gamma) = (1, 3, \gamma)` of the
    double power-law family of [Zhao1996gNFW]_ (Eq. 1), and :math:`\gamma = 1` is the
    NFW profile of :class:`~galax.potential.NFWPotential`.

    For :math:`\gamma \ge 2` the central potential diverges.

    References
    ----------
    .. [Zhao1996gNFW] Zhao, H. (1996). Analytical models for galactic nuclei.
       Monthly Notices of the Royal Astronomical Society 278, 488.
       `ADS <https://ui.adsabs.harvard.edu/abs/1996MNRAS.278..488Z>`__;
       `doi:10.1093/mnras/278.2.488 <https://doi.org/10.1093/mnras/278.2.488>`__;
       `arXiv:astro-ph/9509122 <https://arxiv.org/abs/astro-ph/9509122>`__.
    .. [NFW1996gNFW] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1996). The
       Structure of Cold Dark Matter Halos. Astrophysical Journal 462, 563.
       `ADS <https://ui.adsabs.harvard.edu/abs/1996ApJ...462..563N>`__;
       `doi:10.1086/177173 <https://doi.org/10.1086/177173>`__;
       `arXiv:astro-ph/9508025 <https://arxiv.org/abs/astro-ph/9508025>`__.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    A gNFW potential matches the NFW potential when :math:`\gamma = 1`:

    >>> gnfw = gp.gNFWPotential(m=1e12,r_s=1,gamma=1, units="galactic")
    >>> nfw = gp.NFWPotential(m=1e12,r_s=1, units="galactic")

    >>> x, t = jnp.array([8, 0, 0]), 0
    >>> gnfw.potential(x, t)
    Array(-1.23552744, dtype=float64)

    >>> jnp.isclose(gnfw.potential(x, t), nfw.potential(x, t), atol=1e-8)
    Array(True, dtype=bool)

    Both are finite at the origin, where :math:`\Phi(0) = -G m / r_s`:

    >>> x = jnp.array([0, 0, 0])
    >>> gnfw.potential(x, t), nfw.potential(x, t)
    (Array(-4.49850215, dtype=float64), Array(-4.49850215, dtype=float64))

    The gNFW potential is a generalization of the NFW potential, so it can be
    used to model a wider range of profiles. For example, if we set
    :math:`\gamma = 0.5`, we get a density profile that is shallower than the
    NFW profile at small radii, so the potential is less deep at this radius:

    >>> gnfw2 = gp.gNFWPotential(m=1e12,r_s=1,gamma=0.5, units="galactic")

    >>> x = jnp.array([8, 0, 0])
    >>> gnfw2.potential(x, t)
    Array(-1.09363913, dtype=float64)

    >>> gnfw2.potential(x, t) > gnfw.potential(x, t)
    Array(True, dtype=bool)

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc="""Scale mass. This is NOT the total mass.""",
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]
    gamma: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc="Slope of the density profile at small radii.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _density(  # TODO: inputs w/ units
        self, xyz: gt.Sz3, t: gt.Sz0, /
    ) -> gt.Sz0:
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.Sz3, t: gt.Sz0, /
    ) -> gt.Sz0:
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @vectorize_method(signature="(3),()->(3)")
    @ft.partial(jax.jit)
    def _gradient(  # TODO: inputs w/ units
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0 | gt.Sz0, /
    ) -> gt.FloatSz3:
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        params = self.params(t)
        return gradient(params, xyz)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Density profile for the gNFW model.

    .. math::

        \rho(r) = \rho_s \left(\frac{r}{r_s}\right)^{-\gamma}
            \left(1 + \frac{r}{r_s}\right)^{\gamma-3}

    where :math:`\rho_s` is the characteristic density, :math:`r_s` is the
    scale radius, and :math:`\gamma` is the inner slope. The parameter is not
    :math:`\rho_s` but a characteristic mass :math:`m`, defined as

    .. math::

        m = 4 \pi \rho_s r_s^3.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    A gNFW potential matches the NFW potential when :math:`\gamma = 1`:

    >>> gnfw = gp.gNFWPotential(m=1e12,r_s=1,gamma=1, units="galactic")
    >>> nfw = gp.NFWPotential(m=1e12,r_s=1, units="galactic")

    >>> x, t = jnp.array([8, 0, 0]), 0
    >>> gnfw.density(x, t)
    Array(1.2280474e+08, dtype=float64)

    >>> jnp.isclose(gnfw.density(x, t), nfw.density(x, t), atol=1e-8)
    Array(True, dtype=bool)

    >>> x = jnp.array([0, 0, 0])
    >>> jnp.isclose(gnfw.density(x, t), nfw.density(x, t), atol=1e-8)
    Array(True, dtype=bool)

    The gNFW potential is a generalization of the NFW potential, so it can be
    used to model a wider range of density profiles. For example, if we set
    :math:`\gamma = 0.5`, we get a density profile that is shallower than the
    NFW profile at small radii, and with the same ``m`` the density at this
    radius is lower:

    >>> gnfw2 = gp.gNFWPotential(m=1e12,r_s=1,gamma=0.5, units="galactic")

    >>> x = jnp.array([8, 0, 0])
    >>> gnfw2.density(x, t)
    Array(1.15781419e+08, dtype=float64)

    >>> gnfw2.density(x, t) < gnfw.density(x, t)
    Array(True, dtype=bool)

    """
    x = r / p["r_s"]
    rho0: gt.Sz0 = rho0_of_m(p)
    return rho0 * jnp.power(x, -p["gamma"]) * jnp.power(1 + x, p["gamma"] - 3)


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Enclosed mass for the gNFW model.

    .. math::

        M(<r) = m\, B_{x/(1+x)}(3-\gamma, 0)

    where :math:`x = r / r_s` is the dimensionless radius, :math:`m` is the
    characteristic mass, and :math:`\gamma` is the inner slope. :math:`B_z(a,
    b)` is the incomplete beta function.

    Note that ``jax.scipy.special.betainc`` is the regularized incomplete beta
    function, which is related to the incomplete beta function by

    .. math::

        B_z(a, b) = B(a, b) \cdot \text{betainc}(a, b, z).

    However, :math:`B(a, b)` diverges for :math:`b = 0`, so a hypergeometric
    function is used instead.

    """
    x = r / p["r_s"]
    # z and 1 - z both exact: far out, z rounds to 1 and loses 1 - z.
    z, w = x / (1 + x), 1 / (1 + x)
    a = 3.0 - p["gamma"]
    return p["m"] * incomplete_beta_zw(a, jnp.zeros_like(a), z, w)


@ft.partial(jax.jit)
def dmass_dr(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r""":math:`dM/dr = 4\pi r^2 \rho = m x^{2-\gamma} (1+x)^{\gamma-3} / r_s`."""
    x = r / p["r_s"]
    g = p["gamma"]
    return p["m"] / p["r_s"] * x ** (2 - g) * (1 + x) ** (g - 3)


# -----------------------------------------------


@ft.partial(jax.jit)
@radial_derivatives(mass_enclosed, dmass_dr)
def potential(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Potential for the gNFW model.

    With :math:`x = r / r_s`,

    .. math::

        \Phi(r) = -G \left[ \frac{M(<r)}{r}
            + \frac{m}{r_s} B_{1/(1+x)}(1, 2 - \gamma) \right]

    where :math:`M(<r)` is the enclosed mass, :math:`m` is the characteristic
    mass, :math:`B_z(a, b)` is the incomplete beta function and
    :math:`B_w(1, b) = [1 - (1 - w)^b] / b`, or :math:`-\ln(1 - w)` for
    :math:`b = 0`. The potential vanishes at infinity.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    A gNFW potential matches the NFW potential when :math:`\gamma = 1`:

    >>> gnfw = gp.gNFWPotential(m=1e12,r_s=1,gamma=1, units="galactic")
    >>> nfw = gp.NFWPotential(m=1e12,r_s=1, units="galactic")

    >>> x, t = jnp.array([8, 0, 0]), 0
    >>> gnfw.potential(x, t)
    Array(-1.23552744, dtype=float64)

    >>> jnp.isclose(gnfw.potential(x, t), nfw.potential(x, t), atol=1e-8)
    Array(True, dtype=bool)

    Both are finite at the origin, where :math:`\Phi(0) = -G m / r_s`:

    >>> x = jnp.array([0, 0, 0])
    >>> gnfw.potential(x, t), nfw.potential(x, t)
    (Array(-4.49850215, dtype=float64), Array(-4.49850215, dtype=float64))

    The gNFW potential is a generalization of the NFW potential, so it can be
    used to model a wider range of profiles. For example, if we set
    :math:`\gamma = 0.5`, we get a density profile that is shallower than the
    NFW profile at small radii, so the potential is less deep at this radius:

    >>> gnfw2 = gp.gNFWPotential(m=1e12,r_s=1,gamma=0.5, units="galactic")

    >>> x = jnp.array([8, 0, 0])
    >>> gnfw2.potential(x, t)
    Array(-1.09363913, dtype=float64)

    >>> gnfw2.potential(x, t) > gnfw.potential(x, t)
    Array(True, dtype=bool)

    """
    rs = p["r_s"]

    # Inner term from integral 0 to r
    inner = mass_enclosed(p, r) / r

    # Outer term from integral r to infinity: B_w(1, b) for w = 1/(1+x), i.e.
    # (1 - (1-w)^b) / b, or -ln(1-w) at b = 0. ln(1-w) = -ln(1 + 1/x) is taken
    # directly, since 1 - w loses its digits as x -> 0.
    b = 2.0 - p["gamma"]
    log1mw = -jnp.log1p(rs / r)
    safe_b = jnp.where(b == 0, jnp.ones_like(b), b)  # avoid 0/0 below
    outer = (p["m"] / rs) * jnp.where(
        b == 0, -log1mw, -jnp.expm1(safe_b * log1mw) / safe_b
    )

    return -p["G"] * (inner + outer)  # type: ignore[no-any-return]


dpotential_dr = shell_dphi_dr(mass_enclosed, dmass_dr)
r""":math:`d\Phi/dr = G M(<r) / r^2`, whose :math:`r`-derivative is exact.

A ``jax.custom_jvp`` from ``galax.potential._src.utils.shell_dphi_dr``: its
derivative is :math:`G M'(r)/r^2 - 2 G M(r)/r^3`, not autodiff of
the series in :func:`mass_enclosed`, so :func:`gradient` linearizes to the exact
Hessian.
"""


_spherical_gradient = spherical_gradient(dpotential_dr)


@ft.partial(jax.jit)
def gradient(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtSz3:
    r"""Gradient of the potential for the gNFW model.

    .. math::

        \nabla \Phi(r) = \frac{G M(<r)}{r^2} \hat{r}

    where :math:`M(<r)` is the enclosed mass, by :func:`dpotential_dr`: its jvp is
    the exact Hessian away from the origin, and the gradient is exactly 0 at the
    origin, as for the other spherical modules.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    A gNFW potential matches the NFW potential when :math:`\gamma = 1`:

    >>> gnfw = gp.gNFWPotential(m=1e12, r_s=1, gamma=1, units="galactic")
    >>> nfw = gp.NFWPotential(m=1e12, r_s=1, units="galactic")

    >>> x, t = jnp.array([8, 0, 0]), 0
    >>> gnfw.gradient(x, t)
    Array([0.09196173, 0.        , 0.        ], dtype=float64)

    >>> jnp.allclose(gnfw.gradient(x, t), nfw.gradient(x, t), atol=1e-8)
    Array(True, dtype=bool)

    """
    return _spherical_gradient(p, xyz)


# Derived quantities, built from the kernels above. `dpotential_dr` and
# `gradient` are defined above.
d2potential_dr2 = radial_second_derivative(dpotential_dr)
laplacian = poisson_laplacian(density)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
