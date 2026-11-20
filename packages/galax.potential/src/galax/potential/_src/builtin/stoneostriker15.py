"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "StoneOstriker15Potential",
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

from typing import Final

import jax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.kernels import (
    poisson_laplacian,
    radial_gradient,
    radial_second_derivative,
    spherical_gradient,
    spherical_hessian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical, radial_derivatives


class StoneOstriker15Potential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Stone and Ostriker (2015) potential.

    A spherical model with a constant-density core of radius :math:`r_c`, a
    roughly isothermal zone with :math:`\rho \propto r^{-2}` for
    :math:`r_c \ll r \ll r_h`, and a density that falls as :math:`r^{-4}` beyond
    the halo radius :math:`r_h`, so the total mass :math:`m_{\mathrm{tot}}` is
    finite. [StoneOstriker2015]_ designed it for nearly isothermal star
    clusters and note that, because the rotation curve is nearly flat between
    the core and halo radii, it may also be useful for fitting galactic
    potentials. The three parameters are the total mass ``m_tot``, the core
    radius ``r_c`` and the halo radius ``r_h``.

    The density (Eq. 1 of [StoneOstriker2015]_) is

    .. math::

        \rho(r) = \frac{\rho_c}{\left(1 + r^2 / r_c^2\right)
                                \left(1 + r^2 / r_h^2\right)},
        \qquad
        \rho_c = \frac{m_{\mathrm{tot}} (r_h + r_c)}{2 \pi^2 r_c^2 r_h^2},

    where :math:`\rho_c` is the central density, written here in terms of the
    total mass by inverting Eq. 5 of the paper (:math:`m_{\mathrm{tot}} =
    2 \pi^2 r_c^2 r_h^2 \rho_c / (r_h + r_c)`). The enclosed mass (Eq. 4) is

    .. math::

        M(<r) = \frac{2 m_{\mathrm{tot}}}{\pi (r_h - r_c)}
                \left[ r_h \arctan\frac{r}{r_h} - r_c \arctan\frac{r}{r_c}
                \right],

    which tends to :math:`m_{\mathrm{tot}}` as :math:`r \to \infty`, and the
    potential (Eq. 3) is

    .. math::

        \Phi(r) = -\frac{2 G m_{\mathrm{tot}}}{\pi (r_h - r_c)}
            \left[ \frac{r_h}{r} \arctan\frac{r}{r_h}
                 - \frac{r_c}{r} \arctan\frac{r}{r_c}
                 + \frac{1}{2} \ln\frac{r^2 + r_h^2}{r^2 + r_c^2} \right],

    where :math:`G` is the gravitational constant. The circular speed is
    :math:`v_c^2 = G M(<r) / r`. At the centre the potential is finite,

    .. math::

        \Phi(0) = -\frac{2 G m_{\mathrm{tot}}}{\pi (r_h - r_c)}
                  \ln\frac{r_h}{r_c},

    so the escape speed there is :math:`v_{\mathrm{esc}}(0) = \sqrt{-2 \Phi(0)}`
    (Eq. 7). At large radii :math:`\Phi \to -G m_{\mathrm{tot}} / r`.

    Notes
    -----
    The paper states that :math:`r_h` approaches the half-mass radius when
    :math:`r_c \ll r_h`. The formulas divide by :math:`r_h - r_c`, so
    :math:`r_c = r_h` is excluded. The density, enclosed mass and potential are
    unchanged when :math:`r_c` and :math:`r_h` are exchanged, so the class does
    not check that :math:`r_c < r_h`. The model has a finite core and a finite
    mass, unlike the isothermal sphere. The paper reports that a closed form
    for the isotropic distribution function does not appear to exist.

    References
    ----------
    .. [StoneOstriker2015] Stone, N. C., & Ostriker, J. P. (2015). A Dynamical
       Potential-Density Pair for Star Clusters with Nearly Isothermal
       Interiors. The Astrophysical Journal Letters, 806, L28.
       `ADS <https://ui.adsabs.harvard.edu/abs/2015ApJ...806L..28S>`_;
       `doi:10.1088/2041-8205/806/2/L28 <https://doi.org/10.1088/2041-8205/806/2/L28>`_;
       `arXiv:1503.03479 <https://arxiv.org/abs/1503.03479>`_

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass")  # type: ignore[assignment]

    r_c: AbstractParameter = ParameterField(dimensions="length", doc="Core radius.")  # type: ignore[assignment]

    r_h: AbstractParameter = ParameterField(dimensions="length", doc="Halo radius.")  # type: ignore[assignment]

    # def __check_init__(self) -> None:
    #     _ = eqx.error_if(self.r_c, self.r_c.value >= self.r_h.value, "Core radius must be less than halo radius")   # noqa: E501, ERA001

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def rhoc_of_mtot(p: gt.Params, /) -> gt.FloatSz0:
    r"""Central density from the total mass.

    Solving S&O15 Eq. 5 for the central density,

    .. math::

        \rho_c = \frac{m_{\mathrm{tot}} (r_h + r_c)}{2 \pi^2 r_c^2 r_h^2}.

    """
    r_c, r_h = p["r_c"], p["r_h"]
    rho_c = p["m_tot"] * (r_h + r_c) / (2 * jnp.pi**2 * r_c**2 * r_h**2)
    return rho_c  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density function for Stone-Ostriker 2015 potential.

    S&O15 Eq. 1:

    .. math::

        \rho(r) = \frac{\rho_c}{(1 + r^2 / r_c^2)(1 + r^2 / r_h^2)}

    where :math:`\rho_c` is the central density given by `rhoc_of_mtot`.
    """
    rho_c = rhoc_of_mtot(p)
    core_term = 1 + (r / p["r_c"]) ** 2
    halo_term = 1 + (r / p["r_h"]) ** 2
    return rho_c / (core_term * halo_term)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass function for Stone-Ostriker 2015 potential.

    S&O15 Eq. 4, with the central density written in terms of the total mass:

    .. math::

        M(<r) = \frac{2 m_{\mathrm{tot}}}{\pi (r_h - r_c)}
                \left[ r_h \arctan\frac{r}{r_h} - r_c \arctan\frac{r}{r_c}
                \right]

    """
    r_c, r_h = p["r_c"], p["r_h"]
    A = 2 * p["m_tot"] / (jnp.pi * (r_h - r_c))
    return A * _atan_difference(p, r)  # type: ignore[no-any-return]


_ATAN_TAIL: Final = tuple((-1) ** k / (2 * k + 3) for k in range(16, -1, -1))
"""(y - atan(y)) / y^3 as a polynomial in y^2, highest degree first."""


def _x_minus_atan(x: gt.Sz0, atan_x: gt.Sz0, /) -> gt.FloatSz0:
    r"""$x - \tan^{-1}x$, to a few ulp for every $x \geq 0$, given $\tan^{-1}x$.

    Up to $x = 0.7$ the angle is halved first: with $q = 1 + \sqrt{1 + x^2}$
    and $y = x / q \leq 0.33$, $\tan^{-1}x = 2\tan^{-1}y$, so

    $$ x - \tan^{-1}x = \frac{x^3}{q^2} + 2(y - \tan^{-1}y), $$

    both terms positive, and $y - \tan^{-1}y = y^3 \sum_k (-1)^k y^{2k}/(2k+3)$
    needs 17 terms ($0.33^{34} \approx 10^{-16}$) where $x$ itself needed 52.
    Beyond, directly: it cancels by less than a factor 8.
    """
    xc = jnp.where(x <= 0.7, x, 0.7)
    q = 1 + jnp.sqrt(1 + xc**2)
    y = xc / q
    c = jnp.asarray(_ATAN_TAIL, dtype=y.dtype)
    series = jnp.polyval(c, y**2, unroll=len(_ATAN_TAIL))
    small = xc**3 / q**2 + 2 * y**3 * series
    return jnp.where(x <= 0.7, small, x - atan_x)  # type: ignore[no-any-return]


def _atan_difference(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""$r_h \tan^{-1}(r/r_h) - r_c \tan^{-1}(r/r_c)$, without cancellation.

    Directly, the two terms are both $\approx r$ for $r \ll r_c, r_h$. With
    $g(x) = x - \tan^{-1}x$ the same difference is
    $r_c g(r/r_c) - r_h g(r/r_h)$, whose terms instead both grow like $r$ far
    out. The two pairs of terms sum to $2r$, so taking whichever pair sums to
    less than $r$ never cancels by more than the ratio of $r_h$ and $r_c$.
    Both pairs share the two arctangents.
    """
    r_c, r_h = p["r_c"], p["r_h"]
    x_c, x_h = r / r_c, r / r_h
    a_c, a_h = jnp.atan(x_c), jnp.atan(x_h)
    g_c, g_h = r_c * _x_minus_atan(x_c, a_c), r_h * _x_minus_atan(x_h, a_h)
    direct = r_h * a_h - r_c * a_c
    return jnp.where(g_c + g_h < r, g_c - g_h, direct)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def dmass_dr(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""$dM/dr = 4\pi r^2 \rho$, for `radial_derivatives`.

    The derivative of S&O15 Eq. 4, with
    $(r/r_c)^2 - (r/r_h)^2 = (r/r_c)^2 (1 - (r_c/r_h)^2)$ so it does not cancel.
    """
    r_c, r_h = p["r_c"], p["r_h"]
    A = 2 * p["m_tot"] / (jnp.pi * (r_h - r_c))
    xc2, xh2 = (r / r_c) ** 2, (r / r_h) ** 2
    return A * xc2 * (1 - (r_c / r_h) ** 2) / ((1 + xc2) * (1 + xh2))  # type: ignore[no-any-return]


@ft.partial(jax.jit)
@radial_derivatives(mass_enclosed, dmass_dr)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential function for Stone-Ostriker 2015 potential.

    S&O15 Eq. 3, with the central density written in terms of the total mass:

    .. math::

        \Phi(r) = -\frac{2 G m_{\mathrm{tot}}}{\pi (r_h - r_c)}
            \left[ \frac{r_h}{r} \arctan\frac{r}{r_h}
                 - \frac{r_c}{r} \arctan\frac{r}{r_c}
                 + \frac{1}{2} \ln\frac{r^2 + r_h^2}{r^2 + r_c^2} \right]

    The log term is evaluated as :math:`\ln(1 + (r_h^2 - r_c^2)/(r^2 + r_c^2))`,
    scaled by :math:`r_h^2`: the ratio inside the log is :math:`\approx 1` far
    out, and :math:`r^2` overflows float32 in SI units.
    """
    r_c, r_h = p["r_c"], p["r_h"]
    A = -2 * p["G"] * p["m_tot"] / (jnp.pi * (r_h - r_c))
    atan_term = _atan_difference(p, r) / r
    k = r_c / r_h
    log_term = 0.5 * jnp.log1p((1 - k) * (1 + k) / ((r / r_h) ** 2 + k**2))
    return A * (atan_term + log_term)  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
