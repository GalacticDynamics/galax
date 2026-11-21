"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "PowerLawCutoffPotential",
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
import math
from dataclasses import KW_ONLY

from typing import Final, final

import equinox as eqx
import jax
import numpy as np
from scipy.special import zeta

import quaxed.numpy as jnp
import quaxed.scipy.special as jsp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
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

X_SWITCH: Final = 2.0
"""$x$ where `gamma_upper` changes from its series to its continued fraction."""
CF_TERMS: Final = 60
"""Terms of the continued fraction: $10^{-17}$ at $x = 2$, for $-1/2 < b \\le 1$."""
SERIES_TERMS: Final = 25
"""Terms of the series: $2^{25}/25! \\approx 10^{-18}$."""
B_SMALL: Final = 0.2
"""$|b|$ below which `lgamma1p_over_b` uses its Taylor series."""
LGAMMA1P_COEFFS: Final = np.array(
    [(-1.0) ** k * zeta(k) / k for k in range(25, 1, -1)] + [-np.euler_gamma]
)
r"""Taylor coefficients, highest power first, of $\ln\Gamma(1+b)/b$.

That is $-\gamma_E + \sum_{k\ge2}(-1)^k\zeta(k)b^{k-1}/k$; at $|b| = 0.2$ the last
term is $\sim 10^{-18}$.
"""


def exprel(t: gt.SzN, /) -> gt.SzN:
    r"""$(e^t - 1)/t$, with its removable singularity at $t = 0$ expanded."""
    small = jnp.abs(t) < 1e-2
    series = jnp.polyval(
        jnp.asarray([1 / math.factorial(n + 1) for n in range(6, -1, -1)]), t
    )
    t_safe = jnp.where(small, 1.0, t)
    return jnp.where(small, series, jnp.expm1(t_safe) / t_safe)  # type: ignore[no-any-return]


def lgamma1p_over_b(b: gt.SzN, /) -> gt.SzN:
    r"""$\ln\Gamma(1+b)/b$, finite at $b = 0$ where it is $-\gamma_E$.

    $1 + b$ keeps only the digits of $b$ that fit beside the 1, so near $b = 0$
    the direct quotient loses them all; the Taylor series does not.
    """
    small = jnp.abs(b) < B_SMALL
    b_safe = jnp.where(small, 1.0, b)
    return jnp.where(  # type: ignore[no-any-return]
        small, jnp.polyval(LGAMMA1P_COEFFS, b), jsp.gammaln(1 + b_safe) / b_safe
    )


@ft.partial(jax.jit)
def gamma_upper(b: gt.Sz0, x: gt.SzN) -> gt.SzN:
    r"""Upper incomplete gamma $\Gamma(b, x) = \int_x^\infty u^{b-1}e^{-u}du$.

    For $x > 0$ and any real $b > -1$, including $b \le 0$ where
    `jax.scipy.special.gammaincc` is undefined. It is analytic in $b$, but the
    usual form $\Gamma(b) - \gamma(b, x)$ is a difference of two terms that both
    have a pole at $b = 0$.

    For $x \le 2$,

    .. math::

        \Gamma(b, x) = \frac{\Gamma(1+b) - 1}{b} - \frac{x^b - 1}{b}
            - x^b \sum_{k\ge1} \frac{(-x)^k}{k!\,(b+k)},

    which is the series of $\gamma(b, x)$ with the $k = 0$ term, $x^b/b$, merged
    into $\Gamma(b)$. Both quotients are $\mathrm{expm1}(t)/b$ with $t \propto b$,
    so they are the smooth $\ell\,\mathrm{exprel}(b\ell)$ with
    $\ell = \ln\Gamma(1+b)/b$ resp. $\ln x$, with no cancellation as
    $b \to 0$ (the $x^b$ factor is $\exp(b \ln x)$). For $x > 2$ the Legendre
    continued fraction, which is positive and stable for every $b \le 1$,

    .. math::

        \Gamma(b, x) = x^b e^{-x} \cfrac{1}{x + 1 - b - \cfrac{1(1-b)}{x + 3 - b
            - \cfrac{2(2-b)}{x + 5 - b - \cdots}}}.

    Each form sees only its own $x$ (the other is evaluated at the switch), so the
    unused one cannot put a NaN into a gradient.
    """
    small = x <= X_SWITCH
    xs = jnp.where(small, x, X_SWITCH)
    xl = jnp.where(small, X_SWITCH, x)

    # The series, for x at most the switch.
    ln_x = jnp.log(xs)
    ell = lgamma1p_over_b(b)
    head = ell * exprel(b * ell) - ln_x * exprel(b * ln_x)
    c = jnp.stack(
        [1.0 / (float(math.factorial(k)) * (b + k)) for k in range(SERIES_TERMS, 0, -1)]
    )
    tail = -xs * jnp.polyval(c, -xs, unroll=SERIES_TERMS)
    series = head - jnp.exp(b * ln_x) * tail

    # The continued fraction, evaluated from the tail up.
    def step(i: int, t: gt.SzN) -> gt.SzN:
        n = CF_TERMS - 1 - i
        return xl + 2 * n + 1 - b - (n + 1) * (n + 1 - b) / t  # type: ignore[no-any-return]

    t = jax.lax.fori_loop(0, CF_TERMS, step, xl + 2 * CF_TERMS + 1 - b)
    cf = jnp.exp(b * jnp.log(xl) - xl) / t

    return jnp.where(small, series, cf)  # type: ignore[no-any-return]


@final
class PowerLawCutoffPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Power-law density profile with an exponential cutoff.

    A spherical density that falls as :math:`r^{-\alpha}` at small radii and is cut off
    by a Gaussian factor beyond the cutoff radius ``r_c``. The mass is finite and equal
    to the parameter ``m_tot``. For :math:`\alpha < 3` the enclosed mass converges at
    the centre, and the central density is finite only for :math:`\alpha = 0`. The form
    is used for the bulge of the Milky Way model ``MWPotential2014`` in galpy
    [Bovy2015PowerLawCutoff]_. The model is defined by the density below, and the
    potential follows from Poisson's equation.

    With :math:`s = r / r_c` and :math:`a = (3 - \alpha) / 2` the density is

    .. math::

        \rho(r) = \frac{m_{\mathrm{tot}}}{2 \pi \Gamma(a) r_c^3}
            \left(\frac{r_c}{r}\right)^{\alpha} e^{-s^2},

    the enclosed mass is

    .. math::

        M(<r) = m_{\mathrm{tot}}\, P(a, s^2),

    and the potential, with :math:`\Phi(r) \to 0` as :math:`r \to \infty`, is

    .. math::

        \Phi(r) = -\frac{G\, M(<r)}{r}
            - \frac{G\, m_{\mathrm{tot}}}{r_c}
              \frac{\Gamma(1 - \alpha/2,\, s^2)}{\Gamma(a)},

    where :math:`G` is the gravitational constant, :math:`\Gamma(z)` is the
    gamma function, :math:`P(a, x) = \gamma(a, x) / \Gamma(a)` is the
    regularized lower incomplete gamma function and
    :math:`\Gamma(z, x) = \int_x^\infty t^{z-1} e^{-t}\, dt` is the upper
    incomplete gamma function. The total mass is :math:`M(<\infty) =
    m_{\mathrm{tot}}`, the circular velocity is :math:`v_c^2(r) = G M(<r) / r`,
    and for :math:`\alpha < 2` the central potential is finite,

    .. math::

        \Phi(0) = -\frac{G\, m_{\mathrm{tot}}}{r_c}
            \frac{\Gamma(1 - \alpha/2)}{\Gamma(a)}.

    Parameters
    ----------
    m_tot : :class:`~unxt.Quantity`\ [mass]
        Total mass.
    alpha : :class:`~unxt.Quantity`\ [dimensionless]
        Power law index. Must satisfy: ``0 <= alpha < 3``.
    r_c : :class:`~unxt.Quantity`\ [length]
        Cutoff radius.

    Notes
    -----
    The potential is the sum of the enclosed-mass term and the outer shell, which
    is evaluated with the upper incomplete gamma function of order
    :math:`1 - \alpha/2`. It is finite for the whole range
    :math:`0 \le \alpha < 3`, including :math:`\alpha = 2`. The density and the
    enclosed mass are valid on the same range.

    The parameters (total mass, power-law index, cutoff radius) are those of Gala's
    ``PowerLawCutoffPotential`` [PriceWhelan2017PowerLawCutoff]_. The functional form
    :math:`\rho \propto r^{-\alpha} \exp[-(r / r_c)^2]` is the one of galpy's
    ``PowerSphericalPotentialwCutoff``.

    References
    ----------
    .. [Bovy2015PowerLawCutoff] Bovy, J. (2015). galpy: A Python Library for
       Galactic Dynamics. Astrophysical Journal Supplement Series 216, 29.
       `ADS <https://ui.adsabs.harvard.edu/abs/2015ApJS..216...29B>`__;
       `doi:10.1088/0067-0049/216/2/29 <https://doi.org/10.1088/0067-0049/216/2/29>`__;
       `arXiv:1412.3451 <https://arxiv.org/abs/1412.3451>`__.
    .. [PriceWhelan2017PowerLawCutoff] Price-Whelan, A. M. (2017). Gala: A
       Python package for galactic dynamics. Journal of Open Source Software
       2(18), 388.
       `ADS <https://ui.adsabs.harvard.edu/abs/2017JOSS....2..388P>`__;
       `doi:10.21105/joss.00388 <https://doi.org/10.21105/joss.00388>`__.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]
    """Total mass of the potential."""

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc="Power law index. Must satisfy: ``0 <= alpha < 3``",
    )

    r_c: AbstractParameter = ParameterField(dimensions="length", doc="Cutoff radius.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass, :math:`M(<r) = m_{\rm tot} P((3-\alpha)/2, (r/r_c)^2)`.

    :math:`P` is the regularized lower incomplete gamma function.

    For :math:`x = (r/r_c)^2 \leq 1` by :math:`P(a, x) = x^a/\Gamma(a) \sum_k
    (-x)^k/(k!(a+k))` (20 terms reach :math:`1/20! \approx 10^{-19}`):
    ``jax.scipy.special.gammainc`` forms :math:`x^a e^{-x}` as
    :math:`\exp(a\ln x - x)`, which multiplies the rounding of :math:`a \ln x` by
    :math:`|a \ln x|`, about 20 ulp at :math:`x = 10^{-16}`.
    """
    a = 1.5 - p["alpha"] / 2
    x = (r / p["r_c"]) ** 2
    xc = jnp.where(x <= 1, x, 1.0)
    c = jnp.stack(
        [1.0 / (float(math.factorial(k)) * (a + k)) for k in range(19, -1, -1)]
    )
    series = jnp.polyval(c, -xc, unroll=20)
    small = xc**a / jsp.gamma(a) * series
    return p["m_tot"] * jnp.where(x <= 1, small, jsp.gammainc(a, x))  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def dmass_dr(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""$dM/dr = 4\pi r^2\rho = 2M s^{2-\alpha} e^{-s^2} / (\Gamma(a) r_c)$.

    With $s = r/r_c$ and $a = (3-\alpha)/2$.
    """
    alpha, r_c = p["alpha"], p["r_c"]
    s = r / r_c
    norm = 2 * p["m_tot"] / (jsp.gamma(1.5 - alpha / 2) * r_c)
    return norm * s ** (2 - alpha) * jnp.exp(-(s**2))  # type: ignore[no-any-return]


@ft.partial(jax.jit)
@radial_derivatives(mass_enclosed, dmass_dr)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential for the power-law cutoff density profile, with $\Phi(\infty) = 0$.

    With :math:`s = r / r_c`, :math:`a = (3 - \alpha)/2`, :math:`b = a - 1/2`
    and :math:`x = s^2`, the potential is the sum of the enclosed-mass term and
    the outer shell,

    .. math::

        \Phi(r) = -\frac{G M(<r)}{r}
            - \frac{G m_{\mathrm{tot}}}{r_c}
              \frac{\Gamma(b,\, x)}{\Gamma(a)}

    where :math:`M(<r) = m_{\mathrm{tot}} P(a, x)`, :math:`P` is the
    regularized lower incomplete gamma function and :math:`\Gamma(b, x)` the
    upper one. The outer shell is :math:`4\pi G \int_r^\infty \rho\, s\, ds`.
    Both terms are negative, so they do not cancel.

    The older form, :math:`\gamma(b, x) - \Gamma(b)` as in Gala, has
    :math:`\Gamma(b) = \pm\infty` or NaN once :math:`b \le 0`, that is for
    :math:`\alpha \ge 2`. ``gamma_upper`` evaluates :math:`\Gamma(b, x)`
    directly and is good through :math:`b = 0` (:math:`\alpha = 2`, where
    :math:`\Gamma(0, x) = E_1(x)`), so the potential is finite for the whole
    range :math:`0 \le \alpha < 3`.
    """
    # At r = 0, M(<r)/r is 0/0 and ln x is -inf. Phi(0) is finite for alpha < 2
    # and infinite above, so floor r where r^2 is still a normal float
    # (~1e-154 r_c in float64): the value there is Phi(0) to round-off for
    # alpha well below 2, and for alpha >= 2 a large finite number.
    r = jnp.maximum(r, 2 * jnp.sqrt(jnp.finfo(r.dtype).tiny) * p["r_c"])
    a = 1.5 - p["alpha"] / 2
    x = (r / p["r_c"]) ** 2
    shell = gamma_upper(a - 0.5, x) / (p["r_c"] * jsp.gamma(a))
    return -p["G"] * (mass_enclosed(p, r) / r + p["m_tot"] * shell)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density for the power-law cutoff profile that defines this model.

    .. math::

        \rho(r) = \frac{m_{\mathrm{tot}}}{2\pi \Gamma((3-\alpha)/2) r_c^3}
            \left(\frac{r_c}{r}\right)^\alpha \exp\left[-(r / r_c)^2\right]

    where :math:`m_{\mathrm{tot}}` is the total mass, :math:`\alpha` is the
    power-law index and :math:`r_c` is the cutoff radius.

    """
    alpha, r_c, m_tot = p["alpha"], p["r_c"], p["m_tot"]
    norm = m_tot / (2 * jnp.pi * jsp.gamma(1.5 - alpha / 2) * r_c**3)
    _result = norm * (r_c / r) ** alpha * jnp.exp(-((r / r_c) ** 2))
    return _result  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
