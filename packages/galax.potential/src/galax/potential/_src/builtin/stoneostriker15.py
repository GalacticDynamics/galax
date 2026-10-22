"""galax: Galactic Dynamix in Jax."""

__all__ = ["StoneOstriker15Potential"]

import functools as ft

import jax

import quaxed.numpy as jnp
import unxt as u

import galax.potential.custom_types as gt
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical, radial_derivatives


class StoneOstriker15Potential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Potential from Stone and Ostriker 2015.

    http://dx.doi.org/10.1088/2041-8205/806/2/L28.

    The density profile is given by S&O15 Eq. 1:

    $$ \rho(r) = \frac{\rho_c}{\left(1 + r^2 / r_c^2\right)\left(1 + r^2 /
    r_h^2\right)}. $$

    where we will use later definitions to write $\rho_c$ in terms of total
    $M_{tot}$, core radius $r_c$, and halo radius $r_h$.

    The potential is given by S&O15 Eq. 3:

    $$ \Phi = -\frac{2 G M_{tot}}{\pi (r_h - r_c)}
        \left(
              \frac{r_h}{r} \atan(r/r_h)
            - \frac{r_c}{r} \atan(r/r_c)
            + \frac{1}{2} \log(\frac{r^2 + r_h^2}{r^2 + r_c^2})
        \right)
    $$

    The enclosed mass is given by S&O15 Eq. 4:

    $$ M(<r) = \frac{2 M_{tot}}{\pi (r_h-r_c)} (r_h \atan(r/r_h) - r_c \atan(r/r_c)) $$

    The total mass is given by S&O15 Eq. 5:

    $$ M_{tot} = \frac{2\pi^2 r_c^2 r_h^2 \rho_c}{r_h + r_c}. $$

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
        t = u.Q.from_(t, self.units["time"])

        params = {
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_h": self.r_h(t, ustrip=ul),
            "r_c": self.r_c(t, ustrip=ul),
        }
        return density(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)
        t = u.Q.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_h": self.r_h(t, ustrip=ul),
            "r_c": self.r_c(t, ustrip=ul),
        }
        return potential(params, r)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def rhoc_of_mtot(p: gt.Params, /) -> gt.FloatSz0:
    r"""Density from the mass.

    Solving from S&O15 Eq. 5:

    $$ \rho_c = \frac{ M_{tot} (r_h + r_c)}{2\pi^2 r_c^2 r_h^2 } $$

    """
    r_c, r_h = p["r_c"], p["r_h"]
    rho_c = p["m_tot"] * (r_h + r_c) / (2 * jnp.pi**2 * r_c**2 * r_h**2)
    return rho_c  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density function for Stone-Ostriker 2015 potential.

    S&O15 Eq. 1:

    $$ \rho(r) = \frac{\rho_c}{(1 + r^2 / r_c^2)(1 + r^2 /
    r_h^2)} $$

    where $\rho_c$ is given by the function `rhoc_of_mtot`.
    """
    rho_c = rhoc_of_mtot(p)
    core_term = 1 + (r / p["r_c"]) ** 2
    halo_term = 1 + (r / p["r_h"]) ** 2
    return rho_c / (core_term * halo_term)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass function for Stone-Ostriker 2015 potential.

    S&O15 Eq. 4:

    $$ M(<r) = \frac{2 M_{tot}}{\pi (r_h-r_c)} (r_h \atan(r/r_h) - r_c \atan(r/r_c)) $$

    """
    r_c, r_h = p["r_c"], p["r_h"]
    A = 2 * p["m_tot"] / (jnp.pi * (r_h - r_c))
    return A * _atan_difference(p, r)  # type: ignore[no-any-return]


_ATAN_TAIL = tuple((-1) ** k / (2 * k + 3) for k in range(16, -1, -1))
"""(y - atan(y)) / y^3 as a polynomial in y^2, highest degree first."""


def _x_minus_atan(x: gt.Sz0, /) -> gt.FloatSz0:
    r"""$x - \tan^{-1}x$, to a few ulp for every $x \geq 0$.

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
    return jnp.where(x <= 0.7, small, x - jnp.atan(x))  # type: ignore[no-any-return]


def _atan_difference(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""$r_h \tan^{-1}(r/r_h) - r_c \tan^{-1}(r/r_c)$, without cancellation.

    Directly, the two terms are both $\approx r$ for $r \ll r_c, r_h$. With
    $g(x) = x - \tan^{-1}x$ the same difference is
    $r_c g(r/r_c) - r_h g(r/r_h)$, whose terms instead both grow like $r$ far
    out. The two pairs of terms sum to $2r$, so taking whichever pair sums to
    less than $r$ never cancels by more than the ratio of $r_h$ and $r_c$.
    """
    r_c, r_h = p["r_c"], p["r_h"]
    g_c, g_h = r_c * _x_minus_atan(r / r_c), r_h * _x_minus_atan(r / r_h)
    direct = r_h * jnp.atan2(r, r_h) - r_c * jnp.atan2(r, r_c)
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

    The log term of S&O15 Eq. 3 is evaluated as
    $\ln(1 + (r_h^2 - r_c^2)/(r^2 + r_c^2))$, scaled by $r_h^2$: the ratio
    inside the log is $\approx 1$ far out, and $r^2$ overflows float32 in SI
    units.
    """
    r_c, r_h = p["r_c"], p["r_h"]
    A = -2 * p["G"] * p["m_tot"] / (jnp.pi * (r_h - r_c))
    atan_term = _atan_difference(p, r) / r
    k = r_c / r_h
    log_term = 0.5 * jnp.log1p((1 - k) * (1 + k) / ((r / r_h) ** 2 + k**2))
    return A * (atan_term + log_term)  # type: ignore[no-any-return]
