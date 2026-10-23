"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "GaussianPotential",
    # functions
    "rho0_of_m",
    "m_of_rho0",
    "density",
    "mass_enclosed",
    "potential",
]

import functools as ft
import math
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax

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
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical, radial_derivatives


@final
class GaussianPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Gaussian Potential.

    A spherical mass distribution with a Gaussian density profile.

    The density profile is given by:

    $$ \rho(r) = \frac{ m_\mathrm{tot} }{ (2 \pi)^{3/2} r_s^3 }
    \exp\left(-\frac{r^2}{2 r_s^2}\right) $$

    where :math:`m_\mathrm{tot}` is the total mass and :math:`r_s` is the
    scale radius. Unlike the NFW profile, the Gaussian profile has a finite
    total mass, so :math:`m_\mathrm{tot}` is the total (rather than a
    characteristic) mass.

    Solving Poisson's equation gives the gravitational potential

    $$ \Phi(r) = -\frac{G m_\mathrm{tot}}{r}
    \mathrm{erf}\left(\frac{r}{\sqrt{2} r_s}\right) $$

    Which has the expected behavior of being finite at the center and going to
    the Kepler potential :math:`-G m_\mathrm{tot} / r` at large radii.

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius of the potential."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        r"""Potential energy.

        .. math::

            \Phi(r) = -\frac{G m_\mathrm{tot}}{r} \mathrm{erf}\left(\frac{r}{\sqrt{2}
            r_s}\right)
        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Density.

        .. math::

            \rho_0 = \frac{m_\mathrm{tot}}{(2 \pi)^{3/2} r_s^3}
            \rho(r) = \rho_0 \exp\left(-\frac{r^2}{2 r_s^2}\right)

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _mass_enclosed(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Enclosed mass.

        .. math::

            M(<r) = m_\mathrm{tot} \left[ \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right) -
            \sqrt{\frac{2}{\pi}} x \exp\left(-\frac{x^2}{2}\right) \right]

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.GaussianPotential(m_tot=1e11, r_s=15, units="galactic")

        >>> q = u.Q([10, 0, 0], "kpc")
        >>> t = u.Q(0, "Gyr")
        >>> pot._mass_enclosed(q, t)
        Array(6.90842509e+09, dtype=float64)

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return mass_enclosed(params, r)  # type: ignore[no-any-return]


# =============================================================


# -----------------------------------------------


@ft.partial(jax.jit)
def rho0_of_m(p: gt.Params, /) -> gt.Sz0:
    r"""Central density for the Gaussian model.

    The Gaussian profile is parametrized by a total mass $m_\mathrm{tot}$ and
    a scale radius $r_s$. The central density is given by

    $$ \rho_0 = \frac{m_\mathrm{tot}}{(2 \pi)^{3/2} r_s^3}. $$

    Examples
    --------
    >>> import jax.numpy as jnp

    A quick sanity check:

    >>> rho0_of_m({"m_tot": 1.0, "r_s": 1.0}) - 1 / (2 * jnp.pi) ** 1.5
    Array(0., dtype=float64, weak_type=True)

    """
    return p["m_tot"] / ((2 * jnp.pi) ** 1.5 * p["r_s"] ** 3)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def m_of_rho0(p: gt.Params, /) -> gt.Sz0:
    r"""Total mass for the Gaussian model.

    The Gaussian profile is parametrized by a total mass $m_\mathrm{tot}$ and
    a scale radius $r_s$. The total mass is given by

    $$ m_\mathrm{tot} = (2 \pi)^{3/2} \rho_0 r_s^3. $$

    Examples
    --------
    >>> import jax.numpy as jnp

    A quick sanity check:
    >>> m_of_rho0({"rho0": 1.0, "r_s": 1.0}) - (2 * jnp.pi) ** 1.5
    Array(0., dtype=float64, weak_type=True)

    """
    return (2 * jnp.pi) ** 1.5 * p["rho0"] * p["r_s"] ** 3  # type: ignore[no-any-return]


# -----------------------------------------------


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Density profile for the Gaussian model.

    $$ \rho(r) = \rho_0 \exp\left(-\frac{r^2}{2 r_s^2}\right) $$

    where $\rho_0$ is the central density and $r_s$ is the scale radius. We
    actually parametrize not with $\rho_0$ but by the total mass $m$, defined
    as

    $$ m = (2 \pi)^{3/2} \rho_0 r_s^3. $$

    """
    x = r / p["r_s"]
    rho0: gt.Sz0 = rho0_of_m(p)
    return rho0 * jnp.exp(-(x**2) / 2)  # type: ignore[no-any-return]


def _x_exp(x: gt.BBtSz0, /) -> gt.BBtSz0:
    r"""$x e^{-x^2/2}$, shared by `mass_enclosed` and `dmass_dr`.

    Further factors of $x$ multiply this, so they give 0 rather than
    $\infty \cdot 0$ where both a power of $x$ overflows and the exponential
    underflows.
    """
    return x * jnp.exp(-(x**2) / 2)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Enclosed mass for the Gaussian model.

    $$ M(<r) = m_\mathrm{tot} \left[ \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right) -
    \sqrt{\frac{2}{\pi}} x \exp\left(-\frac{x^2}{2}\right) \right] $$

    where $x = r / r_s$ is the dimensionless radius and $m_\mathrm{tot}$ is
    the total mass. The bracket is the regularized incomplete gamma function
    $P(3/2, z)$ with $z = x^2/2$; its two terms cancel to $\sim x^3$ as
    $x \to 0$, so up to $x = 1$ it is summed as the series of positive terms

    $$ P(3/2, z) = \frac{x^3 e^{-z}}{\sqrt{2\pi}}
       \sum_{k \geq 0} \frac{z^k}{(3/2)(5/2) \cdots (k + 3/2)}, $$

    whose 15 terms reach $10^{-17}$ there. Beyond, the direct difference
    cancels by at most a factor 3.5.

    """
    x = r / p["r_s"]
    z = x**2 / 2
    xe = _x_exp(x)
    small = x <= 1
    c = jnp.asarray(_GAMMA_SERIES, dtype=z.dtype)
    series = jnp.polyval(c, jnp.where(small, z, 0.5), unroll=len(_GAMMA_SERIES))
    core = xe * x * x * series / math.sqrt(2 * math.pi)
    direct = jsp.erf(x / math.sqrt(2)) - math.sqrt(2 / math.pi) * xe
    return p["m_tot"] * jnp.where(small, core, direct)  # type: ignore[no-any-return]


_GAMMA_SERIES = tuple(
    1 / math.prod(j + 1.5 for j in range(k + 1)) for k in range(14, -1, -1)
)
r"""$1 / ((3/2)(5/2) \cdots (k + 3/2))$ for $k \leq 14$, highest degree first."""


@ft.partial(jax.jit)
def dmass_dr(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""$dM/dr = 4\pi r^2 \rho$, for `radial_derivatives`."""
    x = r / p["r_s"]
    dbracket = math.sqrt(2 / math.pi) * _x_exp(x) * x
    return p["m_tot"] / p["r_s"] * dbracket


@ft.partial(jax.jit)
@radial_derivatives(mass_enclosed, dmass_dr)
def potential(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Potential for the Gaussian model.

    $$ \Phi(r) = -\frac{G m_\mathrm{tot}}{r} \mathrm{erf}\left(\frac{r}{\sqrt{2}
    r_s}\right) $$

    where $m_\mathrm{tot}$ is the total mass and $r_s$ is the scale radius.
    At $r=0$ this is the $0/0$ limit $\Phi(0) = -G m_\mathrm{tot}
    \sqrt{2/\pi} / r_s$, which we substitute explicitly since the profile is
    finite (unlike e.g. the Kepler potential) at the center.

    """
    r_s = p["r_s"]
    # erf(x)/x -> sqrt(2/pi)/r_s as r -> 0; guard the division so autodiff
    # doesn't see a 0/0 either.
    safe_r = jnp.where(r == 0, 1.0, r)
    x = safe_r / r_s
    phi0 = -p["G"] * p["m_tot"] / safe_r
    phi = phi0 * jsp.erf(x / jnp.sqrt(2.0))
    phi_origin = -p["G"] * p["m_tot"] * jnp.sqrt(2.0 / jnp.pi) / r_s
    return jnp.where(r == 0, phi_origin, phi)  # type: ignore[no-any-return]
