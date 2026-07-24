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
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import quaxed.scipy.special as jsp
import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


@final
class GaussianPotential(AbstractSinglePotential):
    r"""Gaussian Potential.

    A spherical mass distribution with a Gaussian density profile.

    The density profile is given by:

    $$ \rho(r) = \frac{ m }{ (2 \pi)^{3/2} r_s^3 } \exp\left(-\frac{r^2}{2
    r_s^2}\right) $$

    where :math:`m` is the total mass and :math:`r_s` is the scale radius.
    Unlike the NFW profile, the Gaussian profile has a finite total mass, so
    :math:`m` is the total (rather than a characteristic) mass.

    Solving Poisson's equation gives the gravitational potential

    $$ \Phi(r) = -\frac{G m}{r} \mathrm{erf}\left(\frac{r}{\sqrt{2} r_s}\right)
    $$

    Which has the expected behavior of being finite at the center and going to
    the Kepler potential :math:`-G m / r` at large radii.

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
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

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        r"""Potential energy.

        .. math::

            \Phi(r) = -\frac{G m}{r} \mathrm{erf}\left(\frac{r}{\sqrt{2}
            r_s}\right)
        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Q.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return potential(params, r)

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Density.

        .. math::

            \rho_0 = \frac{m}{(2 \pi)^{3/2} r_s^3}
            \rho(r) = \rho_0 \exp\left(-\frac{u^2}{2}\right)

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Q.from_(t, self.units["time"])

        params = {
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return density(params, r)

    @ft.partial(jax.jit)
    def _mass_enclosed(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Enclosed mass.

        .. math::

            M(<r) = m \left[ \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right) -
            \sqrt{\frac{2}{\pi}} x \exp\left(-\frac{x^2}{2}\right) \right]

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.GaussianPotential(m=1e11, r_s=15, units="galactic")

        >>> q = u.Q([10, 0, 0], "kpc")
        >>> t = u.Q(0, "Gyr")
        >>> pot._mass_enclosed(q, t)
        Array(6.90842509e+09, dtype=float64)

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Q.from_(t, self.units["time"])

        params = {
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return mass_enclosed(params, r)


# =============================================================


# -----------------------------------------------


@ft.partial(jax.jit)
def rho0_of_m(p: gt.Params, /) -> gt.Sz0:
    r"""Central density for the Gaussian model.

    The Gaussian profile is parametrized by a total mass $m$ and a scale
    radius $r_s$. The central density is given by

    $$ \rho_0 = \frac{m}{(2 \pi)^{3/2} r_s^3}. $$

    Examples
    --------
    >>> import jax.numpy as jnp

    A quick sanity check:

    >>> rho0_of_m({"m": 1.0, "r_s": 1.0}) - 1 / (2 * jnp.pi) ** 1.5
    Array(0., dtype=float64, weak_type=True)

    """
    return p["m"] / ((2 * jnp.pi) ** 1.5 * p["r_s"] ** 3)


@ft.partial(jax.jit)
def m_of_rho0(p: gt.Params, /) -> gt.Sz0:
    r"""Total mass for the Gaussian model.

    The Gaussian profile is parametrized by a total mass $m$ and a scale
    radius $r_s$. The total mass is given by

    $$ m = (2 \pi)^{3/2} \rho_0 r_s^3. $$

    Examples
    --------
    >>> import jax.numpy as jnp

    A quick sanity check:
    >>> m_of_rho0({"rho0": 1.0, "r_s": 1.0}) - (2 * jnp.pi) ** 1.5
    Array(0., dtype=float64, weak_type=True)

    """
    return (2 * jnp.pi) ** 1.5 * p["rho0"] * p["r_s"] ** 3


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
    return rho0 * jnp.exp(-(x**2) / 2)


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Enclosed mass for the Gaussian model.

    $$ M(<r) = m \left[ \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right) -
    \sqrt{\frac{2}{\pi}} x \exp\left(-\frac{x^2}{2}\right) \right] $$

    where $x = r / r_s$ is the dimensionless radius and $m$ is the total
    mass.

    """
    x = r / p["r_s"]
    m = p["m"]
    erf_term = jsp.erf(x / jnp.sqrt(2.0))
    exp_term = jnp.sqrt(2.0 / jnp.pi) * x * jnp.exp(-(x**2) / 2)
    return m * (erf_term - exp_term)


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Potential for the Gaussian model.

    $$ \Phi(r) = -\frac{G m}{r} \mathrm{erf}\left(\frac{r}{\sqrt{2}
    r_s}\right) $$

    where $m$ is the total mass and $r_s$ is the scale radius.

    """
    r_s = p["r_s"]
    x = r / r_s
    phi0 = -p["G"] * p["m"] / r
    return phi0 * jsp.erf(x / jnp.sqrt(2.0))
