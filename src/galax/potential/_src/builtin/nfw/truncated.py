"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "HardCutoffNFWPotential",
    # functions
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
import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from .base import (
    density as nfw_density,
    enclosed_mass as nfw_enclosed_mass,
    potential as nfw_potential,
)
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.builtin.kepler import point_mass_potential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


@final
class HardCutoffNFWPotential(AbstractSinglePotential):
    r"""Sharply Truncated NFW Potential.

    Unlike a standard NFW potential this potential is truncated at a specified
    radius, meaning it has a finite mass.

    Terms:

    - $r$ is the radial coordinate
    - $r_s$ is the scale radius
    - $r_t$ is the truncation radius
    - $M_{tot}$ is the total mass of the potential

    The mass enclosed within a radius is given by

    $$

    M(<r)= \begin{cases}
        M_{NFW}(<r), & \text{if } r \le r_t, \\
        M_{NFW}(<r_t), & \text{if } r > r_t,
    \end{cases}

    $$

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc=r"""Total mass.""",
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]
    r_t: AbstractParameter = ParameterField(
        dimensions="length", doc="Truncation radius. This should be larger than r_s."
    )  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    # TODO: make public
    @ft.partial(jax.jit)
    def _mass_enclosed(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Mass enclosed within a radius.

        $$
        M(<r)= \begin{cases}
            M_{\mathrm{tot}};\dfrac{\ln(1+x) - \dfrac{x}{1+x}}{\ln(1+x_{\max}) -
            \dfrac{x_{\max}}{1+x_{\max}}}, & \text{if } r \le r_{\max}, \\
            M_{\mathrm{tot}}, & \text{if } r > r_{\max},
        \end{cases}
        $$

        where $x = r / r_s$ is the dimensionless radius, $x_{\max} = r_{\max} /
        r_s$ and the other parameters are defined on this class as the
        parameters of this potential.

        Parameters
        ----------
        xyz
            Cartesian position vector.
        t
            Time.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.HardCutoffNFWPotential(m=1e11, r_s=15, r_t=20, units="galactic")

        Evaluating at the truncation radius:

        >>> q = u.Quantity([20, 0, 0], "kpc")
        >>> t = u.Quantity(0, "Gyr")
        >>> pot._mass_enclosed(q, t)
        Array(2.75869289e+10, dtype=float64)

        Evaluating at a radius larger than the truncation radius:
        >>> q = u.Quantity([25, 0, 0], "kpc")
        >>> pot._mass_enclosed(q, t)
        Array(2.75869289e+10, dtype=float64)

        Evaluating at a radius smaller than the truncation radius:
        >>> q = u.Quantity([10, 0, 0], "kpc")
        >>> pot._mass_enclosed(q, t)
        Array(1.10825624e+10, dtype=float64)

        For comparison, here's a standard NFW potential:

        >>> nfw = gp.NFWPotential(m=1e11, r_s=15, units="galactic")
        >>> nfw._mass_enclosed(q, t)
        Array(1.10825624e+10, dtype=float64)

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "r_t": self.r_t(t, ustrip=self.units["length"]),
        }
        return mass_enclosed(params, r)

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.Sz3, t: gt.Sz0, /
    ) -> gt.Sz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "r_t": self.r_t(t, ustrip=self.units["length"]),
        }
        return potential(params, r)

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Density.

        $$ \rho(r) = \frac{\rho_0}{u (1 + u)^2} $$

        where $\rho_0$ is the central density and $u = r / r_s$ is the
        dimensionless radius. We actually parametrize not with $\rho_0$ but by a
        characteristic mass $m$, defined as $m = 4\pi\rho_0 r_s^3$.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.HardCutoffNFWPotential(m=1e11, r_s=15, r_t=20, units="galactic")

        Evaluating at the truncation radius:
        >>> q = u.Quantity([20, 0, 0], "kpc")
        >>> t = u.Quantity(0, "Gyr")
        >>> pot.density(q, t)
        Quantity[...](Array(324806.00630999, dtype=float64), unit='solMass / kpc3')

        Evaluating at a radius larger than the truncation radius:
        >>> q = u.Quantity([25, 0, 0], "kpc")
        >>> pot.density(q, t)
        Quantity[...](Array(0., dtype=float64), unit='solMass / kpc3')

        Evaluating at a radius smaller than the truncation radius:
        >>> q = u.Quantity([10, 0, 0], "kpc")
        >>> pot.density(q, t)
        Quantity[...](Array(1273239.54473516, dtype=float64), unit='solMass / kpc3')

        For comparison, here's a standard NFW potential:

        >>> nfw = gp.NFWPotential(m=1e11, r_s=15, units="galactic")
        >>> nfw.density(q, t)
        Quantity[...](Array(1273239.54473516, dtype=float64), unit='solMass / kpc3')

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "r_t": self.r_t(t, ustrip=self.units["length"]),
        }
        return density(params, r)


# ===================================================================


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the truncated NFW model.

    $$
    \rho(<r)=
        \begin{cases}
            \rho_{NFW}(r) & r \le r_{t}, \\
            0,            & r > r_{t},
        \end{cases}
    $$

    where $\rho_{NFW}(r)$ is the NFW density profile and $r_{t}$ is the
    truncation radius beyond which the density is zero.

    """
    return jax.lax.select(r <= p["r_t"], nfw_density(p, r), jnp.zeros_like(r))


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Mass enclosed within a radius.

    $$
    M(<r)= \begin{cases}
        M_{NFW}(<r), & \text{if } r \le r_t, \\
        M_{NFW}(<r_t), & \text{if } r > r_t,
    \end{cases}
    $$

    where $M_{NFW}(<r)$ is the NFW mass profile and $r_t$ is the truncation
    radius beyond which the mass is constant.
    $M_{NFW}(<r)$ is given by:

    $$ M_{NFW}(<r) = m (\ln(1+x) - \frac{x}{1+x}) $$

    where $x = r / r_s$ is the dimensionless radius, $r_s$ is the scale radius
    and $m = 4\pi\rho_0 r_s^3$ is the scale mass of the NFW profile.

    For more details see the NFW ``mass_enclosed`` method.

    """
    r_t = p["r_t"]
    return nfw_enclosed_mass(p, jnp.where(r <= r_t, r, r_t))


# -------------------------------------------------------------------


@ft.partial(jax.jit)
def _inner_potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""NFW potential within the truncation.

    $$ \Phi(r) = \Phi_{NFW}(r) + \frac{Gm}{r_s + r_t} $$

    """
    nfw = nfw_potential(p, r)
    constant = p["G"] * p["m"] / (p["r_s"] + p["r_t"])
    return nfw + constant


@ft.partial(jax.jit)
def _outer_potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Map to Kepler potential.

    $$ \Phi(r) = -\frac{G M_{tot}}{r} $$

    where $M_{tot}$ is the total mass of the potential and $r$ is the distance
    from the center of the potential. The total mass is given by the NFW mass
    enclosed within the truncation radius $r_t$.
    """
    m_tot = nfw_enclosed_mass(p, p["r_t"])
    return point_mass_potential(p["G"], m_tot, r)


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Specific potential energy.

    $$ \Phi(r) = \begin{cases}
        \Phi_{NFW}(r) + \frac{Gm}{r_s + r_t}, & r \le r_t, \\ -\frac{G
        M_{NFW}(<r_t)}{r}, & r > r_t,
    \end{cases} $$

    where

    - $ \Phi_{NFW}(r) $ is the NFW potential
    - $ G, m, r_s $ are the parameters of the NFW potential
    - $ r_t $ is the truncation radius
    - $ M_{NFW}(<r_t) $ is the enclosed mass of the NFW potential, given by $
      M_{NFW}(<r_t) = m (\ln(1+x_t) - \frac{x_t}{1+x_t}) $ for $x_t = r_t /
      r_s$

    """
    return jnp.where(r <= p["r_t"], _inner_potential(p, r), _outer_potential(p, r))
