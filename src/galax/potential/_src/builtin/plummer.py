"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "PlummerPotential",
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
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


@final
class PlummerPotential(AbstractSinglePotential):
    r"""Plummer Potential.

    The Plummer potential is a simple model for a spherical distribution of
    mass, often used in astrophysics to describe globular clusters or
    star clusters. It is characterized by a total mass and a scale length.

    The density profile is given by:

    $$
    \rho(r) = \frac{3 M_{tot}}{4 \pi r_s^3}
              \frac{1}{\left(1 + \frac{r^2}{r_s^2}\right)^{5/2}}
    $$

    where $M_{tot}$ is the total mass and $r_s$ is the scale length.

    The potential is given by:

    $$ \Phi(r) = -\frac{G M_{tot}}{\sqrt{r^2 + r_s^2}} $$

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=ul),
        }
        return density(params, r)

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=ul),
        }
        return potential(params, r)


# ===================================================================


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the Plummer potential.

    $$
    \rho(r) = \frac{3 M_{tot}}{4 \pi r_s^3}
              \frac{1}{\left(1 + \frac{r^2}{r_s^2}\right)^{5/2}}
    $$

    where $M_{tot}$ is the total mass and $r_s$ is the scale length.

    """
    rho0 = 3 * p["m_tot"] / (4 * jnp.pi * p["r_s"] ** 3)
    return rho0 / jnp.power(1 + (r / p["r_s"]) ** 2, 2.5)


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass for the Plummer potential.

    $$ M(<r) = \frac{M_{tot} r^3}{(r^2 + r_s^2)^{3/2}} $$

    """
    return p["m_tot"] * r**3 / jnp.power(r**2 + p["r_s"] ** 2, 1.5)


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential function for Plummer potential.

    $$ \Phi(r) = -\frac{G M_{tot}}{\sqrt{r^2 + r_s^2}} $$

    where $G$ is the gravitational constant, $M_{tot}$ is the total mass, and
    $r_s$ is the scale length.

    """
    return -p["G"] * p["m_tot"] / jnp.sqrt(r**2 + p["r_s"] ** 2)
