"""galax: Galactic Dynamix in Jax."""

__all__ = ["StoneOstriker15Potential"]

import functools as ft

import jax

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


class StoneOstriker15Potential(AbstractSinglePotential):
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

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_h": self.r_h(t, ustrip=ul),
            "r_c": self.r_c(t, ustrip=ul),
        }
        return density(params, r)

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_h": self.r_h(t, ustrip=ul),
            "r_c": self.r_c(t, ustrip=ul),
        }
        return potential(params, r)


# ===================================================================


@ft.partial(jax.jit)
def rhoc_of_mtot(p: gt.Params, /) -> gt.FloatSz0:
    r"""Density from the mass.

    Solving from S&O15 Eq. 5:

    $$ \rho_c = \frac{ M_{tot} (r_h + r_c)}{2\pi^2 r_c^2 r_h^2 } $$

    """
    r_c, r_h = p["r_c"], p["r_h"]
    rho_c = p["m_tot"] * (r_h + r_c) / (2 * jnp.pi**2 * r_c**2 * r_h**2)
    return rho_c


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
    return rho_c / (core_term * halo_term)


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass function for Stone-Ostriker 2015 potential.

    S&O15 Eq. 4:

    $$ M(<r) = \frac{2 M_{tot}}{\pi (r_h-r_c)} (r_h \atan(r/r_h) - r_c \atan(r/r_c)) $$

    """
    r_c, r_h = p["r_c"], p["r_h"]
    A = 2 * p["m_tot"] / (jnp.pi * (r_h - r_c))
    atan_term = r_h * jnp.atan2(r, r_h) - r_c * jnp.atan2(r, r_c)
    return A * atan_term


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    """Potential function for Stone-Ostriker 2015 potential."""
    r_c, r_h = p["r_c"], p["r_h"]
    A = -2 * p["G"] * p["m_tot"] / (jnp.pi * (r_h - r_c))
    atan_term = (r_h * jnp.atan2(r, r_h) - r_c * jnp.atan2(r, r_c)) / r
    log_term = 0.5 * jnp.log((r**2 + r_h**2) / (r**2 + r_c**2))
    return A * (atan_term + log_term)
