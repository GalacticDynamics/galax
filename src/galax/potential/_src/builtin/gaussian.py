"""galax: Galactic Dynamix in Jax."""

__all__ = ["GaussianDensityPotential"]

import functools as ft
from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax
import jax.scipy.special as jsp

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
class GaussianDensityPotential(AbstractSinglePotential):
    r"""Potential of a spherical Gaussian density profile.

    The gravitational potential corresponding to a spherical Gaussian density profile:

    .. math::

        \rho(r) = \frac{M}{(2 \pi)^{3/2} \, r_s^3}\exp\left(-\frac{r^2}{2 r_s^2}\right)

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius (standard deviation of the Gaussian)."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return potential(params, r)

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return density(params, r)


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Gaussian density profile."""
    rho0 = p["m_tot"] / ((2 * jnp.pi) ** (3 / 2) * p["r_s"] ** 3)
    return rho0 * jnp.exp(-(r**2) / (2 * p["r_s"] ** 2))


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.Sz0:
    r"""Potential corresponding to a spherical Gaussian density profile in 3D."""
    return -p["G"] * p["m_tot"] / r * jsp.erf(r / (jnp.sqrt(2) * p["r_s"]))
