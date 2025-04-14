"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "KeplerPotential",
    # functions
    "density",
    "potential",
    "point_mass_potential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.lax as qlax
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
class KeplerPotential(AbstractSinglePotential):
    r"""The Kepler potential for a point mass.

    .. math::

        \Phi = -\frac{G M(t)}{r}
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
        }
        return potential(params, r)

    @partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {"m_tot": self.m_tot(t, ustrip=self.units["mass"])}
        return density(params, r)


# ============================================


@partial(jax.jit, inline=True)
def point_mass_potential(G: gt.Sz0, m: gt.Sz0, r: gt.Sz0, /) -> gt.Sz0:
    r"""Potential energy for a point mass.

    $$ \Phi(r) = -\frac{G m}{r} $$

    Where $G$ is the gravitational constant and $m$ is the mass of the point
    mass.

    """
    return -G * m / r


# TODO: with units
@partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.Sz0:
    r"""Specific potential energy.

    $$ \Phi(r) = -\frac{G m_{tot}}{r} $$

    Where $m_{tot}$ is the total mass of the potential and $r$ is the distance
    from the center of the potential.

    """
    return point_mass_potential(p["G"], p["m_tot"], r)


@partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the Kepler potential.

    $$ \rho(r) = \delta(r) $$

    Where  $\delta(r)$ is the Dirac delta function. The density is only non-zero
    at the origin, where the mass is located. The density is zero everywhere
    else.

    """
    pred = jnp.logical_or(  # are we at the origin with non-zero mass?
        jnp.greater(r, jnp.zeros_like(r)),
        jnp.equal(p["m_tot"], jnp.zeros_like(p["m_tot"])),
    )
    return qlax.select(pred, jnp.zeros_like(r), jnp.full_like(r, fill_value=jnp.inf))
