"""Disk-like potentials."""

__all__ = (
    # class
    "KuzminPotential",
    # functions
    "potential",
)

import functools as ft
from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class KuzminPotential(AbstractSinglePotential):
    r"""Kuzmin Potential.

    .. math::

        \Phi(x, t) = -\frac{G M(t)}{\sqrt{R^2 + (a(t) + |z|)^2}}

    See https://galaxiesbook.org/chapters/II-01.-Flattened-Mass-Distributions.html#Razor-thin-disk:-The-Kuzmin-model

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale length of the 'disk'."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit, inline=True)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute
        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return potential(params, xyz)


# ====================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz0:
    r"""Potential for the Kuzmin potential.

    $$ \Phi(r) = -\frac{G M}{\sqrt{r^2 + a^2}} $$

    where $M$ is the total mass and $a$ is the scale radius.
    """
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = xyz[..., 2]
    return -p["G"] * p["m_tot"] / jnp.sqrt(R2 + (jnp.abs(z) + p["r_s"]) ** 2)
