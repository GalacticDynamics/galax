"""Miyamoto-Nagai Potential."""

__all__ = (
    # class
    "MiyamotoNagaiPotential",
    # functions
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
class MiyamotoNagaiPotential(AbstractSinglePotential):
    """Miyamoto-Nagai Potential."""

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    # TODO: rename
    a: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale length in the major-axis (x-y) plane."
    )

    b: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale length in the minor-axis (x-y) plane."
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

        # Compute parameters
        ul = self.units["length"]
        p = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=ul),
            "b": self.b(t, ustrip=ul),
        }
        return potential(p, xyz)


# ===================================================================
# Functions


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3) -> gt.Sz0:
    """Miyamoto-Nagai potential function."""
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    zp2 = (jnp.sqrt(xyz[..., 2] ** 2 + p["b"] ** 2) + p["a"]) ** 2
    return -p["G"] * p["m_tot"] / jnp.sqrt(R2 + zp2)
