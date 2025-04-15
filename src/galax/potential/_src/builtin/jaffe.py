"""Jaffe potential."""

__all__ = [
    # class
    "JaffePotential",
    # functions
    "potential",
]

import functools as ft
from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


@final
class JaffePotential(AbstractSinglePotential):
    """Jaffe Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass", doc="Characteristic mass.")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return potential(params, r)


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential function for the Jaffe potential.

    $$
    \phi(r) = -\frac{G M}{r_s} \log\left(1 + \frac{r}{r_s}\right)
    $$

    """
    return -p["G"] * p["m"] / p["r_s"] * jnp.log(1 + p["r_s"] / r)
