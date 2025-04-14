"""Bar-typed potentials."""

__all__ = [
    # class
    "LongMuraliBarPotential",
    # function
    "potential",
]

import functools as ft
from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class LongMuraliBarPotential(AbstractSinglePotential):
    """Long & Murali Bar Potential.

    A simple, triaxial model for a galaxy bar. This is a softened “needle”
    density distribution with an analytic potential form. See Long & Murali
    (1992) for details.

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="The total mass.")  # type: ignore[assignment]

    a: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Half-length defining the semi-major axis"
    )
    b: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Thickness softening length"
    )
    c: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Vertical softening length"
    )

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="angle", doc="Position angle of the bar in the x-y plane."
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        ul = self.units["length"]
        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=ul),
            "b": self.b(t, ustrip=ul),
            "c": self.c(t, ustrip=ul),
            "alpha": self.alpha(t, ustrip=self.units["angle"]),
        }
        return potential(params, xyz)


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Specific potential energy."""
    alpha = p["alpha"]
    a, b, c = p["a"], p["b"], p["c"]

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    xp = x * jnp.cos(alpha) + y * jnp.sin(alpha)
    yp = -x * jnp.sin(alpha) + y * jnp.cos(alpha)

    yz2 = yp**2 + (b + jnp.sqrt(c**2 + z**2)) ** 2
    T_plus = jnp.sqrt((a + xp) ** 2 + yz2)
    T_minus = jnp.sqrt((a - xp) ** 2 + yz2)

    GM_R = p["G"] * p["m_tot"] / (2.0 * a)
    return GM_R * jnp.log((xp - a + T_minus) / (xp + a + T_plus))
