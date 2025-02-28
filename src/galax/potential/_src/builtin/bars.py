"""Bar-typed potentials."""

__all__ = [
    "LongMuraliBarPotential",
]

from functools import partial
from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class LongMuraliBarPotential(AbstractSinglePotential):
    """Long & Murali Bar Potential.

    A simple, triaxial model for a galaxy bar. This is a softened “needle”
    density distribution with an analytic potential form. See Long & Murali
    (1992) for details.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """The total mass."""

    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    alpha: AbstractParameter = ParameterField(dimensions="angle")  # type: ignore[assignment]
    """Position angle of the bar in the x-y plane."""

    @partial(jax.jit, inline=True)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        ul = self.units["length"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        a = self.a(t, ustrip=ul)
        b = self.b(t, ustrip=ul)
        c = self.c(t, ustrip=ul)
        alpha = self.alpha(t, ustrip=self.units["angle"])

        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        xp = x * jnp.cos(alpha) + y * jnp.sin(alpha)
        yp = -x * jnp.sin(alpha) + y * jnp.cos(alpha)

        Tm = jnp.sqrt((a - xp) ** 2 + yp**2 + (b + jnp.hypot(c, z)) ** 2)
        Tp = jnp.sqrt((a + xp) ** 2 + yp**2 + (b + jnp.hypot(c, z)) ** 2)

        GM_R = self.constants["G"].value * m_tot / (2.0 * a)

        return GM_R * jnp.log((xp - a + Tm) / (xp + a + Tp))
