"""Bar-typed potentials."""

__all__ = [
    "BarPotential",
    "LongMuraliBarPotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import Any, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.utils._jax import vectorize_method
from galax.utils._unxt import AllowValue

# -------------------------------------------------------------------


def _make_rotation_matrix(angle: Any, /) -> Any:
    return jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


@final
class BarPotential(AbstractSinglePotential):
    """Rotating bar potentil, with hard-coded rotation.

    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Mass of the bar."""

    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    Omega: AbstractParameter = ParameterField(dimensions="frequency")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @vectorize_method(signature="(3),()->()")
    @partial(jax.jit)
    def _potential(self, xyz: gt.QuSz3, t: gt.RealQuSz0, /) -> gt.BtSz0:
        # Parse input and params
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)

        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        omega = self.Omega(t, ustrip=self.units["frequency"])
        a = self.a(t, ustrip=self.units["length"])
        b = self.b(t, ustrip=self.units["length"])
        c = self.c(t, ustrip=self.units["length"])

        # First take the simulation frame coordinates and rotate them by Omega*t
        R = _make_rotation_matrix(-omega * t)
        xr, yr, zr = jnp.matmul(R, xyz)

        T_plus = jnp.sqrt((a + xr) ** 2 + yr**2 + (b + jnp.sqrt(c**2 + zr**2)) ** 2)
        T_minus = jnp.sqrt((a - xr) ** 2 + yr**2 + (b + jnp.sqrt(c**2 + zr**2)) ** 2)

        # potential in a corotating frame
        GM_R = self.constants["G"].value * m_tot / (2.0 * a)
        return GM_R * jnp.log((xr - a + T_minus) / (xr + a + T_plus))


# -------------------------------------------------------------------


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
    def _potential(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtRealQuSz0 | gt.BBtRealSz0, /
    ) -> gt.BBtSz0:
        # Parse inputs and params
        ul = self.units["length"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        a = self.a(t, ustrip=ul)
        b = self.b(t, ustrip=ul)
        c = self.c(t, ustrip=ul)
        alpha = self.alpha(t, ustrip=self.units["angle"])

        xyz = u.ustrip(AllowValue, ul, xyz)
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        xp = x * jnp.cos(alpha) + y * jnp.sin(alpha)
        yp = -x * jnp.sin(alpha) + y * jnp.cos(alpha)

        temp = yp**2 + (b + jnp.sqrt(c**2 + z**2)) ** 2
        Tm = jnp.sqrt((a - xp) ** 2 + temp)
        Tp = jnp.sqrt((a + xp) ** 2 + temp)

        return (
            self.constants["G"].value
            * m_tot
            / (2 * a)
            * jnp.log((xp - a + Tm) / (xp + a + Tp))
        )
