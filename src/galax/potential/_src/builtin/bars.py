"""Bar-typed potentials."""

__all__ = [
    "LongMuraliBarPotential",
    "MonariEtAl2016BarPotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.utils._jax import vectorize_method


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

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        ul = self.units["length"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        a, b, c = self.a(t, ustrip=ul), self.b(t, ustrip=ul), self.c(t, ustrip=ul)
        alpha = self.alpha(t, ustrip=self.units["angle"])

        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        xp = x * jnp.cos(alpha) + y * jnp.sin(alpha)
        yp = -x * jnp.sin(alpha) + y * jnp.cos(alpha)

        T_plus = jnp.sqrt((a + xp) ** 2 + yp**2 + (b + jnp.sqrt(c**2 + z**2)) ** 2)
        T_minus = jnp.sqrt((a - xp) ** 2 + yp**2 + (b + jnp.sqrt(c**2 + z**2)) ** 2)

        GM_R = self.constants["G"].value * m_tot / (2.0 * a)
        return GM_R * jnp.log((xp - a + T_minus) / (xp + a + T_plus))


@final
class MonariEtAl2016BarPotential(AbstractSinglePotential):
    """Monari et al. (2016) Bar Potential.

    This is an generalization to 3D of the Dehnen 2000 bar potential.
    We take the defaults from Monari et al. (2016) paper.

    https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.3835M/abstract

    """

    _: KW_ONLY

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(0.01, ""),
        dimensions="dimensionless",
        doc="""The amplitude.

    the ratio between the bar's and axisymmetric contribution to the radial
    force, along the bar's long axis at (R,z) = (R0,0).
    """,
    )

    R0: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="The Galactocentric radius of the Sun."
    )

    v0: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="speed", doc="The circular velocity at R0"
    )

    Rb: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(3.5, "kpc"),
        dimensions="length",
        doc="The length of the bar.",
    )

    phi_b: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(25, "deg"), dimensions="angle", doc="Bar angle."
    )

    Omega: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(52.2, "km/(s kpc)"),
        dimensions="frequency",
        doc="Bar pattern speed.",
    )

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential(self, xyz: gt.QuSz3 | gt.Sz3, t: gt.QuSz0 | gt.Sz0) -> gt.Sz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        tq = u.Quantity.from_(t, self.units["time"])
        t = u.ustrip(AllowValue, self.units["time"], t)

        # Compute parameters
        ul = self.units["length"]
        alpha = self.alpha(t, ustrip=self.units["dimensionless"])
        v0 = self.v0(tq, ustrip=self.units["speed"])
        R0 = self.R0(tq, ustrip=ul)
        Rb = self.Rb(tq, ustrip=ul)
        phi_b = self.phi_b(tq, ustrip=self.units["angle"])
        Omega = self.Omega(tq, ustrip=self.units["frequency"])

        def U_of_r(s: gt.Sz0) -> gt.Sz0:
            # M+2016 eq.3, modified to work on s=r/Rb
            def gtr_func(s: gt.Sz0) -> gt.Sz0:
                return -(s**-3)

            def less_func(s: gt.Sz0) -> gt.Sz0:
                return s**3 - 2.0

            pred = s >= 1
            return jax.lax.cond(pred, gtr_func, less_func, s)

        R2 = xyz[0] ** 2 + xyz[1] ** 2
        r2 = R2 + xyz[2] ** 2
        phi = jnp.arctan2(xyz[1], xyz[0])

        prefactor = alpha * (v0**2 / 3) * (R0 / Rb) ** 3
        u_of_r = U_of_r(jnp.sqrt(r2) / Rb)
        gamma_b = 2 * (phi - phi_b - Omega * t)  # M+2016 eq.2

        return prefactor * u_of_r * (R2 / r2) * jnp.cos(gamma_b)  # M+2016 eq.1
