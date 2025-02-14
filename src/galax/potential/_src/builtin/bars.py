"""Bar-typed potentials."""

__all__ = [
    "BarPotential",
    "LongMuraliBarPotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.typing as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.utils._jax import vectorize_method

# -------------------------------------------------------------------


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
    def _potential(self, q: gt.QuSz3, t: gt.RealQuSz0, /) -> gt.SpecificEnergyBtSz0:
        ## First take the simulation frame coordinates and rotate them by Omega*t
        ang = -self.Omega(t) * t
        rotation_matrix = jnp.asarray(
            [
                [jnp.cos(ang), -jnp.sin(ang), 0],
                [jnp.sin(ang), jnp.cos(ang), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        q_corot = jnp.matmul(rotation_matrix, q)

        a = self.a(t)
        b = self.b(t)
        c = self.c(t)
        T_plus = jnp.sqrt(
            (a + q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (b + jnp.sqrt(c**2 + q_corot[2] ** 2)) ** 2
        )
        T_minus = jnp.sqrt(
            (a - q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (b + jnp.sqrt(c**2 + q_corot[2] ** 2)) ** 2
        )

        # potential in a corotating frame
        return (self.constants["G"] * self.m_tot(t) / (2.0 * a)) * jnp.log(
            (q_corot[0] - a + T_minus) / (q_corot[0] + a + T_plus),
        )


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
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        m_tot = self.m_tot(t)
        a, b, c = self.a(t), self.b(t), self.c(t)
        alpha = self.alpha(t)

        x = q[..., 0] * jnp.cos(alpha) + q[..., 1] * jnp.sin(alpha)
        y = -q[..., 0] * jnp.sin(alpha) + q[..., 1] * jnp.cos(alpha)
        z = q[..., 2]

        _temp = y**2 + (b + jnp.sqrt(c**2 + z**2)) ** 2
        Tm = jnp.sqrt((a - x) ** 2 + _temp)
        Tp = jnp.sqrt((a + x) ** 2 + _temp)

        return (
            self.constants["G"] * m_tot / (2 * a) * jnp.log((x - a + Tm) / (x + a + Tp))
        )
