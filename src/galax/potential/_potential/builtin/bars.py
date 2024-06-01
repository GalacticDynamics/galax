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
from quax import quaxify

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity, unitsystem

import galax.typing as gt
from galax.potential._potential.base import default_constants
from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.utils import ImmutableDict
from galax.utils._jax import vectorize_method

# -------------------------------------------------------------------


@final
class BarPotential(AbstractPotential):
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
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableDict[Quantity] = eqx.field(
        default=default_constants, converter=ImmutableDict
    )

    # TODO: inputs w/ units
    @quaxify  # type: ignore[misc]
    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential(self, q: gt.QVec3, t: gt.RealQScalar, /) -> gt.FloatQScalar:
        ## First take the simulation frame coordinates and rotate them by Omega*t
        ang = -self.Omega(t) * t
        rotation_matrix = xp.asarray(
            [
                [xp.cos(ang), -xp.sin(ang), 0],
                [xp.sin(ang), xp.cos(ang), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        q_corot = xp.matmul(rotation_matrix, q)

        a = self.a(t)
        b = self.b(t)
        c = self.c(t)
        T_plus = xp.sqrt(
            (a + q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (b + xp.sqrt(c**2 + q_corot[2] ** 2)) ** 2
        )
        T_minus = xp.sqrt(
            (a - q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (b + xp.sqrt(c**2 + q_corot[2] ** 2)) ** 2
        )

        # potential in a corotating frame
        return (self.constants["G"] * self.m_tot(t) / (2.0 * a)) * xp.log(
            (q_corot[0] - a + T_minus) / (q_corot[0] + a + T_plus),
        )


# -------------------------------------------------------------------


@final
class LongMuraliBarPotential(AbstractPotential):
    """Long & Murali Bar Potential.

    A simple, triaxial model for a galaxy bar. This is a softened “needle”
    density distribution with an analytic potential form. See Long & Murali
    (1992) for details.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    alpha: AbstractParameter = ParameterField(dimensions="angle")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.BatchFloatQScalar:
        m_tot = self.m_tot(t)
        a, b, c = self.a(t), self.b(t), self.c(t)
        alpha = self.alpha(t)

        x = q[..., 0] * xp.cos(alpha) + q[..., 1] * xp.sin(alpha)
        y = -q[..., 0] * xp.sin(alpha) + q[..., 1] * xp.cos(alpha)
        z = q[..., 2]

        _temp = y**2 + (b + xp.sqrt(c**2 + z**2)) ** 2
        Tm = xp.sqrt((a - x) ** 2 + _temp)
        Tp = xp.sqrt((a + x) ** 2 + _temp)

        return (
            self.constants["G"] * m_tot / (2 * a) * xp.log((x - a + Tm) / (x + a + Tp))
        )
