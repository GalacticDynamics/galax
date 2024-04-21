"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "LogarithmicPotential",
    "LMJ09LogarithmicPotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
from unxt import AbstractUnitSystem, Quantity, unitsystem, ustrip
from xmmutablemap import ImmutableMap

import galax.typing as gt
from galax.potential._src.base import default_constants
from galax.potential._src.core import AbstractPotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class LogarithmicPotential(AbstractPotential):
    """Logarithmic Potential."""

    v_c: AbstractParameter = ParameterField(dimensions="speed")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        r_s = ustrip(self.units["length"], self.r_s(t))
        r = ustrip(self.units["length"], jnp.linalg.vector_norm(q, axis=-1))
        return 0.5 * self.v_c(t) ** 2 * jnp.log(r_s**2 + r**2)


@final
class LMJ09LogarithmicPotential(AbstractPotential):
    """Logarithmic Potential from LMJ09.

    https://ui.adsabs.harvard.edu/abs/2009ApJ...703L..67L/abstract
    """

    v_c: AbstractParameter = ParameterField(dimensions="speed")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    q1: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]
    q2: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]
    q3: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]

    phi: AbstractParameter = ParameterField(dimensions="angle")  # type: ignore[assignment]

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=unitsystem, static=True)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        # Load parameters
        q1, q2, q3 = self.q1(t), self.q2(t), self.q3(t)
        phi = self.phi(t)

        # Rotated and scaled coordinates
        sphi, cphi = jnp.sin(phi), jnp.cos(phi)
        x = q[..., 0] * cphi + q[..., 1] * sphi
        y = -q[..., 0] * sphi + q[..., 1] * cphi
        r2 = (x / q1) ** 2 + (y / q2) ** 2 + (q[..., 2] / q3) ** 2

        # Potential energy
        return (
            0.5
            * self.v_c(t) ** 2
            * jnp.log(
                ustrip(self.units["length"], self.r_s(t)) ** 2
                + ustrip(self.units["area"], r2)
            )
        )