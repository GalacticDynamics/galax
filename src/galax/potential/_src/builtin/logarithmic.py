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
import unxt as u
from unxt.unitsystems import AbstractUnitSystem
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.utils._unxt import AllowValue


@final
class LogarithmicPotential(AbstractSinglePotential):
    """Logarithmic Potential."""

    v_c: AbstractParameter = ParameterField(dimensions="speed")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtQuSz0 | gt.BBtSz0, /
    ) -> gt.BBtSz0:
        # Compute parameters
        r_s = self.r_s(t, ustrip=self.units["length"])
        v_c = self.v_c(t, ustrip=self.units["speed"])

        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        r = jnp.linalg.vector_norm(xyz, axis=-1)
        return 0.5 * v_c**2 * jnp.log(r_s**2 + r**2)


@final
class LMJ09LogarithmicPotential(AbstractSinglePotential):
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
    units: AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtQuSz0 | gt.BBtSz0, /
    ) -> gt.BBtSz0:
        # Load parameters
        u1 = self.units["dimensionless"]
        r_s = self.r_s(t, ustrip=self.units["length"])
        q1, q2, q3 = self.q1(t, ustrip=u1), self.q2(t, ustrip=u1), self.q3(t, ustrip=u1)
        phi = self.phi(t, ustrip=self.units["angle"])
        v_c = self.v_c(t, ustrip=self.units["speed"])

        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Rotated and scaled coordinates
        sphi, cphi = jnp.sin(phi), jnp.cos(phi)
        x = xyz[..., 0] * cphi + xyz[..., 1] * sphi
        y = -xyz[..., 0] * sphi + xyz[..., 1] * cphi
        r2 = (x / q1) ** 2 + (y / q2) ** 2 + (xyz[..., 2] / q3) ** 2

        # Potential energy
        return 0.5 * v_c**2 * jnp.log(r_s**2 + r2)
