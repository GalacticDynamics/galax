"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "LogarithmicPotential",
    "LMJ09LogarithmicPotential",
]

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
from galax.potential._src.utils import r_spherical


@final
class LogarithmicPotential(AbstractSinglePotential):
    """Logarithmic Potential."""

    v_c: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="speed", doc="Circular velocity."
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])
        # Compute parameters
        r_s = self.r_s(t, ustrip=self.units["length"])
        v_c = self.v_c(t, ustrip=self.units["speed"])

        return 0.5 * v_c**2 * jnp.log(r_s**2 + r**2)


@final
class LMJ09LogarithmicPotential(AbstractSinglePotential):
    """Logarithmic Potential from LMJ09.

    https://ui.adsabs.harvard.edu/abs/2009ApJ...703L..67L/abstract
    """

    v_c: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="speed", doc="Circular velocity."
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", doc="X flattening."
    )
    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", doc="Y flattening."
    )
    q3: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", doc="Z flattening"
    )

    phi: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="angle", doc="Rotation in X-Y plane."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        u1 = self.units["dimensionless"]
        r_s = self.r_s(t, ustrip=self.units["length"])
        q1, q2, q3 = self.q1(t, ustrip=u1), self.q2(t, ustrip=u1), self.q3(t, ustrip=u1)
        phi = self.phi(t, ustrip=self.units["angle"])
        v_c = self.v_c(t, ustrip=self.units["speed"])

        # Rotated and scaled coordinates
        sphi, cphi = jnp.sin(phi), jnp.cos(phi)
        x = xyz[..., 0] * cphi + xyz[..., 1] * sphi
        y = -xyz[..., 0] * sphi + xyz[..., 1] * cphi
        r2 = (x / q1) ** 2 + (y / q2) ** 2 + (xyz[..., 2] / q3) ** 2

        # Potential energy
        return 0.5 * v_c**2 * jnp.log(r_s**2 + r2)
