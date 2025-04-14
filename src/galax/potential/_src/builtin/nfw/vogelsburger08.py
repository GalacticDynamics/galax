"""galax: Galactic Dynamix in Jax."""

__all__ = ["Vogelsberger08TriaxialNFWPotential"]

from dataclasses import KW_ONLY
from functools import partial
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


@final
class Vogelsberger08TriaxialNFWPotential(AbstractSinglePotential):
    """Triaxial NFW Potential from DOI 10.1111/j.1365-2966.2007.12746.x."""

    m: AbstractParameter = ParameterField(dimensions="mass", doc="Scale mass.")  # type: ignore[assignment]
    # TODO: note the different definitions of m.

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.Quantity(1.0, ""),
        doc="""y/x axis ratio.

    The z/x axis ratio is defined as :math:`q_2^2 = 3 - q_1^2`
    """,
    )

    a_r: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.Quantity(1.0, ""),
        doc="""Transition radius relative to :math:`r_s`.

    :math:`r_a = a_r r_s  is a transition scale where the potential shape
    changes from ellipsoidal to near spherical.
    """,
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _r_e(self, xyz: gt.BtSz3, t: gt.BBtSz0) -> gt.BtFloatSz0:
        q1sq = self.q1(t, ustrip=self.units["dimensionless"]) ** 2
        q2sq = 3 - q1sq
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        return jnp.sqrt(x**2 + y**2 / q1sq + z**2 / q2sq)

    @partial(jax.jit, inline=True)
    def _r_tilde(self, xyz: gt.BtSz3, t: gt.BBtSz0) -> gt.BtFloatSz0:
        a_r = self.a_r(t, ustrip=self.units["dimensionless"])
        r_a = a_r * self.r_s(t, ustrip=self.units["length"])

        r_e = self._r_e(xyz, t)
        r = jnp.linalg.vector_norm(xyz, axis=-1)
        return (r_a + r) * r_e / (r_a + r_e)

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m = self.m(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        r = self._r_tilde(xyz, t)
        return -self.constants["G"].value * m * jnp.log(1.0 + r / r_s) / r
