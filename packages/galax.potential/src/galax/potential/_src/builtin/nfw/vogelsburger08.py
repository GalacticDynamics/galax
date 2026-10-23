"""galax: Galactic Dynamix in Jax."""

__all__ = ["Vogelsberger08TriaxialNFWPotential"]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax

import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .base import potential as nfw_potential
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import safe_sqrt, safe_vector_norm


@final
class Vogelsberger08TriaxialNFWPotential(AbstractSinglePotential):
    """Triaxial NFW Potential from DOI 10.1111/j.1365-2966.2007.12746.x."""

    m: AbstractParameter = ParameterField(dimensions="mass", doc="Scale mass.")  # type: ignore[assignment]
    # TODO: note the different definitions of m.

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.Q(1.0, ""),
        doc="""y/x axis ratio.

    The z/x axis ratio is defined as :math:`q_2^2 = 3 - q_1^2`
    """,
    )

    a_r: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.Q(1.0, ""),
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

    symmetry = Symmetry.PLANE_REFLECTION
    """`Symmetry.PLANE_REFLECTION`: the potential is even in $x$, $y$ and $z$."""

    @ft.partial(jax.jit, inline=True)
    def _r_e(self, xyz: gt.BtSz3, p: gt.Params) -> gt.BtFloatSz0:
        q1sq = p["q1"] ** 2
        q2sq = 3 - q1sq
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        _result = safe_sqrt(x**2 + y**2 / q1sq + z**2 / q2sq)
        return _result  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _r_tilde(self, xyz: gt.BtSz3, p: gt.Params) -> gt.BtFloatSz0:
        r_a = p["a_r"] * p["r_s"]

        r_e = self._r_e(xyz, p)
        r = safe_vector_norm(xyz)
        return (r_a + r) * r_e / (r_a + r_e)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute parameters
        p = self.params(t)

        r = self._r_tilde(xyz, p)
        # NFW's radial function of r~, whose derivatives do not cancel.
        return nfw_potential(p, r)  # type: ignore[no-any-return]
