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

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical


@final
class LogarithmicPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Logarithmic Potential.

    The spherical (:math:`q=1`) case of the flattened logarithmic potential
    of Binney & Tremaine 2008, *Galactic Dynamics*, 2nd ed., eq. 2.72 (a
    classic flat-rotation-curve, cored-halo model). The corresponding
    density (their eq. 2.71, with :math:`q=1`) is:

    $$
    \rho(r) = \frac{v_c^2 (3 r_s^2 + r^2)}{4\pi G (r_s^2+r^2)^2}
    $$

    """

    v_c: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="speed", doc="Circular velocity."
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        # Compute parameters
        p = self.params(t)

        return 0.5 * p["v_c"] ** 2 * jnp.log(p["r_s"] ** 2 + r**2)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        # Compute parameters
        p = self.params(t)

        numer = p["v_c"] ** 2 * (3 * p["r_s"] ** 2 + r**2)
        denom = 4 * jnp.pi * p["G"] * (p["r_s"] ** 2 + r**2) ** 2
        return numer / denom  # type: ignore[no-any-return]


@final
class LMJ09LogarithmicPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Logarithmic Potential from LMJ09.

    https://ui.adsabs.harvard.edu/abs/2009ApJ...703L..67L/abstract

    The corresponding (triaxial) density, derived from the potential above
    via Poisson's equation, is (using the bar-frame coordinates :math:`x,
    y, z` after undoing the :math:`\phi` rotation):

    $$
    \rho(x,y,z) = \frac{v_c^2}{4\pi G D^2} \left[
        D \left(q_1^2q_2^2 + q_1^2q_3^2 + q_2^2q_3^2\right)
        - 2\left(q_1^4q_2^4 z^2 + q_1^4q_3^4 y^2 + q_2^4q_3^4 x^2\right)
        \right]
    $$

    where :math:`D = q_1^2 q_2^2 q_3^2 r_s^2 + q_1^2 q_2^2 z^2 + q_1^2 q_3^2
    y^2 + q_2^2 q_3^2 x^2`.

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

        # Compute parameters
        p = self.params(t)

        # Rotated and scaled coordinates
        sphi, cphi = jnp.sin(p["phi"]), jnp.cos(p["phi"])
        x = xyz[..., 0] * cphi + xyz[..., 1] * sphi
        y = -xyz[..., 0] * sphi + xyz[..., 1] * cphi
        r2 = (x / p["q1"]) ** 2 + (y / p["q2"]) ** 2 + (xyz[..., 2] / p["q3"]) ** 2

        # Potential energy
        return 0.5 * p["v_c"] ** 2 * jnp.log(p["r_s"] ** 2 + r2)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute parameters
        p = self.params(t)

        # Rotated coordinates (bar frame)
        sphi, cphi = jnp.sin(p["phi"]), jnp.cos(p["phi"])
        x = xyz[..., 0] * cphi + xyz[..., 1] * sphi
        y = -xyz[..., 0] * sphi + xyz[..., 1] * cphi
        z = xyz[..., 2]

        q1sq, q2sq, q3sq = p["q1"] ** 2, p["q2"] ** 2, p["q3"] ** 2
        D = q1sq * q2sq * q3sq * p["r_s"] ** 2 + q1sq * q2sq * z**2 + q1sq * q3sq * y**2
        D = D + q2sq * q3sq * x**2
        numer = D * (q1sq * q2sq + q1sq * q3sq + q2sq * q3sq) - 2 * (
            q1sq**2 * q2sq**2 * z**2
            + q1sq**2 * q3sq**2 * y**2
            + q2sq**2 * q3sq**2 * x**2
        )
        return p["v_c"] ** 2 * numer / (4 * jnp.pi * p["G"] * D**2)  # type: ignore[no-any-return]
