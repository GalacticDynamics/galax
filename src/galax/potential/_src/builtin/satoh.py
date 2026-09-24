"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "SatohPotential",
    # functions
    "potential",
    "density",
]

import functools as ft

from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class SatohPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""SatohPotential(m, a, b, units=None, origin=None, R=None).

    Satoh, C. 1980, PASJ, 32, 41.
    https://ui.adsabs.harvard.edu/abs/1980PASJ...32...41S

    Satoh potential for a flattened mass distribution.
    This is a good distribution for both disks and spheroids.

    .. math::

        \Phi = -\frac{G M}{\sqrt{R^2 + z^2 + a(a + 2\sqrt{z^2 + b^2})}}

    with corresponding density (derived from the potential above via
    Poisson's equation):

    $$
    \rho(R,z) = \frac{M a b^2 \left[\zeta(R^2+a^2+6b^2+7z^2) + 5a\zeta^2\right]}
        {4\pi \zeta^4 \left(R^2+z^2+a(a+2\zeta)\right)^{5/2}},
        \qquad \zeta = \sqrt{z^2+b^2}
    $$

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]

    a: AbstractParameter = ParameterField(dimensions="length", doc="Scale length")  # type: ignore[assignment]

    b: AbstractParameter = ParameterField(dimensions="length", doc="Scale height.")  # type: ignore[assignment]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=self.units["length"]),
            "b": self.b(t, ustrip=self.units["length"]),
        }
        return potential(params, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        params = {
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=self.units["length"]),
            "b": self.b(t, ustrip=self.units["length"]),
        }
        return density(params, xyz)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Specific potential energy."""
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = xyz[..., 2]
    term = R2 + z**2 + p["a"] * (p["a"] + 2 * jnp.sqrt(z**2 + p["b"] ** 2))
    return -p["G"] * p["m_tot"] / jnp.sqrt(term)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Density function for the Satoh potential (Satoh 1980)."""
    a, b, m_tot = p["a"], p["b"], p["m_tot"]
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = xyz[..., 2]
    zeta = jnp.sqrt(z**2 + b**2)
    term = R2 + z**2 + a * (a + 2 * zeta)
    numer = (
        m_tot * a * b**2 * (zeta * (R2 + a**2 + 6 * b**2 + 7 * z**2) + 5 * a * zeta**2)
    )
    denom = 4 * jnp.pi * zeta**4 * term**2.5
    return numer / denom  # type: ignore[no-any-return]
