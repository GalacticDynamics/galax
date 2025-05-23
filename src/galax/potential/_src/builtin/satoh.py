"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "SatohPotential",
    # functions
    "potential",
]

import functools as ft
from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class SatohPotential(AbstractSinglePotential):
    r"""SatohPotential(m, a, b, units=None, origin=None, R=None).

    Satoh potential for a flattened mass distribution.
    This is a good distribution for both disks and spheroids.

    .. math::

        \Phi = -\frac{G M}{\sqrt{R^2 + z^2 + a(a + 2\sqrt{z^2 + b^2})}}

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]

    a: AbstractParameter = ParameterField(dimensions="length", doc="Scale length")  # type: ignore[assignment]

    b: AbstractParameter = ParameterField(dimensions="length", doc="Scale height.")  # type: ignore[assignment]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=self.units["length"]),
            "b": self.b(t, ustrip=self.units["length"]),
        }
        return potential(params, xyz)


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Specific potential energy."""
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = xyz[..., 2]
    term = R2 + z**2 + p["a"] * (p["a"] + 2 * jnp.sqrt(z**2 + p["b"] ** 2))
    return -p["G"] * p["m_tot"] / jnp.sqrt(term)
