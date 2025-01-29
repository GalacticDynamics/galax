"""galax: Galactic Dynamix in Jax."""

__all__ = ["SatohPotential"]

from functools import partial
from typing import final

import jax

import quaxed.numpy as jnp

import galax.typing as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class SatohPotential(AbstractSinglePotential):
    r"""SatohPotential(m, a, b, units=None, origin=None, R=None).

    Satoh potential for a flattened mass distribution.
    This is a good distribution for both disks and spheroids.

    .. math::

        \Phi = -\frac{G M}{\sqrt{R^2 + z^2 + a(a + 2\sqrt{z^2 + b^2})}}

    """

    #: Characteristic mass
    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]

    #: Scale length
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    #: Scale height
    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @partial(jax.jit, inline=True)
    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        a, b = self.a(t), self.b(t)
        R2 = q[..., 0] ** 2 + q[..., 1] ** 2
        z = q[..., 2]
        term = R2 + z**2 + a * (a + 2 * jnp.sqrt(z**2 + b**2))
        return -self.constants["G"] * self.m_tot(t) / jnp.sqrt(term)
