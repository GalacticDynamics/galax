"""galax: Galactic Dynamix in Jax."""

__all__ = ["SatohPotential"]

from functools import partial
from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.utils._unxt import AllowValue


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

    @partial(jax.jit)
    def _potential(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtRealQuSz0 | gt.BBtRealSz0, /
    ) -> gt.BBtSz0:
        ul = self.units["length"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        a = self.a(t, ustrip=ul)
        b = self.b(t, ustrip=ul)
        xyz = u.ustrip(AllowValue, ul, xyz)

        R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
        z = xyz[..., 2]
        term = R2 + z**2 + a * (a + 2 * jnp.sqrt(z**2 + b**2))
        return -self.constants["G"].value * m_tot / jnp.sqrt(term)
