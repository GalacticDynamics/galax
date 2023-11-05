from __future__ import annotations

__all__ = ["NFWPotential", "NFWPotential_holder"]


import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import PotentialBase
from galdynamix.utils import jit_method


class NFWPotential_holder(PotentialBase):
    """
    Flattening in potential, not density
    Form from http://gala.adrian.pw/en/v0.1.2/api/gala.potential.FlattenedNFWPotential.html
    """

    v_c: jt.Array
    r_s: jt.Array
    q: jt.Array

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        m = xp.sqrt(q[0] ** 2 + q[1] ** 2 + (q[2] / self.q) ** 2)
        return (
            -((self.v_c**2) / xp.sqrt(xp.log(2.0) - 0.5))
            * xp.log(1.0 + m / self.r_s)
            / (m / self.r_s)
        )


class NFWPotential(PotentialBase):
    """
    standard def see spherical model @ https://github.com/adrn/gala/blob/main/gala/potential/potential/builtin/builtin_potentials.c
    """

    m: jt.Array
    r_s: jt.Array

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        v_h2 = -self._G * self.m / self.r_s
        m = (
            xp.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + 0.001) / self.r_s
        )  ##added softening!
        return (
            v_h2 * xp.log(1.0 + m) / m
        )  # -((self.v_c**2)/xp.sqrt(xp.log(2.0)-0.5) )*xp.log(1.0 + m/self.r_s)/(m/self.r_s)
