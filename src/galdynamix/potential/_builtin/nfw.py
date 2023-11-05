from __future__ import annotations

__all__ = ["NFWPotential", "NFWPotential_holder"]


import jax.numpy as xp
import jax.typing as jt
from gala.units import UnitSystem

from galdynamix.potential._base import PotentialBase
from galdynamix.utils import jit_method


class NFWPotential_holder(PotentialBase):
    """
    Flattening in potential, not density
    Form from http://gala.adrian.pw/en/v0.1.2/api/gala.potential.FlattenedNFWPotential.html
    """

    def __init__(
        self, v_c: jt.Array, r_s: jt.Array, q: jt.Array, units: UnitSystem | None = None
    ) -> None:
        self.v_c: jt.Array
        self.r_s: jt.Array
        self.q: jt.Array
        super().__init__(units, {"v_c": v_c, "r_s": r_s, "q": q})

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

    def __init__(
        self, m: jt.Array, r_s: jt.Array, units: UnitSystem | None = None
    ) -> None:
        self.m: jt.Array
        self.r_s: jt.Array
        super().__init__(units, {"m": m, "r_s": r_s})

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        v_h2 = -self._G * self.m / self.r_s
        m = (
            xp.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + 0.001) / self.r_s
        )  ##added softening!
        return (
            v_h2 * xp.log(1.0 + m) / m
        )  # -((self.v_c**2)/xp.sqrt(xp.log(2.0)-0.5) )*xp.log(1.0 + m/self.r_s)/(m/self.r_s)
