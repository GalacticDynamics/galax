from __future__ import annotations

__all__ = ["MiyamotoNagaiDisk"]


import jax.numpy as xp
import jax.typing as jt
from gala.units import UnitSystem

from galdynamix.potential._base import PotentialBase
from galdynamix.utils import jit_method


class MiyamotoNagaiDisk(PotentialBase):
    m: jt.Array
    a: jt.Array
    b: jt.Array

    def __init__(
        self, m: jt.Array, a: jt.Array, b: jt.Array, units: UnitSystem | None = None
    ) -> None:
        super().__init__(units, {"m": m, "a": a, "b": b})

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        R2 = q[0] ** 2 + q[1] ** 2
        return (
            -self._G
            * self.m
            / xp.sqrt(R2 + xp.square(xp.sqrt(q[2] ** 2 + self.b**2) + self.a))
        )
