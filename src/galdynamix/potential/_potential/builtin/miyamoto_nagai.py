from __future__ import annotations

__all__ = ["MiyamotoNagaiDisk"]


import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import PotentialBase
from galdynamix.utils import jit_method


class MiyamotoNagaiDisk(PotentialBase):
    m: jt.Array
    a: jt.Array
    b: jt.Array

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        R2 = q[0] ** 2 + q[1] ** 2
        return (
            -self._G
            * self.m
            / xp.sqrt(R2 + xp.square(xp.sqrt(q[2] ** 2 + self.b**2) + self.a))
        )
