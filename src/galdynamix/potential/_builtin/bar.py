from __future__ import annotations

__all__ = ["BarPotential"]

import jax.numpy as xp
import jax.typing as jt
from gala.units import UnitSystem

from galdynamix.potential._base import PotentialBase
from galdynamix.utils import jit_method


class BarPotential(PotentialBase):
    """
    Rotating bar potentil, with hard-coded rotation.
    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """

    def __init__(
        self,
        m: jt.Array,
        a: jt.Array,
        b: jt.Array,
        c: jt.Array,
        Omega: jt.Array,
        *,
        units: UnitSystem | None = None,
    ) -> None:
        self.m: jt.Array
        self.a: jt.Array
        self.b: jt.Array
        self.c: jt.Array
        self.Omega: jt.Array
        super().__init__(units, {"m": m, "a": a, "b": b, "c": c, "Omega": Omega})

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        ## First take the simulation frame coordinates and rotate them by Omega*t
        ang = -self.Omega * t
        Rot_mat = xp.array(
            [
                [xp.cos(ang), -xp.sin(ang), 0],
                [xp.sin(ang), xp.cos(ang), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Rot_inv = xp.linalg.inv(Rot_mat)
        q_corot = xp.matmul(Rot_mat, q)

        T_plus = xp.sqrt(
            (self.a + q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (self.b + xp.sqrt(self.c**2 + q_corot[2] ** 2)) ** 2
        )
        T_minus = xp.sqrt(
            (self.a - q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (self.b + xp.sqrt(self.c**2 + q_corot[2] ** 2)) ** 2
        )

        # potential in a corotating frame
        return (self._G * self.m / (2.0 * self.a)) * xp.log(
            (q_corot[0] - self.a + T_minus) / (q_corot[0] + self.a + T_plus)
        )
