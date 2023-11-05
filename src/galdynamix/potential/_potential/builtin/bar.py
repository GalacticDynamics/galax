from __future__ import annotations

__all__ = ["BarPotential"]

import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import PotentialBase
from galdynamix.utils import jit_method


class BarPotential(PotentialBase):
    """
    Rotating bar potentil, with hard-coded rotation.
    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """

    m: jt.Array
    a: jt.Array
    b: jt.Array
    c: jt.Array
    Omega: jt.Array

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
