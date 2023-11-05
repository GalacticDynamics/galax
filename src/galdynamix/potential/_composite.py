from __future__ import annotations

__all__ = ["CompositePotential"]


import jax.numpy as xp
import jax.typing as jt
from gala.units import UnitSystem

from galdynamix.potential._base import PotentialBase
from galdynamix.utils import jit_method


class CompositePotential(PotentialBase):
    """Composite Potential."""

    def __init__(
        self, potential_list: list[PotentialBase], units: UnitSystem | None = None
    ) -> None:
        self._potential_list: list[PotentialBase]
        super().__init__(units, {"_potential_list": potential_list})

    @jit_method()
    def energy(
        self,
        q: jt.Array,
        t: jt.Array,
    ) -> jt.Array:
        output = []
        for i in range(len(self._potential_list)):
            output.append(self._potential_list[i].energy(q, t))
        return xp.sum(xp.array(output))
