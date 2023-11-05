from __future__ import annotations

__all__ = ["CompositePotential"]


from dataclasses import InitVar

import jax.numpy as xp
import jax.typing as jt

from galdynamix.utils import jit_method

from .base import PotentialBase


class CompositePotential(PotentialBase):
    """Composite Potential."""

    potential_list: InitVar[list[PotentialBase]]

    def __post_init__(self, potential_list: list[PotentialBase]) -> None:
        super().__post_init__()
        object.__setattr__(self, "_potential_list", potential_list)

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
