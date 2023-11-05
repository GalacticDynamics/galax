from __future__ import annotations

__all__ = ["CompositePotential"]


from collections.abc import Mapping
from typing import TypeVar

import equinox as eqx
import jax.numpy as xp
import jax.typing as jt

from galdynamix.utils import jit_method

from .base import PotentialBase

V = TypeVar("V")


class FrozenDict(Mapping[str, V]):
    def __init__(self, *args: V, **kwargs: V) -> None:
        self._data: dict[str, V] = dict(*args, **kwargs)

    def __getitem__(self, key: str) -> V:
        return self._data[key]

    def __iter__(self) -> iter[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        return hash(tuple(self._data.items()))

    def keys(self) -> iter[str]:
        return self._data.keys()

    def values(self) -> iter[V]:
        return self._data.values()

    def items(self) -> iter[tuple[str, V]]:
        return self._data.items()


class CompositePotential(PotentialBase):
    """Composite Potential."""

    # potentials: FrozenDict[str, PotentialBase] = eqx.field(converter=FrozenDict)

    potentials: tuple[PotentialBase] = eqx.field(converter=tuple)

    # def __post_init__(self) -> None:
    #     super().__post_init__()
    #     self._potentials: dict[str, PotentialBase]
    #     object.__setattr__(self, "_potentials", list(self.potentials.values()))

    # === Mapping ===

    def __getitem__(self, key: str) -> PotentialBase:
        return self.potentials[key]

    def __iter__(self) -> iter[str]:
        return iter(self.potentials)

    def __len__(self) -> int:
        return len(self.potentials)

    # === Potential ===

    @jit_method()
    def energy(
        self,
        q: jt.Array,
        t: jt.Array,
    ) -> jt.Array:
        output = []
        for p in self.potentials:
            output.append(p.energy(q, t))
        return xp.sum(xp.array(output))
