from __future__ import annotations

__all__ = ["Integrator"]

import abc
from typing import Any, Protocol

import equinox as eqx
import jax.typing as jt


class FCallable(Protocol):
    def __call__(self, t: jt.Array, xv: jt.Array, args: Any) -> jt.Array:
        ...


class Integrator(eqx.Module):  # type: ignore[misc]
    F: FCallable

    @abc.abstractmethod
    def run(
        self, w0: jt.Array, t0: jt.Array, t1: jt.Array, ts: jt.Array | None
    ) -> jt.Array:
        ...
