from __future__ import annotations

__all__ = ["AbstractIntegrator"]

import abc
from typing import Any, Protocol

import equinox as eqx
import jax.typing as jt


class FCallable(Protocol):
    def __call__(self, t: jt.Array, xv: jt.Array, args: Any) -> jt.Array:
        ...


class AbstractIntegrator(eqx.Module):  # type: ignore[misc]
    """Integrator Class."""

    F: FCallable
    """The function to integrate."""
    # TODO: should this be moved to be the first argument of the run method?

    @abc.abstractmethod
    def run(
        self, w0: jt.Array, t0: jt.Array, t1: jt.Array, ts: jt.Array | None
    ) -> jt.Array:
        ...
