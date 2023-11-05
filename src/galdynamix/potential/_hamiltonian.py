"""galdynamix: Galactic Dynamix in Jax"""
# ruff: noqa: F403

from __future__ import annotations

__all__ = ["Hamiltonian"]

import equinox as eqx
import jax.typing as jt

from galdynamix.integrate._base import Integrator
from galdynamix.potential._potential.base import PotentialBase


class Hamiltonian(eqx.Module):  # type: ignore[misc]
    potential: PotentialBase

    def integrate_orbit(
        self,
        w0: jt.Array,
        /,
        Integrator: type[Integrator],
        *,
        t0: jt.Array,
        t1: jt.Array,
        ts: jt.Array | None,
    ) -> jt.Array:
        return Integrator(self.potential._velocity_acceleration).run(w0, t0, t1, ts)
