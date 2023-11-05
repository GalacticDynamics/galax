from __future__ import annotations

__all__ = ["PotentialBase"]

import abc
from dataclasses import KW_ONLY, fields
from typing import Any

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt
from astropy.constants import G as apy_G
from gala.units import UnitSystem, dimensionless

from galdynamix.utils import jit_method


class PotentialBase(eqx.Module):  # type: ignore[misc]
    """Potential Class."""

    _: KW_ONLY
    units: UnitSystem = eqx.field(default=None, static=True)
    _G: float = eqx.field(init=False, static=True)

    def __post_init__(self) -> None:
        units = dimensionless if self.units is None else self.units
        object.__setattr__(self, "units", UnitSystem(units))

        G = 1 if self.units == dimensionless else apy_G.decompose(self.units).value
        object.__setattr__(self, "_G", G)

        for f in fields(self):
            param = getattr(self, f.name)
            if hasattr(param, "unit"):
                param = xp.asarray(param.decompose(self.units).value)
                object.__setattr__(self, f.name, param)

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the potential energy at the given position(s)."""
        raise NotImplementedError

    ###########################################################################
    # Core methods that use the above implemented functions

    @jit_method()
    def gradient(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the gradient."""
        return jax.grad(self.energy)(q, t)

    @jit_method()
    def density(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        lap = xp.trace(jax.hessian(self.energy)(q, t))
        return lap / (4 * xp.pi * self._G)

    @jit_method()
    def hessian(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        return jax.hessian(self.energy)(q, t)

    @jit_method()
    def acceleration(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        return -self.gradient(q, t)

    ###########################################################################

    @jit_method()
    def _vel_acc(self, t: jt.Array, xv: jt.Array, args: Any) -> jt.Array:
        return xp.hstack([xv[3:], -self.gradient(xv[:3], t)])

    @jit_method()
    def integrate_orbit(
        self, w0: jt.Array, t0: jt.Array, t1: jt.Array, ts: jt.Array | None
    ) -> jt.Array:
        from galdynamix.integrate._builtin.diffrax import (
            DiffraxIntegrator as Integrator,
        )

        return Integrator(self._vel_acc).run(w0, t0, t1, ts)
