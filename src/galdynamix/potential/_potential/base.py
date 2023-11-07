from __future__ import annotations

__all__ = ["AbstractPotentialBase", "AbstractPotential"]

import abc
from dataclasses import KW_ONLY, fields
from typing import Any

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt
from astropy.constants import G as apy_G

from galdynamix.integrate._base import AbstractIntegrator
from galdynamix.integrate._builtin import DiffraxIntegrator
from galdynamix.potential._potential.param.field import ParameterField
from galdynamix.units import UnitSystem, dimensionless
from galdynamix.utils import partial_jit


class AbstractPotentialBase(eqx.Module):  # type: ignore[misc]
    """Potential Class."""

    units: eqx.AbstractVar[UnitSystem]

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    def potential_energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the potential energy at the given position(s)."""
        raise NotImplementedError

    ###########################################################################
    # Parsing

    def _init_units(self) -> None:
        G = 1 if self.units == dimensionless else apy_G.decompose(self.units).value
        object.__setattr__(self, "_G", G)

        # Handle unit conversion for all ParameterField
        for f in fields(self):
            param = getattr(self.__class__, f.name, None)
            if not isinstance(param, ParameterField):
                continue

            value = getattr(self, f.name)
            if isinstance(value, u.Quantity):
                value = value.to_value(
                    self.units[param.physical_type], equivalencies=param.equivalencies
                )
                object.__setattr__(self, f.name, value)

        # other parameters, check their metadata
        for f in fields(self):
            if "physical_type" not in f.metadata:
                continue

            value = getattr(self, f.name)
            if isinstance(value, u.Quantity):
                value = value.to_value(
                    self.units[f.metadata["physical_type"]],
                    equivalencies=f.metadata.get("equivalencies", None),
                )
                object.__setattr__(self, f.name, value)

    ###########################################################################
    # Core methods that use the above implemented functions

    @partial_jit()
    def __call__(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the potential energy at the given position(s)."""
        return self.potential_energy(q, t)

    @partial_jit()
    def gradient(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the gradient."""
        return jax.grad(self.potential_energy)(q, t)

    @partial_jit()
    def density(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        lap = xp.trace(jax.hessian(self.potential_energy)(q, t))
        return lap / (4 * xp.pi * self._G)

    @partial_jit()
    def hessian(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        return jax.hessian(self.potential_energy)(q, t)

    @partial_jit()
    def acceleration(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        return -self.gradient(q, t)

    ###########################################################################
    # Convenience methods

    @partial_jit()
    def _vel_acc(self, t: jt.Array, qp: jt.Array, args: tuple[Any, ...]) -> jt.Array:
        return xp.hstack([qp[3:], self.acceleration(qp[:3], t)])

    @partial_jit(static_argnames=("Integrator", "integrator_kw"))
    def integrate_orbit(
        self,
        w0: jt.Array,
        t0: jt.Array,
        t1: jt.Array,
        ts: jt.Array | None,
        *,
        Integrator: type[AbstractIntegrator] = DiffraxIntegrator,
        integrator_kw: dict[str, Any] | None = None,
    ) -> jt.Array:
        return Integrator(self._vel_acc, **(integrator_kw or {})).run(w0, t0, t1, ts)


# ===========================================================================


class AbstractPotential(AbstractPotentialBase):
    _: KW_ONLY
    units: UnitSystem = eqx.field(
        default=None,
        converter=lambda x: dimensionless if x is None else UnitSystem(x),
        static=True,
    )
    _G: float = eqx.field(init=False, static=True, repr=False)

    def __post_init__(self) -> None:
        self._init_units()
