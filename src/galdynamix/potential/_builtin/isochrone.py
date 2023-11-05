from __future__ import annotations

__all__ = ["Isochrone", "Isochrone_centered"]

from typing import Any, Callable

import jax
import jax.numpy as xp
import jax.typing as jt
from gala.units import UnitSystem

from galdynamix.potential._base import PotentialBase
from galdynamix.utils import jit_method


class Isochrone(PotentialBase):
    def __init__(
        self, m: jt.Array, a: jt.Array, units: UnitSystem | None = None
    ) -> None:
        self.m: jt.Array
        self.a: jt.Array
        super().__init__(units, {"m": m, "a": a})

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        r = xp.linalg.norm(q, axis=0)
        return -self._G * self.m / (self.a + xp.sqrt(r**2 + self.a**2))


class Isochrone_centered(PotentialBase):
    def __init__(
        self,
        m: jt.Array,
        a: jt.Array,
        spline_eval_func: Callable[[jt.Array, Any], jt.Array],
        splines: Any,
        t_min: jt.Array,
        t_max: jt.Array,
        m_ext: jt.Array,
        a_ext: jt.Array,
        units: UnitSystem | None = None,
    ) -> None:
        self.m: jt.Array
        self.a: jt.Array
        self.spline_eval_func: Callable[[jt.Array, Any], jt.Array]
        self.splines: Any
        self.t_min: jt.Array
        self.t_max: jt.Array
        self.m_ext: jt.Array
        self.a_ext: jt.Array
        super().__init__(
            units,
            {
                "m": m,
                "a": a,
                "spline_eval_func": spline_eval_func,
                "splines": splines,
                "t_min": t_min,
                "t_max": t_max,
                "m_ext": m_ext,
                "a_ext": a_ext,
            },
        )

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        is_cond_met = (t > self.t_min) & (t < self.t_max)  # True if yes, False if no
        pot_ext = Isochrone(m=self.m_ext, a=self.a_ext, units=self.units)

        def true_func(q_t: jt.Array) -> jt.Array:
            q_, t = q_t[:3], q_t[-1]
            q = q_ - self.spline_eval_func(t, self.splines)
            r = xp.linalg.norm(q, axis=0)
            return -self._G * self.m / (
                self.a + xp.sqrt(r**2 + self.a**2)
            ) + pot_ext.energy(
                q_, t
            )  # + self.pot_ext.energy(q_,t)

        def false_func(q_t: jt.Array) -> jt.Array:
            q, t = q_t[:3], q_t[-1]
            return pot_ext.energy(q, t)  # 0.#self.pot_ext.energy(q,t)

        q_t = xp.hstack([q, t])
        return jax.lax.cond(
            pred=is_cond_met, true_fun=true_func, false_fun=false_func, operand=q_t
        )
