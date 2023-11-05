from __future__ import annotations

__all__ = []  # WIP

from functools import partial
from typing import Callable, Protocol

import jax
import jax.numpy as xp
import jax.typing as jt


def leapfrog_step(
    func: Callable[[jt.Array, jt.Array], jt.Array],
    y0: jt.Array,
    t0: jt.Array,
    dt: jt.Array,
    a0: jt.Array,
) -> jt.Array:
    """Leapfrog step for a single particle."""
    ndim = y0.shape[0] // 2
    tf = t0 + dt

    x0 = y0[:ndim]
    v0 = y0[ndim:]

    v1_2 = v0 + a0 * dt / 2.0
    xf = x0 + v1_2 * dt
    af = -func(xf, tf)

    vf = v1_2 + af * dt / 2

    return tf, xp.concatenate((xf, vf)), af


class _PotentialGradientCallable(Protocol):
    def __call__(self, y: jt.Array, t: jt.Array, *args: jt.Array) -> jt.Array:
        ...


@partial(jax.jit, static_argnames=["potential_gradient", "args"])
def leapfrog_run(
    w0: jt.Array,
    ts: jt.Array,
    potential_gradient: _PotentialGradientCallable,
    args: tuple[jt.Array, ...] = (),
) -> jt.Array:
    """Leapfrog integration for a single particle."""
    func_: Callable[[jt.Array, jt.Array], jt.Array]

    func_ = lambda y, t: potential_gradient(y, t, *args)  # noqa: E731

    def scan_fun(
        carry: tuple[int, jt.Array, jt.Array, jt.Array, jt.Array], _: jt.Array
    ) -> tuple[list[int | jt.Array], jt.Array]:
        i, y0, t0, dt, a0 = carry
        tf, yf, af = leapfrog_step(func_, y0, t0, dt, a0)
        dt_new = ts[i + 1] - ts[i]
        # NOTE: ADDED xp.abs AFTER derivs worked. Note for future debugging efforts!
        is_cond_met = xp.abs(dt_new) > 0.0

        def true_func(dt_new: jt.Array) -> jt.Array:
            # NOTE: ASSUMING dt = 0.5 Myr by default!
            return ts[-1] - ts[-2]

        def false_func(dt_new: jt.Array) -> jt.Array:
            return 0.0

        dt_new = jax.lax.cond(
            pred=is_cond_met, true_fun=true_func, false_fun=false_func, operand=dt_new
        )

        ###tf = tf + dt_new
        return [i + 1, yf, tf, dt_new, af], yf

    ndim = w0.shape[0] // 2
    a0 = -func_(w0[:ndim], ts[0])  # TODO: SHOULD THIS BE NEGATIVE???
    dt = ts[1] - ts[0]  ## I ADDED THIS
    init_carry = [0, w0, ts[0], dt, a0]
    _, ws = jax.lax.scan(scan_fun, init_carry, ts[1:])
    return xp.concatenate((w0[None], ws))
