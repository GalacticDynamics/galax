"""Helpers for the action-angle module."""

__all__ = [
    "TWO_PI",
    "G_in_units",
    "fill_masked",
    "mod_2pi",
    "safe_div",
    "safe_sqrt",
]

import functools as ft

from typing import Any

import jax
import jax.numpy as jnp

TWO_PI = 2 * jnp.pi


@ft.partial(jax.jit, inline=True)
def safe_sqrt(q2: jax.Array, /) -> jax.Array:
    """Square root with a finite derivative at 0 (copy of galax.potential's)."""
    return jnp.sqrt(q2 + jnp.finfo(jnp.promote_types(q2.dtype, float)).tiny)  # type: ignore[no-untyped-call]


@ft.partial(jax.jit, inline=True)
def safe_div(a: jax.Array, b: jax.Array, /) -> jax.Array:
    """Divide `a / b`, giving 0 where `b == 0`, with a NaN-free gradient."""
    zero = b == 0
    return jnp.where(zero, 0.0, a / jnp.where(zero, 1.0, b))


def fill_masked(x: jax.Array, ok: jax.Array) -> jax.Array:
    """Replace rows where ``ok`` is False by the previous good row (else the next)."""
    i = jnp.arange(len(ok))
    prev = jax.lax.cummax(jnp.where(ok, i, -1))
    nxt = jax.lax.cummin(jnp.where(ok, i, len(ok)), reverse=True)
    idx = jnp.where(prev >= 0, prev, jnp.minimum(nxt, len(ok) - 1))
    return x[idx]


def mod_2pi(x: jax.Array, /) -> jax.Array:
    """``x`` modulo 2 pi, in ``[0, 2 pi)``; `jnp.mod` can round up to 2 pi."""
    x = jnp.mod(x, TWO_PI)
    return jnp.where(x >= TWO_PI, x - TWO_PI, x)


def G_in_units(pot: Any, /) -> Any:
    """Return the gravitational constant as a bare value in ``pot.units``."""
    usys = pot.units
    return pot.constants["G"].ustrip(
        usys["length"] ** 3 / (usys["mass"] * usys["time"] ** 2)
    )
