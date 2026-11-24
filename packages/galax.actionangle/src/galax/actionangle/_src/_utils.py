"""Helpers for the action-angle module."""

__all__ = [
    "G_in_units",
    "fill_masked",
    "finite_rows",
    "masked_span",
    "mod_2pi",
    "safe_div",
    "safe_sqrt",
]

import functools as ft
import operator

from typing import Any, Final

import jax
import jax.numpy as jnp

_TWO_PI: Final = 2 * jnp.pi


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


def finite_rows(*xs: jax.Array) -> jax.Array:
    """Mark the rows (leading axis) where every input is finite."""
    rows = (jnp.isfinite(x).reshape(len(x), -1).all(axis=1) for x in xs)
    return ft.reduce(operator.and_, rows)


def masked_span(x: jax.Array, mask: jax.Array) -> jax.Array:
    """``max - min`` of ``x`` along axis 0 over ``mask``; 0 where none is kept."""
    span = jnp.where(mask, x, -jnp.inf).max(axis=0) - jnp.where(mask, x, jnp.inf).min(
        axis=0
    )
    return jnp.where(mask.any(axis=0), span, 0.0)


def mod_2pi(x: jax.Array, /) -> jax.Array:
    """``x`` modulo 2 pi, in ``[0, 2 pi)``; `jnp.mod` can round up to 2 pi."""
    x = jnp.mod(x, _TWO_PI)
    return jnp.where(x >= _TWO_PI, x - _TWO_PI, x)


def G_in_units(pot: Any, /) -> Any:
    """Return the gravitational constant as a bare value in ``pot.units``."""
    usys = pot.units
    return pot.constants["G"].ustrip(
        usys["length"] ** 3 / (usys["mass"] * usys["time"] ** 2)
    )
