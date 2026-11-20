"""Helpers for the action-angle module."""

__all__ = ["safe_div", "safe_sqrt"]

import functools as ft

import jax
import jax.numpy as jnp


@ft.partial(jax.jit, inline=True)
def safe_sqrt(q2: jax.Array, /) -> jax.Array:
    """Square root with a finite derivative at 0 (copy of galax.potential's)."""
    return jnp.sqrt(q2 + jnp.finfo(jnp.promote_types(q2.dtype, float)).tiny)  # type: ignore[no-untyped-call]


@ft.partial(jax.jit, inline=True)
def safe_div(a: jax.Array, b: jax.Array, /) -> jax.Array:
    """Divide `a / b`, giving 0 where `b == 0`, with a NaN-free gradient."""
    zero = b == 0
    return jnp.where(zero, 0.0, a / jnp.where(zero, 1.0, b))
