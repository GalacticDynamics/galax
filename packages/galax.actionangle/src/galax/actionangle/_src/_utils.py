"""Helpers for the action-angle module."""

__all__ = ["safe_sqrt"]

import functools as ft

import jax
import jax.numpy as jnp


@ft.partial(jax.jit, inline=True)
def safe_sqrt(q2: jax.Array, /) -> jax.Array:
    """Square root with a finite derivative at 0 (copy of galax.potential's)."""
    return jnp.sqrt(q2 + jnp.finfo(jnp.promote_types(q2.dtype, float)).tiny)  # type: ignore[no-untyped-call]
