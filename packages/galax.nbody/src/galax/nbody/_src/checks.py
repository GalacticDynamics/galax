"""Checks of masses and softening shared by `System` and `NBodyField`."""

__all__: list[str] = []

from jaxtyping import Array

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

MASS_MSG = "masses must be finite and non-negative"
SOFT_MSG = "softening must be finite and non-negative"


def tracer_msg(n_massive: int) -> str:
    return (
        f"masses after the first n_massive={n_massive} rows must be zero; "
        "those rows are massless tracers"
    )


def input_oks(m: Array, h: Array, n_massive: int) -> tuple[Array, Array, Array]:
    """Return the per-row masks for valid masses, tracer masses and softening."""
    # `>= 0` is False for NaN; `isfinite` also rejects +-inf.
    return jnp.isfinite(m) & (m >= 0), m[n_massive:] == 0, jnp.isfinite(h) & (h >= 0)


def check(x: Array, ok: Array, msg: str) -> Array:
    """Return ``x``, raising if any of ``ok`` is False.

    A concrete ``ok`` raises `ValueError` now. A traced one is checked at run
    time by `equinox.error_if`.
    """
    if isinstance(ok, jax.core.Tracer):
        out: Array = eqx.error_if(x, ~jnp.all(ok), msg)
        return out
    if not np.all(np.asarray(ok)):
        raise ValueError(msg)
    return x
