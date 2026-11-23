"""Circulation detection and axis alignment for O2GF orbits."""

__all__ = ["align_circulation_with_z", "circulation", "is_loop"]

from jaxtyping import Array, Bool, Int
from typing import cast

import jax.numpy as jnp

import galax.orbits as go


def circulation(
    q: Array, p: Array, ok: Bool[Array, " T"] | None = None
) -> Bool[Array, "3"]:
    """Return which axes the orbit circulates about, from its kept samples.

    ``q`` and ``p`` have shape ``(T, 3)``. This is `galax.orbits.circulation`
    with the samples outside ``ok`` left out; by default it keeps those where
    ``q`` and ``p`` are finite.
    """
    if ok is not None:
        q = jnp.where(ok[:, None], q, jnp.nan)
    return cast("Bool[Array, '3']", go.circulation(q, p))


def is_loop(circ: Bool[Array, "3"]) -> Bool[Array, ""]:
    """Return True if the orbit circulates about any axis."""
    return jnp.any(circ)


def align_circulation_with_z(
    q: Array, p: Array, circ: Bool[Array, "3"]
) -> tuple[Array, Array, Int[Array, ""]]:
    """Swap the circulating axis with z in ``q`` and ``p``.

    Nothing changes if z circulates or no axis does, and ``axis`` is then 2.
    Otherwise ``axis`` is the first circulating axis among x and y, preferring x.
    """
    axis = jnp.where(circ[2] | ~(circ[0] | circ[1]), 2, jnp.where(circ[0], 0, 1))
    perm = jnp.where(
        axis == 0,
        jnp.array([2, 1, 0]),
        jnp.where(axis == 1, jnp.array([0, 2, 1]), jnp.array([0, 1, 2])),
    )
    return q[:, perm], p[:, perm], axis
