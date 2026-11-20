"""Circulation detection and axis alignment for O2GF orbits (pure JAX)."""

__all__ = ["align_circulation", "circulation", "is_loop"]

from jaxtyping import Array, Bool, Int

import jax.numpy as jnp

_FLOOR = 1e-13


def circulation(
    q: Array, p: Array, ok: Bool[Array, " T"] | None = None
) -> Bool[Array, "3"]:
    """Return which axes the orbit circulates about.

    ``q`` and ``p`` have shape ``(T, 3)``.  Axis ``i`` circulates when ``L_i(t)``
    has the sign of ``L_i`` at the first kept sample at every kept sample and
    ``|L_i(t)| >= 1e-13 |L(t)|``. The floor is relative to ``|L|``, so it does
    not depend on the units. ``ok`` marks the kept samples; by default they
    are those where ``q`` and ``p`` are finite.
    """
    if ok is None:
        ok = jnp.isfinite(q).all(axis=1) & jnp.isfinite(p).all(axis=1)
    L = jnp.cross(q, p)
    same_sign = jnp.sign(L) == jnp.sign(L[jnp.argmax(ok)])
    big = jnp.abs(L) >= _FLOOR * jnp.linalg.norm(L, axis=1, keepdims=True)
    nonzero = L != 0
    return jnp.all(same_sign & big & nonzero | ~ok[:, None], axis=0)


def is_loop(circ: Bool[Array, "3"]) -> Bool[Array, ""]:
    """Return True if the orbit circulates about any axis."""
    return jnp.any(circ)


def align_circulation(
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
