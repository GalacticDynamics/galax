"""Circulation detection and axis alignment for O2GF orbits (pure JAX)."""

__all__ = ["align_circulation_with_z", "circulating_axes", "is_loop"]

from jaxtyping import Array, Bool, Int
from typing import Final

import jax.numpy as jnp
import numpy as np

_FLOOR: Final = 1e-13
_FLOOR_EPS: Final = 64


# This duplicates `galax.orbits.circulation` for Cartesian (q, p) arrays, and
# tests/integration/test_o2gf_circulation.py checks that the two agree.
# Unifying them would be nicer, but galax.actionangle does not depend on
# galax.orbits, which keeps galax.orbits free to use action-angles later, and
# galax.coordinates has no notion of circulation to share it from.
def circulating_axes(
    q: Array, p: Array, ok: Bool[Array, " T"] | None = None
) -> Bool[Array, "3"]:
    """Return which axes the orbit circulates about, from its kept samples.

    ``q`` and ``p`` have shape ``(T, 3)``. Axis ``i`` circulates when ``L_i``
    keeps one sign at every kept sample and stays at or above
    ``max(1e-13, 64 eps) |q| |p|``. ``ok`` marks the kept samples; by default
    they are those where ``q`` and ``p`` are finite. An orbit with no kept
    samples circulates about no axis.
    """
    finite = jnp.isfinite(q).all(axis=1) & jnp.isfinite(p).all(axis=1)
    ok = finite if ok is None else ok & finite
    # Divide each sample by a power of 2, which is exact, so |L|^2 cannot
    # overflow and the signs of L do not change.
    q = jnp.ldexp(q, -jnp.frexp(jnp.max(jnp.abs(q), axis=1, keepdims=True))[1])
    p = jnp.ldexp(p, -jnp.frexp(jnp.max(jnp.abs(p), axis=1, keepdims=True))[1])
    L = jnp.cross(q, p)
    eps = max(float(np.finfo(q.dtype).eps), float(np.finfo(p.dtype).eps))
    qp = jnp.linalg.norm(q, axis=1, keepdims=True) * jnp.linalg.norm(
        p, axis=1, keepdims=True
    )
    big = jnp.abs(L) >= max(_FLOOR, _FLOOR_EPS * eps) * qp
    skip = ~ok[:, None]
    pos = jnp.all((L > 0) & big | skip, axis=0)
    neg = jnp.all((L < 0) & big | skip, axis=0)
    return (pos | neg) & ok.any()


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
