"""Circulation of orbits about the coordinate axes.

This is private API.

"""

__all__ = ["align_circulation", "circulation"]

from dataclasses import replace

from jaxtyping import Array, ArrayLike, Bool, Int
from typing import Any, Final

import jax
import numpy as np
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
from .base import AbstractOrbit

# The smallest |L_i| that counts as circulation, relative to |L|.
_FLOOR: Final = 1e-13

# The permutations that swap x or y with z.
_PERM_X: Final = np.array([2, 1, 0])
_PERM_Y: Final = np.array([0, 2, 1])


# ===================================================================
# Array cores


def _value(x: Any) -> Array:
    """Return the numbers of an array or a Quantity."""
    return jnp.asarray(x.value if isinstance(x, u.AbstractQuantity) else x)  # type: ignore[no-any-return]


def _circulation_one(q: Array, p: Array) -> Bool[Array, "3"]:
    """Return which axes one orbit of shape (T, 3) circulates about."""
    ok = jnp.isfinite(q).all(axis=1) & jnp.isfinite(p).all(axis=1)
    L = jnp.cross(q, p)
    same_sign = jnp.sign(L) == jnp.sign(L[jnp.argmax(ok)])
    big = jnp.abs(L) >= _FLOOR * jnp.linalg.norm(L, axis=1, keepdims=True)
    nonzero = L != 0
    return jnp.all(same_sign & big & nonzero | ~ok[:, None], axis=0)  # type: ignore[no-any-return]


def _circulation_qp(q: Any, p: Any) -> Bool[Array, "*batch 3"]:
    """Vmap the single-orbit core over the batch axes of (..., T, 3) inputs."""
    q, p = _value(q), _value(p)
    q, p = jnp.broadcast_arrays(q, p)
    q, p = jax.lax.stop_gradient(q), jax.lax.stop_gradient(p)
    fn = _circulation_one
    for _ in range(q.ndim - 2):
        fn = jax.vmap(fn)
    return fn(q, p)


def _circ_axis(circ: Bool[ArrayLike, "*batch 3"]) -> Int[Array, "*batch"]:
    """Return the axis to swap with z: 2 for none, else x before y."""
    c: Array = jnp.asarray(circ, dtype=bool)
    keep = c[..., 2] | ~(c[..., 0] | c[..., 1])
    return jnp.where(keep, 2, jnp.where(c[..., 0], 0, 1))  # type: ignore[no-any-return]


def _swap[QP: (Array, u.AbstractQuantity)](x: QP, axis: Int[Array, "*batch"]) -> QP:
    """Swap the last-axis component ``axis`` of ``x`` with z, per orbit."""
    a = axis[..., None, None]
    return jnp.where(a == 0, x[..., _PERM_X], jnp.where(a == 1, x[..., _PERM_Y], x))  # type: ignore[no-any-return]


def _cartesian_qp(w: Any) -> tuple[u.AbstractQuantity, u.AbstractQuantity]:
    """Return Cartesian q and p of a coordinate as Quantities of shape (..., 3)."""
    q: u.AbstractQuantity = convert(cx.vconvert(cx.CartesianPos3D, w.q), u.Quantity)
    p: u.AbstractQuantity = convert(
        cx.vconvert(cx.CartesianVel3D, w.p, w.q), u.Quantity
    )
    return q, p


# ===================================================================
# circulation


@dispatch
def circulation(q: Any, p: Any, /) -> Bool[Array, "*batch 3"]:
    return _circulation_qp(q, p)


@dispatch
def circulation(w: gc.AbstractPhaseSpaceCoordinate, /) -> Bool[Array, "*batch 3"]:
    return _circulation_qp(*_cartesian_qp(w))


@dispatch
def circulation(w: gc.AbstractPhaseSpaceCoordinate, t: Any, /) -> Any:  # noqa: ARG001
    msg = (
        f"circulation({type(w).__name__}, t) is not supported: a phase-space "
        "coordinate has no interpolant to evaluate at new times. Pass an "
        "AbstractOrbit integrated with dense=True, or call circulation(w) to "
        "use the stored samples."
    )
    raise TypeError(msg)


@dispatch
def circulation(orbit: AbstractOrbit, /) -> Bool[Array, "*batch 3"]:
    return _circulation_qp(*_cartesian_qp(orbit))


@dispatch
def circulation(orbit: AbstractOrbit, t: Any, /) -> Bool[Array, "*batch 3"]:
    return _circulation_qp(*_cartesian_qp(orbit(t)))


# ===================================================================
# align_circulation


@dispatch
def align_circulation(
    q: Any, p: Any, circ: Any = None, /
) -> tuple[Any, Any, Int[Array, "*batch"]]:
    if circ is None:
        circ = _circulation_qp(q, p)
    axis = _circ_axis(circ)
    q, p = jnp.broadcast_arrays(q, p)
    return _swap(q, axis), _swap(p, axis), axis


@dispatch
def align_circulation(
    w: gc.AbstractBasicPhaseSpaceCoordinate, circ: Any = None, /
) -> tuple[gc.AbstractBasicPhaseSpaceCoordinate, Int[Array, "*batch"]]:
    q, p = _cartesian_qp(w)
    q2, p2, axis = align_circulation(q, p, circ)
    changes: dict[str, Any] = {
        "q": cx.CartesianPos3D.from_(q2),
        "p": cx.CartesianVel3D.from_(p2),
    }
    if isinstance(w, AbstractOrbit):
        # The interpolant still describes the unswapped axes, so drop it.
        changes["interpolant"] = None
    return replace(w, **changes), axis
