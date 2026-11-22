"""Circulation of orbits about the coordinate axes.

This is private API.

"""

__all__ = ["align_circulation", "circulation"]

from dataclasses import replace

from jaxtyping import Array, ArrayLike, Bool, Int
from typing import Any, Final, TypeAlias

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
    circ = jnp.all(same_sign & big & nonzero | ~ok[:, None], axis=0)
    # An orbit with no finite samples circulates about no axis.
    return circ & ok.any()  # type: ignore[no-any-return]


def _circulation_qp(q: Any, p: Any) -> Bool[Array, "*batch 3"]:
    """Vmap the single-orbit core over the batch axes of (..., T, 3) inputs."""
    q, p = _value(q), _value(p)
    q, p = jnp.broadcast_arrays(q, p)
    if q.ndim < 2 or q.shape[-1] != 3 or q.shape[-2] == 0:
        msg = (
            f"circulation needs q and p of shape (..., T, 3) with T > 0, but "
            f"they broadcast to shape {q.shape}."
        )
        raise ValueError(msg)
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
    a = jnp.reshape(axis, axis.shape + (1,) * (jnp.ndim(x) - axis.ndim))
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

QPLike: TypeAlias = ArrayLike | u.AbstractQuantity


@dispatch
def circulation(q: QPLike, p: QPLike, /) -> Bool[Array, "*batch 3"]:
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


def _align_axis(q: Any, p: Any, circ: Any) -> Int[Array, "*batch"]:
    """Return the axis to swap with z, computing and checking ``circ``."""
    if circ is None:
        return _circ_axis(_circulation_qp(q, p))
    batch = jnp.broadcast_shapes(jnp.shape(q), jnp.shape(p))[:-2]
    if jnp.shape(circ) != (*batch, 3):
        msg = (
            f"circ has shape {jnp.shape(circ)}, but the orbits need shape "
            f"{(*batch, 3)}: one row of 3 booleans per orbit."
        )
        raise ValueError(msg)
    return _circ_axis(circ)


def _aligned(w: Any, axis: Int[Array, "*batch"]) -> Any:
    """Rebuild a coordinate with its circulation axis swapped with z."""
    if isinstance(w, gc.AbstractCompositePhaseSpaceCoordinate):
        # The components share the composite's batch axes, so each one is
        # swapped with the same axis.
        parts = {k: _aligned(v, axis) for k, v in w.items()}
        return type(w)(parts, frame=w.frame)
    q, p = _cartesian_qp(w)
    changes: dict[str, Any] = {
        "q": cx.CartesianPos3D.from_(_swap(q, axis)),
        "p": cx.CartesianVel3D.from_(_swap(p, axis)),
    }
    if isinstance(w, AbstractOrbit):
        # The interpolant still describes the unswapped axes, so drop it.
        changes["interpolant"] = None
    return replace(w, **changes)


@dispatch
def align_circulation(
    q: QPLike, p: QPLike, circ: Any = None, /
) -> tuple[Any, Any, Int[Array, "*batch"]]:
    axis = _align_axis(q, p, circ)
    q, p = jnp.broadcast_arrays(q, p)
    return _swap(q, axis), _swap(p, axis), axis


@dispatch
def align_circulation(
    w: gc.AbstractPhaseSpaceCoordinate, circ: Any = None, /
) -> tuple[gc.AbstractPhaseSpaceCoordinate, Int[Array, "*batch"]]:
    axis = _align_axis(*_cartesian_qp(w), circ)
    return _aligned(w, axis), axis
