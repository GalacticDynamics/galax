"""Circulation of orbits about the coordinate axes.

This is private API.

"""

__all__ = ["align_circulation", "circulation"]

import functools as ft
from dataclasses import fields, replace

from jaxtyping import Array, ArrayLike, Bool, Int
from typing import Any, Final, TypeAlias

import jax
import numpy as np
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.coordinates as gc
from .base import AbstractOrbit

# The smallest |L_i| that counts as circulation, relative to |q| |p|, is
# max(_FLOOR, _FLOOR_EPS * eps), where eps is the machine epsilon of the
# coarser float dtype of q and p. Round-off in q x p for a radial sample is
# about eps (measured at most 0.65 eps in float32), so the second term keeps
# float32 radial samples from circulating; for float64 _FLOOR_EPS * eps is
# below _FLOOR.
_FLOOR: Final = 1e-13
_FLOOR_EPS: Final = 64

# The units q and p are stripped into. The result depends only on the signs of
# q x p and on a floor relative to |q| |p|, so it does not depend on this choice.
# Quantities of the wrong dimension raise; plain arrays are taken to be in a
# consistent system and pass through unchanged.
_Q_UNIT: Final = "kpc"
_P_UNIT: Final = "kpc/Myr"

# The permutations that swap x or y with z.
_PERM_X: Final = np.array([2, 1, 0])
_PERM_Y: Final = np.array([0, 2, 1])


# ===================================================================
# Array cores


def _unit_scale(x: Array) -> Array:
    """Divide each sample of ``x`` by a power of 2 near its largest component.

    Dividing by a power of 2 is exact, so the signs and zeros of ``q x p`` do
    not change unless a component underflows.
    """
    # Ints become the default float, half precision becomes float32.
    if jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    else:
        x = x.astype(float)
    m = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    # Clearing the mantissa bits of m leaves the power of 2 at or below it.
    # jnp.frexp would do the same, but is several times slower.
    info = jnp.finfo(x.dtype)
    uint = jnp.dtype(f"uint{info.bits}")
    mask = ((1 << (info.bits - 1)) - 1) & ~((1 << info.nmant) - 1)
    bits = jax.lax.bitcast_convert_type(m, uint) & np.array(mask, dtype=uint)
    s = jax.lax.bitcast_convert_type(bits, x.dtype)
    return x / jnp.where(s == 0, 1, s)  # type: ignore[no-any-return]


@ft.partial(jax.jit, inline=True)
def _circulation_core(q: Array, p: Array) -> Bool[Array, "*batch 3"]:
    """Return which axes orbits of shape (..., T, 3) circulate about."""
    ok = (jnp.isfinite(q) & jnp.isfinite(p)).all(axis=-1, keepdims=True)
    # Scale each sample to order 1 so that |L|^2 cannot overflow. A positive
    # scale does not change the signs of L.
    q, p = _unit_scale(q), _unit_scale(p)
    L = jnp.cross(q, p)
    # The floor is relative to |q| |p|, not |L|, so the rounding residual of a
    # radial sample (q parallel to p) does not count as circulation.
    floor = max(
        _FLOOR, _FLOOR_EPS * max(jnp.finfo(q.dtype).eps, jnp.finfo(p.dtype).eps)
    ) * (
        jnp.linalg.norm(q, axis=-1, keepdims=True)
        * jnp.linalg.norm(p, axis=-1, keepdims=True)
    )
    big = jnp.abs(L) >= floor
    # An axis circulates if L_i keeps one sign, clear of the floor, at every
    # finite sample.
    pos = jnp.all((L > 0) & big | ~ok, axis=-2)
    neg = jnp.all((L < 0) & big | ~ok, axis=-2)
    # An orbit with no finite samples circulates about no axis.
    return (pos | neg) & ok.any(axis=-2)  # type: ignore[no-any-return]


def _check_shape(q: Any, p: Any) -> tuple[int, ...]:
    """Check that q and p broadcast to shape (..., T, 3) with T > 0."""
    shape = jnp.broadcast_shapes(jnp.shape(q), jnp.shape(p))
    if len(shape) < 2 or shape[-1] != 3 or shape[-2] == 0:
        msg = (
            f"circulation needs q and p of shape (..., T, 3) with T > 0, but "
            f"they broadcast to shape {shape}."
        )
        raise ValueError(msg)
    return shape  # type: ignore[no-any-return]


def _circulation_qp(q: Any, p: Any) -> Bool[Array, "*batch 3"]:
    """Check the shapes of (..., T, 3) inputs and run the jitted core."""
    q = jnp.asarray(u.ustrip(AllowValue, _Q_UNIT, q))
    p = jnp.asarray(u.ustrip(AllowValue, _P_UNIT, p))
    _check_shape(q, p)
    q, p = jax.lax.stop_gradient(q), jax.lax.stop_gradient(p)
    return _circulation_core(q, p)  # type: ignore[no-any-return]


def _circ_axis(circ: Bool[ArrayLike, "*batch 3"]) -> Int[Array, "*batch"]:
    """Return the axis to swap with z: 2 for none, else x before y."""
    c: Array = jnp.asarray(circ, dtype=bool)
    keep = c[..., 2] | ~(c[..., 0] | c[..., 1])
    return jnp.where(keep, 2, jnp.where(c[..., 0], 0, 1))  # type: ignore[no-any-return]


@ft.partial(jax.jit, inline=True)
def _swap[QP: (Array, u.AbstractQuantity)](x: QP, axis: Int[Array, "*batch"]) -> QP:
    """Swap the last-axis component ``axis`` of ``x`` with z, per orbit."""
    a = jnp.reshape(axis, axis.shape + (1,) * (jnp.ndim(x) - axis.ndim))
    return jnp.where(a == 0, x[..., _PERM_X], jnp.where(a == 1, x[..., _PERM_Y], x))  # type: ignore[no-any-return]


def _broadcast(v: Any, shape: tuple[int, ...]) -> Any:
    """Broadcast the components of ``v``, not its other fields (e.g. Delta)."""
    return replace(
        v, **{c: jnp.broadcast_to(getattr(v, c), shape) for c in v.components}
    )


def _cartesian_qp(w: Any) -> tuple[u.AbstractQuantity, u.AbstractQuantity]:
    """Return Cartesian q and p of a coordinate as Quantities of shape (..., 3)."""
    if isinstance(w, gc.AbstractCompositePhaseSpaceCoordinate):
        # Convert each component before joining them: the composite's own
        # ``q`` joins the raw vectors, which fails when their representations
        # differ.
        qs, ps = zip(*(_cartesian_qp(v) for v in w.values()), strict=True)
        order = w._time_sorter  # noqa: SLF001
        return (
            jnp.concat([jnp.atleast_2d(x) for x in qs], axis=-2)[..., order, :],
            jnp.concat([jnp.atleast_2d(x) for x in ps], axis=-2)[..., order, :],
        )
    # coordinax's velocity vconvert fails to broadcast with 2 or more batch
    # axes, so flatten them around the conversion.
    # q and p of one coordinate need not have the same batch shape, so
    # broadcast them first.
    shape = jnp.broadcast_shapes(w.q.shape, w.p.shape)
    wq, wp = w.q, w.p
    if len(shape) > 1 or wq.shape != wp.shape:
        wq, wp = _broadcast(wq, shape).reshape(-1), _broadcast(wp, shape).reshape(-1)
    q: u.AbstractQuantity = convert(cx.vconvert(cx.CartesianPos3D, wq), u.Quantity)
    p: u.AbstractQuantity = convert(cx.vconvert(cx.CartesianVel3D, wp, wq), u.Quantity)
    return q.reshape(*shape, 3), p.reshape(*shape, 3)


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
def circulation(orbit: AbstractOrbit, t: None, /) -> Bool[Array, "*batch 3"]:  # noqa: ARG001
    return _circulation_qp(*_cartesian_qp(orbit))


@dispatch
def circulation(orbit: AbstractOrbit, t: Any, /) -> Bool[Array, "*batch 3"]:
    if jnp.ndim(t) != 1:
        msg = (
            f"circulation(orbit, t) needs t of shape (T,), but t has shape "
            f"{jnp.shape(t)}."
        )
        raise ValueError(msg)
    return _circulation_qp(*_cartesian_qp(orbit(t)))


# ===================================================================
# align_circulation


def _align_axis(q: Any, p: Any, circ: Any) -> Int[Array, "*batch"]:
    """Return the axis to swap with z, computing and checking ``circ``."""
    if circ is None:
        return _circ_axis(_circulation_qp(q, p))
    batch = _check_shape(q, p)[:-2]
    if jnp.shape(circ) != (*batch, 3):
        msg = (
            f"circ has shape {jnp.shape(circ)}, but the orbits need shape "
            f"{(*batch, 3)}: one row of 3 booleans per orbit."
        )
        raise ValueError(msg)
    return _circ_axis(circ)


def _vconvert(rep: Any, v: Any, at: Any = None, **kwargs: Any) -> Any:
    """Convert a vector to ``rep``, flattening 2 or more batch axes.

    coordinax's velocity ``vconvert`` fails to broadcast with 2 or more batch
    axes.
    """
    shape = v.shape
    if len(shape) >= 2:
        v = v.reshape(-1)
        at = None if at is None else at.reshape(-1)
    out = cx.vconvert(rep, v, **kwargs) if at is None else cx.vconvert(rep, v, at)
    return out if len(shape) < 2 else out.reshape(shape)


def _like(old: Any, new: Any, keep: Any) -> Any:
    """Return ``old`` with its components from ``new`` where ``keep`` is False.

    Each component keeps the unit of ``old``'s, and its dtype if that is a
    floating one. Fields that are not components, e.g. ``Delta`` of a
    prolate spheroidal vector, are kept.
    """
    comps = {}
    for c in old.components:
        o = getattr(old, c)
        n = u.uconvert(u.unit_of(o), getattr(new, c))
        # Keep a floating dtype; an integer one would truncate the result.
        if jnp.issubdtype(o.dtype, jnp.floating):
            n = n.astype(o.dtype)
        comps[c] = jnp.where(keep, o, n)
    return replace(old, **comps)


def _aligned(w: Any, axis: Int[Array, "*batch"]) -> Any:
    """Rebuild a coordinate with its circulation axis swapped with z."""
    if isinstance(w, gc.AbstractCompositePhaseSpaceCoordinate):
        # The components share the composite's batch axes, so each one is
        # swapped with the same axis.
        parts = {k: _aligned(v, axis) for k, v in w.items()}
        return type(w)(parts, frame=w.frame)
    # An orbit with axis 2 keeps its input. Its round trip through Cartesian
    # can be NaN, e.g. on the z axis in spherical coordinates, so those rows
    # go through the conversion as an off-axis point that the where drops,
    # which keeps the gradient finite.
    # Shape keep like the components, (*batch, T), or () for a single point.
    ndim = len(jnp.broadcast_shapes(w.q.shape, w.p.shape))
    keep = jnp.reshape(axis == 2, axis.shape + (1,) * (ndim - axis.ndim))
    q, p = _cartesian_qp(w)
    q = jnp.where(keep[..., None], jnp.ones_like(q), _swap(q, axis))
    p = jnp.where(keep[..., None], jnp.ones_like(p), _swap(p, axis))
    extra = {
        f.name: getattr(w.q, f.name)
        for f in fields(w.q)
        if f.name not in w.q.components
    }
    qn = _vconvert(type(w.q), cx.CartesianPos3D.from_(q), **extra)
    pn = _vconvert(type(w.p), cx.CartesianVel3D.from_(p), qn)
    changes: dict[str, Any] = {"q": _like(w.q, qn, keep), "p": _like(w.p, pn, keep)}
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
