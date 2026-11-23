"""Circulation of orbits about the coordinate axes.

This is private API.

"""

__all__ = ["circulation"]

import functools as ft
from dataclasses import replace

from jaxtyping import Array, ArrayLike, Bool
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

# ===================================================================
# Array cores


def _to_float(x: Any) -> Any:
    """Cast ints to the default float and half precision to float32.

    This runs before the unit conversion, which would otherwise underflow or
    overflow in half precision and truncate integers.
    """
    x = x if hasattr(x, "dtype") else jnp.asarray(x)
    floating = jnp.issubdtype(x.dtype, jnp.floating)
    return x.astype(jnp.promote_types(x.dtype, jnp.float32) if floating else float)


def _unit_scale(x: Array) -> Array:
    """Divide each sample of ``x`` by a power of 2 near its largest component.

    Dividing by a power of 2 is exact, so the signs and zeros of ``q x p`` do
    not change unless a component underflows.
    """
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
    eps = max(jnp.finfo(q.dtype).eps, jnp.finfo(p.dtype).eps)
    qp = jnp.linalg.norm(q, axis=-1, keepdims=True) * jnp.linalg.norm(
        p, axis=-1, keepdims=True
    )
    floor = max(_FLOOR, _FLOOR_EPS * eps) * qp
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
    q = jnp.asarray(u.ustrip(AllowValue, _Q_UNIT, _to_float(q)))
    p = jnp.asarray(u.ustrip(AllowValue, _P_UNIT, _to_float(p)))
    _check_shape(q, p)
    q, p = jax.lax.stop_gradient(q), jax.lax.stop_gradient(p)
    return _circulation_core(q, p)  # type: ignore[no-any-return]


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
def circulation(orbit: AbstractOrbit, t: Any, /) -> Bool[Array, "*batch 3"]:
    if t is None:
        return _circulation_qp(*_cartesian_qp(orbit))
    if jnp.ndim(t) != 1:
        msg = (
            f"circulation(orbit, t) needs t of shape (T,), but t has shape "
            f"{jnp.shape(t)}."
        )
        raise ValueError(msg)
    return _circulation_qp(*_cartesian_qp(orbit(t)))
