"""Input parsing shared by the action-angle-frequency estimators."""

__all__ = ["parse_qpt", "parse_w", "wrap_result"]

from jaxtyping import Array
from typing import Any, overload

from plum import convert

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.coordinates as gc
from .result import ActionsAnglesFrequencies


def _float(x: Array) -> Array:
    """Promote to a strongly-typed float of at least 32 bits; reject complex.

    Ints and bools become the default float. Half precision (float16,
    bfloat16) becomes float32: the round-off tolerances assume >= float32.
    """
    if jnp.iscomplexobj(x):
        msg = f"complex input is not supported, got dtype {x.dtype}"
        raise TypeError(msg)
    dtype = (
        jnp.promote_types(x.dtype, "float32")
        if jnp.issubdtype(x.dtype, jnp.floating)
        else float
    )
    out: Array = jnp.asarray(x, dtype=dtype)
    return out


@overload
def parse_qpt(
    pot: Any, q: Any, p: None, t: Any, /
) -> tuple[Array, None, Array, bool]: ...


@overload
def parse_qpt(
    pot: Any, q: Any, p: Any, t: Any, /
) -> tuple[Array, Array, Array, bool]: ...


def parse_qpt(
    pot: Any, q: Any, p: Any, t: Any, /
) -> tuple[Array, Array | None, Array, bool]:
    """Strip ``q``, ``p`` and ``t`` to arrays in ``pot.units``.

    ``p=None`` means positions and time only (``p`` is returned as `None`).
    The flag is True if any *given* input is a Quantity, i.e. if the result
    should carry units. Unit-less inputs are assumed to be in ``pot.units``.
    Integer inputs are promoted to float, half precision to float32, and
    complex inputs raise `TypeError`, as does a phase-space coordinate
    passed as ``q``. The batch shapes of ``q``, ``p``
    (``shape[:-1]``) and ``t`` are broadcast together.
    """
    if isinstance(q, gc.AbstractPhaseSpaceCoordinate):
        msg = (
            "q is a phase-space coordinate: it already carries its times; "
            "pass (pot, w)"
        )
        raise TypeError(msg)
    given = (q, t) if p is None else (q, p, t)
    as_quantity = any(isinstance(x, u.AbstractQuantity) for x in given)
    usys = pot.units
    q_: Array = _float(jnp.asarray(u.ustrip(AllowValue, usys["length"], q)))
    p_: Array | None = (
        None
        if p is None
        else _float(jnp.asarray(u.ustrip(AllowValue, usys["speed"], p)))
    )
    if t is None:
        msg = "t is required (the potential may depend on time)"
        raise ValueError(msg)
    # a strongly-typed float t, so int / weakly-typed t don't retrace the core
    t_: Array = _float(jnp.asarray(u.ustrip(AllowValue, usys["time"], t)))
    for name, x in (("q", q_), ("p", p_)):
        if x is not None and x.shape[-1:] != (3,):
            msg = f"{name} must have shape (..., 3), got {x.shape}"
            raise ValueError(msg)

    shapes = [q_.shape[:-1], t_.shape] + ([] if p_ is None else [p_.shape[:-1]])
    batch = jnp.broadcast_shapes(*shapes)
    q_ = jnp.broadcast_to(q_, (*batch, q_.shape[-1]))
    if p_ is not None:
        p_ = jnp.broadcast_to(p_, (*batch, p_.shape[-1]))
    return q_, p_, jnp.broadcast_to(t_, batch), as_quantity


def parse_w(
    pot: Any, w: gc.AbstractPhaseSpaceCoordinate, /
) -> tuple[Array, Array, Array]:
    """Strip a phase-space coordinate to ``(q, p, t)`` arrays in ``pot.units``.

    The batch shapes are broadcast together.
    """
    w = w.to_frame(gc.frames.simulation_frame)
    xq: u.Quantity = convert(cx.vconvert(cx.CartesianPos3D, w.q), u.Quantity)
    vq: u.Quantity = convert(cx.vconvert(cx.CartesianVel3D, w.p, w.q), u.Quantity)
    q, p, t, _ = parse_qpt(pot, xq, vq, w.t)
    return q, p, t


def wrap_result(
    actions: Array,
    angles: Array,
    freqs: Array,
    pot: Any,
    as_quantity: bool,  # noqa: FBT001
) -> ActionsAnglesFrequencies:
    """Attach units (length^2/time, rad, rad/time) if requested."""
    if as_quantity:
        usys = pot.units
        actions = u.Q(actions, usys["length"] ** 2 / usys["time"])
        angles = u.Q(angles, "rad")
        freqs = u.Q(freqs, u.unit("rad") / usys["time"])
    return ActionsAnglesFrequencies(actions, angles, freqs)
