"""Input parsing shared by the action-angle-frequency estimators."""

__all__ = ["parse_qpt", "parse_w", "wrap_result"]

from jaxtyping import Array
from typing import Any

from plum import convert

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.coordinates as gc
from .result import ActionsAnglesFrequencies


def parse_qpt(
    pot: Any, q: Any, p: Any, t: Any, /
) -> tuple[Array, Array | None, Array, bool]:
    """Strip ``q``, ``p`` and ``t`` to arrays in ``pot.units``.

    ``p=None`` means positions and time only (``p`` is returned as `None`).
    The flag is True if any *given* input is a Quantity, i.e. if the result
    should carry units. Unit-less inputs are assumed to be in ``pot.units``.
    """
    given = (q, t) if p is None else (q, p, t)
    as_quantity = any(isinstance(x, u.AbstractQuantity) for x in given)
    usys = pot.units
    q_: Array = jnp.asarray(u.ustrip(AllowValue, usys["length"], q))
    p_: Array | None = (
        None if p is None else jnp.asarray(u.ustrip(AllowValue, usys["speed"], p))
    )
    t_: Array = jnp.asarray(u.ustrip(AllowValue, usys["time"], t))
    return q_, p_, t_, as_quantity


def parse_w(
    pot: Any, w: gc.AbstractPhaseSpaceCoordinate, /
) -> tuple[Array, Array, Array]:
    """Strip a phase-space coordinate to ``(q, p, t)`` arrays in ``pot.units``.

    ``t`` is broadcast against the batch shape of ``q``.
    """
    w = w.to_frame(gc.frames.simulation_frame)
    xq: u.Quantity = convert(cx.vconvert(cx.CartesianPos3D, w.q), u.Quantity)
    vq: u.Quantity = convert(cx.vconvert(cx.CartesianVel3D, w.p, w.q), u.Quantity)
    q, p, t, _ = parse_qpt(pot, xq, vq, w.t)
    assert p is not None  # noqa: S101
    return q, p, jnp.broadcast_to(t, q.shape[:-1])


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
