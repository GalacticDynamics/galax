"""Analytic action-angle-frequency transform for the harmonic oscillator.

Follows gala's ``harmonic_oscillator_xv_to_aa``: each axis is an independent
oscillator with ``J_i = (v_i^2 + omega_i^2 x_i^2) / (2 omega_i)``. The angle
is ``atan2(-v_i / omega_i, x_i)`` taken modulo ``2 pi``, which agrees with
gala's branch handling and has no special cases.
"""

__all__: list[str] = []

import functools as ft

from jaxtyping import Array
from typing import Any

import jax
import jax.numpy as jnp
from plum import dispatch

import unxt as u

import galax.coordinates as gc
from ._parse import parse_qpt, parse_w, wrap_result
from ._utils import mod_2pi
from .result import ActionsAnglesFrequencies
from galax.potential import HarmonicOscillatorPotential


def _ho_aaf(omega: Array, x: Array, v: Array) -> tuple[Array, Array, Array]:
    """Actions, angles and frequencies of one point, in the (x, y, z) order.

    ``omega``, ``x`` and ``v`` have shape (3,).
    """
    actions = (v**2 + (omega * x) ** 2) / (2 * omega)
    # atan2(0, 0) has a NaN gradient, so an axis at rest takes atan2(0, 1)
    rest = (x == 0) & (v == 0)
    y_ = jnp.where(rest, 0.0, -v / omega)
    angles = mod_2pi(jnp.arctan2(y_, jnp.where(rest, 1.0, x)))
    return actions, angles, omega


@ft.partial(jax.jit, inline=True)
def _aaf(
    pot: HarmonicOscillatorPotential, q: Array, p: Array, t: Array
) -> tuple[Array, Array, Array]:
    """Unitless batched transform; ``q``, ``p`` (..., 3) and ``t`` (...)."""
    usys = pot.units
    tq = u.Q(t, usys["time"])
    omega = pot.omega(tq, ustrip=usys["frequency"])
    omega = jnp.broadcast_to(jnp.asarray(omega), (*q.shape[:-1], 3))
    f = jnp.vectorize(_ho_aaf, signature="(3),(3),(3)->(3),(3),(3)")
    out: tuple[Array, Array, Array] = f(omega, q, p)
    return out


@dispatch
def harmonic_oscillator_actions_angles_freqs(
    pot: HarmonicOscillatorPotential, q: Any, p: Any, t: Any, /
) -> ActionsAnglesFrequencies:
    """Actions, angles and frequencies from positions, velocities and time."""
    q, p, t, as_quantity = parse_qpt(pot, q, p, t)
    return wrap_result(*_aaf(pot, q, p, t), pot, as_quantity)


@dispatch
def harmonic_oscillator_actions_angles_freqs(
    pot: HarmonicOscillatorPotential, w: gc.AbstractPhaseSpaceCoordinate, /
) -> ActionsAnglesFrequencies:
    """Actions, angles and frequencies of a phase-space coordinate or orbit."""
    q, p, t = parse_w(pot, w)
    return wrap_result(*_aaf(pot, q, p, t), pot, as_quantity=True)
