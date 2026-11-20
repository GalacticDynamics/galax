"""Toy potentials of the O2GF method, and their fits to an orbit (pure JAX).

A toy is an analytic potential whose actions, angles and frequencies are known.
`fit_isochrone` and `fit_oscillator` choose the toy that makes the toy energy
as constant as possible along one orbit, by minimising
``sum_t (H_toy(t) - mean H_toy)^2`` with Levenberg-Marquardt.  Everything is a
bare array in the units of the potential, so the functions vmap and jit, and
optimistix's implicit differentiation gives gradients of the fitted toy with
respect to the orbit.
"""

__all__ = [
    "IsochroneToy",
    "OscillatorToy",
    "fit_isochrone",
    "fit_oscillator",
    "toy_aaf",
]

from jaxtyping import Array, Bool
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from plum import dispatch

from .harmonic_oscillator import _ho_aaf
from .isochrone import _isochrone_aaf

_BISECT_STEPS = 20
# Weight of the pull toward the seed, relative to the mean |H| at the seed. It
# makes the least-squares problem full rank where H is flat in the parameters
# (circular orbits, an oscillator axis at rest) and is far too small to bias an
# orbit that does constrain them.
_TIKHONOV = 1e-6
_BRACKET = 1e3  # the isochrone seed searches b in [r0 / 1e3, 1e3 r0]


class IsochroneToy(eqx.Module):
    """Isochrone toy with mass ``m`` and scale length ``b``, in ``pot.units``."""

    m: Array
    b: Array


class OscillatorToy(eqx.Module):
    """Harmonic-oscillator toy with frequencies ``omega`` of shape (..., 3)."""

    omega: Array


# ---------------------------------------------------------------------------
# Energy and actions, angles, frequencies


def _isochrone_energy(m: Array, b: Array, G: Array, q: Array, p: Array) -> Array:
    r = jnp.linalg.norm(q, axis=-1)
    return 0.5 * jnp.sum(p**2, axis=-1) - G * m / (b + jnp.sqrt(b**2 + r**2))


def _oscillator_energy(omega: Array, q: Array, p: Array) -> Array:
    return 0.5 * jnp.sum(p**2 + (omega * q) ** 2, axis=-1)


@dispatch
def toy_aaf(toy: IsochroneToy, G: Any, q: Any, p: Any) -> tuple[Array, Array, Array]:
    """Actions, angles and frequencies, each (T, 3), of samples ``q``, ``p``."""
    return jax.vmap(lambda x, v: _isochrone_aaf(toy.m, toy.b, G, x, v))(q, p)


@dispatch
def toy_aaf(toy: OscillatorToy, G: Any, q: Any, p: Any) -> tuple[Array, Array, Array]:
    """Actions, angles and frequencies, each (T, 3); ``G`` is unused."""
    del G
    return jax.vmap(lambda x, v: _ho_aaf(toy.omega, x, v))(q, p)


# ---------------------------------------------------------------------------
# Fits


def _lm_fit(residual: Any, y0: Array) -> tuple[Array, Bool[Array, ""]]:
    sol = optx.least_squares(
        residual,
        optx.LevenbergMarquardt(rtol=1e-10, atol=1e-10),
        y0,
        max_steps=256,
        throw=False,
    )
    return sol.value, sol.result == optx.RESULTS.successful


def _isochrone_seed(pot: Any, G: Array, q: Array, t0: Array) -> tuple[Array, Array]:
    """Seed ``(m0, b0)`` from the potential at the mean radius, as gala does.

    The enclosed mass is ``r0^2 |dPhi/dr| / G`` at ``(r0, 0, 0)``; this is
    used instead of ``pot.mass_enclosed`` because it only sets the seed and
    works for every potential.  The scale
    length solves ``-G M0 / r0^3 a0 (b + a0) = Phi0`` with ``a0 = sqrt(r0^2 +
    b^2)`` by 20 bisection steps on ``log b``, falling back to ``b = r0`` when
    the bracket has no sign change.
    """
    r0 = jnp.mean(jnp.linalg.norm(q, axis=-1))
    x0 = jnp.array([r0, 0.0, 0.0])
    phi0 = pot._potential(x0, t0)  # noqa: SLF001
    dphi = jax.grad(pot._potential)(x0, t0)[0]  # noqa: SLF001
    m_enc = r0**2 * jnp.abs(dphi) / G

    def g(logb: Array) -> Array:
        b = jnp.exp(logb)
        a0 = jnp.sqrt(r0**2 + b**2)
        out: Array = -G * m_enc / r0**3 * a0 * (b + a0) - phi0
        return out

    lo0, hi0 = jnp.log(r0 / _BRACKET), jnp.log(r0 * _BRACKET)
    has_root = jnp.sign(g(lo0)) != jnp.sign(g(hi0))

    def step(_: int, lohi: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        left = jnp.sign(g(mid)) == jnp.sign(g(lo))
        return jnp.where(left, mid, lo), jnp.where(left, hi, mid)

    lo, hi = jax.lax.fori_loop(0, _BISECT_STEPS, step, (lo0, hi0))
    b = jnp.where(has_root, jnp.exp(0.5 * (lo + hi)), r0)
    a0 = jnp.sqrt(r0**2 + b**2)
    return m_enc / r0**3 * a0 * (b + a0) ** 2, b


def fit_isochrone(
    pot: Any, q: Array, p: Array, t: Array
) -> tuple[IsochroneToy, Bool[Array, ""]]:
    """Fit the isochrone toy to one orbit; ``q``, ``p`` are (T, 3), ``t`` (T,).

    Everything is in ``pot.units``.  The caller jits.  A weak pull toward
    the seed keeps the problem full rank, so circular orbits give a finite fit
    and gradient.  Returns the toy and whether the solver converged.
    """
    G = pot.constants["G"].ustrip(
        pot.units["length"] ** 3 / (pot.units["mass"] * pot.units["time"] ** 2)
    )
    m0, b0 = _isochrone_seed(pot, G, q, t[0])
    y0 = jax.lax.stop_gradient(jnp.log(jnp.stack([m0, b0])))

    def energy(y: Array) -> Array:
        return _isochrone_energy(jnp.exp(y[0]), jnp.exp(y[1]), G, q, p)

    lam = _TIKHONOV * jnp.mean(jnp.abs(energy(y0)))

    def residual(y: Array, _: None) -> Array:
        H = energy(y)
        return jnp.concat([H - jnp.mean(H), lam * (y - y0)])

    y, ok = _lm_fit(residual, y0)
    return IsochroneToy(m=jnp.exp(y[0]), b=jnp.exp(y[1])), ok


def _oscillator_seed(q: Array, t: Array) -> Array:
    """``2 pi / period`` per axis, from the mean spacing of mean-crossings.

    Consecutive sign changes of ``q_i - mean q_i`` are half a period apart.
    With fewer than two crossings the period is the span of ``t``.
    """
    f = q - jnp.mean(q, axis=0)
    cross = jnp.sign(f[1:]) * jnp.sign(f[:-1]) < 0  # (T-1, 3)
    dt = (t[1:] - t[:-1])[:, None]
    df = f[1:] - f[:-1]
    tc = t[:-1, None] - f[:-1] * dt / jnp.where(cross, df, 1.0)
    n = jnp.sum(cross, axis=0)
    span = jnp.max(jnp.where(cross, tc, -jnp.inf), axis=0) - jnp.min(
        jnp.where(cross, tc, jnp.inf), axis=0
    )
    half = span / jnp.where(n > 1, n - 1, 1)
    period = jnp.where(n > 1, 2 * half, t[-1] - t[0])
    return 2 * jnp.pi / period


@jax.jit
def fit_oscillator(
    q: Array, p: Array, t: Array
) -> tuple[OscillatorToy, Bool[Array, ""]]:
    """Fit the harmonic-oscillator toy to one orbit; ``q``, ``p`` are (T, 3).

    Returns the toy, with ``omega = |omega|``, and whether the solver converged.
    """
    y0 = jax.lax.stop_gradient(_oscillator_seed(q, t))

    lam = _TIKHONOV * jnp.mean(jnp.abs(_oscillator_energy(y0, q, p)))

    def residual(omega: Array, _: None) -> Array:
        H = _oscillator_energy(omega, q, p)
        return jnp.concat([H - jnp.mean(H), lam * (omega - y0)])

    omega, ok = _lm_fit(residual, y0)
    return OscillatorToy(omega=jnp.abs(omega)), ok
