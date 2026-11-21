"""Toy potentials of the O2GF method, and their fits to an orbit (pure JAX).

A toy is an analytic potential whose actions, angles and frequencies are known.
`fit_isochrone` and `fit_oscillator` choose the toy that makes the toy energy
as constant as possible along one orbit, by minimising
``sum_t (H_toy(t) - mean H_toy)^2`` with Levenberg-Marquardt. Samples with a
non-finite position, velocity or time are left out of the fit and its seed,
as they are in the linear solves. Everything is a
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

from ._utils import G_in_units, fill_masked
from .harmonic_oscillator import _ho_aaf
from .isochrone import _isochrone_aaf

_BISECT_STEPS = 20
# Weight of the pull toward the seed, relative to the mean |H| at the seed. It
# makes the least-squares problem full rank where H is flat in the parameters
# (circular orbits, an oscillator axis at rest) and is far too small to bias an
# orbit that does constrain them.
_TIKHONOV = 1e-6
_BRACKET = 1e3  # the isochrone seed searches b in [r0 / 1e3, 1e3 r0]
# The next step of a fit that has converged to round-off is at most 3e-10 of
# 1 + |y| on the test orbits, and that of a fit stopped early is above 2e-6.
_STALL_STEP = 1e-7
_MAX_STEPS = 256


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


def _kept(
    q: Array, p: Array, t: Array, ok: Array | None
) -> tuple[Array, Array, Array, Array]:
    """Return the kept-sample mask and ``q``, ``p``, ``t`` with masked rows filled.

    A masked row is replaced by the previous kept row (the next one for
    leading rows), so every value is finite and gradients stay finite too.
    """
    if ok is None:
        ok = jnp.isfinite(q).all(axis=1) & jnp.isfinite(p).all(axis=1)
        ok &= jnp.isfinite(t)
    return ok, fill_masked(q, ok), fill_masked(p, ok), fill_masked(t, ok)


def _kept_mean(x: Array, ok: Array) -> Array:
    """Mean of ``x`` over the kept samples (leading axis)."""
    w = ok.reshape(ok.shape + (1,) * (x.ndim - 1))
    return jnp.sum(jnp.where(w, x, 0.0), axis=0) / jnp.sum(ok)


def _lm_fit(residual: Any, y0: Array) -> tuple[Array, Bool[Array, ""]]:
    """Solve the least-squares fit and say whether it converged.

    The tight tolerance keeps the implicit gradient accurate, but it sits
    below the round-off floor of the loss, so near the end whether a trial
    step is accepted depends on round-off and can differ between platforms.
    The fit also counts as converged when it runs out of steps while the next
    step it would take is below ``_STALL_STEP`` relative to ``1 + |y|``.
    """
    sol = optx.least_squares(
        residual,
        optx.LevenbergMarquardt(rtol=1e-10, atol=1e-10),
        y0,
        max_steps=_MAX_STEPS,
        throw=False,
    )
    y = jax.lax.stop_gradient(sol.value)
    step = jnp.abs(jax.lax.stop_gradient(sol.state.y_eval) - y) / (1 + jnp.abs(y))
    stalled = (sol.result == optx.RESULTS.nonlinear_max_steps_reached) & jnp.all(
        step <= _STALL_STEP
    )
    return sol.value, (sol.result == optx.RESULTS.successful) | stalled


def _isochrone_seed(
    pot: Any, G: Array, q: Array, t0: Array, ok: Array
) -> tuple[Array, Array]:
    """Seed ``(m0, b0)`` from the potential at the mean radius, as gala does.

    The enclosed mass is ``r0^2 |dPhi/dr| / G`` at ``(r0, 0, 0)``; this is
    used instead of ``pot.mass_enclosed`` because it only sets the seed and
    works for every potential.  The scale
    length solves ``-G M0 / r0^3 a0 (b + a0) = Phi0`` with ``a0 = sqrt(r0^2 +
    b^2)`` by 20 bisection steps on ``log b``, falling back to ``b = r0`` when
    the bracket has no sign change.
    """
    r0 = _kept_mean(jnp.linalg.norm(q, axis=-1), ok)
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
    pot: Any, q: Array, p: Array, t: Array, ok: Array | None = None
) -> tuple[IsochroneToy, Bool[Array, ""]]:
    """Fit the isochrone toy to one orbit; ``q``, ``p`` are (T, 3), ``t`` (T,).

    Everything is in ``pot.units``.  The caller jits.  A weak pull toward
    the seed keeps the problem full rank, so circular orbits give a finite fit
    and gradient. ``ok`` marks the samples to fit, by default the finite
    ones. Returns the toy and whether the solver converged.
    """
    G = G_in_units(pot)
    keep, q, p, t = _kept(q, p, t, ok)
    m0, b0 = _isochrone_seed(pot, G, q, t[0], keep)
    y0 = jax.lax.stop_gradient(jnp.log(jnp.stack([m0, b0])))

    def energy(y: Array) -> Array:
        return _isochrone_energy(jnp.exp(y[0]), jnp.exp(y[1]), G, q, p)

    lam = _TIKHONOV * _kept_mean(jnp.abs(energy(y0)), keep)

    def residual(y: Array, _: None) -> Array:
        H = energy(y)
        return jnp.concat(
            [jnp.where(keep, H - _kept_mean(H, keep), 0.0), lam * (y - y0)]
        )

    y, ok = _lm_fit(residual, y0)
    return IsochroneToy(m=jnp.exp(y[0]), b=jnp.exp(y[1])), ok


def _oscillator_seed(q: Array, t: Array, ok: Array) -> Array:
    """``2 pi / period`` per axis, from the mean spacing of mean-crossings.

    Consecutive sign changes of ``q_i - mean q_i`` are half a period apart.
    With fewer than two crossings the period is the span of ``t``. Masked
    rows must already repeat a kept row, so they add no crossing.
    """
    f = q - _kept_mean(q, ok)
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
    q: Array, p: Array, t: Array, ok: Array | None = None
) -> tuple[OscillatorToy, Bool[Array, ""]]:
    """Fit the harmonic-oscillator toy to one orbit; ``q``, ``p`` are (T, 3).

    ``ok`` marks the samples to fit, by default the finite ones. Returns the
    toy, with ``omega = |omega|``, and whether the solver converged.
    """
    keep, q, p, t = _kept(q, p, t, ok)
    y0 = jax.lax.stop_gradient(_oscillator_seed(q, t, keep))

    lam = _TIKHONOV * _kept_mean(jnp.abs(_oscillator_energy(y0, q, p)), keep)

    def residual(omega: Array, _: None) -> Array:
        H = _oscillator_energy(omega, q, p)
        return jnp.concat(
            [jnp.where(keep, H - _kept_mean(H, keep), 0.0), lam * (omega - y0)]
        )

    omega, ok = _lm_fit(residual, y0)
    return OscillatorToy(omega=jnp.abs(omega)), ok
