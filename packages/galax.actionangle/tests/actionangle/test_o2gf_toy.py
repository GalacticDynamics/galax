"""Tests for the O2GF toy potentials and their fits."""

import pathlib

import jax
import numpy as np

import quaxed.numpy as jnp

import galax.actionangle._src._o2gf_toy as _toy
from ._helpers import HO, HO_OMEGA, ISO, KPC, MYR, integrate
from galax.actionangle import isochrone_actions_angles_freqs as iaaf
from galax.actionangle._src._o2gf_toy import (
    IsochroneToy,
    OscillatorToy,
    fit_isochrone,
    fit_oscillator,
    toy_aaf,
)
from galax.actionangle._src._utils import finite_rows

RTOL_NEAR_CIRC = 1e-3  # measured 1.1e-4, with margin
G = ISO.constants["G"].ustrip("kpc3 / (Msun Myr2)")


def _iso_orbit(x0=8.0):
    # loop orbit, e ~ 0.3
    return integrate(ISO, [x0, 0.0, 1.0], [0.0, 0.12, 0.05], 1500.0, 300)


def test_fit_isochrone_recovers():
    q, p, t = _iso_orbit()
    toy, ok = fit_isochrone(ISO, q, p, t, finite_rows(q, p, t))
    assert bool(ok)
    np.testing.assert_allclose(toy.m, 1e11, rtol=1e-6)
    np.testing.assert_allclose(toy.b, 2.0, rtol=1e-6)


def test_fit_oscillator_recovers():
    q, p, t = integrate(HO, [1.0, 0.5, 0.3], [0.1, -0.2, 0.3], 30.0, 200)
    toy, ok = fit_oscillator(q, p, t, finite_rows(q, p, t))
    assert bool(ok)
    np.testing.assert_allclose(toy.omega, HO_OMEGA, rtol=1e-8)


def test_toy_aaf_matches_public():
    q, p, t = _iso_orbit()
    toy = IsochroneToy(m=jnp.asarray(1e11), b=jnp.asarray(2.0))
    J, th, om = toy_aaf(toy, G, q, p)
    ref = iaaf(ISO, q, p, t)
    np.testing.assert_allclose(J, ref.actions, rtol=1e-12)
    np.testing.assert_allclose(th, ref.angles, rtol=1e-12)
    np.testing.assert_allclose(om, ref.freqs, rtol=1e-12)


def test_toy_aaf_oscillator():
    q, p, _ = integrate(HO, [1.0, 0.5, 0.3], [0.1, -0.2, 0.3], 30.0, 20)
    J, _, om = toy_aaf(OscillatorToy(omega=jnp.asarray(HO_OMEGA)), G, q, p)
    np.testing.assert_allclose(om, np.broadcast_to(HO_OMEGA, (20, 3)))
    np.testing.assert_allclose(J, J[0] + 0 * J, rtol=1e-7)


def _grad_vs_fd(q, p, t, h=1e-4):
    q, p, t = jnp.asarray(q), jnp.asarray(p), jnp.asarray(t)

    @jax.jit
    def fit(x0):
        toy, ok = fit_isochrone(ISO, q.at[0, 0].set(x0), p, t, finite_rows(q, p, t))
        return toy.b, ok

    x0 = q[0, 0]
    g = jax.grad(lambda x: fit(x)[0])(x0)
    fd = (fit(x0 + h)[0] - fit(x0 - h)[0]) / (2 * h)
    return g, fd, fit(x0)[1]


def test_fit_grad_fd():
    g, fd, _ = _grad_vs_fd(*_iso_orbit())
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_circular_orbit_grad_finite():
    # H is constant for every (m, b) on an exactly circular orbit
    r = 8.0
    a = np.sqrt(4.0 + r**2)
    vc = float(np.sqrt(G * 1e11 * r**2 / (a * (2.0 + a) ** 2)))
    t = np.linspace(0.0, 1500.0, 300)
    w = vc / r
    q = np.stack([r * np.cos(w * t), r * np.sin(w * t), 0 * t], axis=1)
    p = np.stack([-vc * np.sin(w * t), vc * np.cos(w * t), 0 * t], axis=1)
    g, _, ok = _grad_vs_fd(q, p, t)
    assert bool(ok)
    assert np.isfinite(g)


def test_near_circular_grad_bounded():
    # e ~ 0.02 (r in 7.6-8.0 kpc): gradient vs FD differs by 1.1e-4 (measured)
    q, p, t = integrate(ISO, [8.0, 0.0, 0.0], [0.0, 0.1789, 0.0], 1500.0, 300)
    # the valley is nearly flat, so the FD step must be well above the solver noise
    g, fd, ok = _grad_vs_fd(q, p, t, h=1e-3)
    assert bool(ok)
    assert np.isfinite(g)
    np.testing.assert_allclose(g, fd, rtol=RTOL_NEAR_CIRC)


def test_fit_not_converged_flag():
    # The fit leaves out a NaN sample, but with every sample NaN each residual
    # is NaN, so the solver cannot succeed.
    rng = np.random.default_rng(0)
    q, p, t = rng.normal(size=(50, 3)), rng.normal(size=(50, 3)), np.arange(50.0)
    _, ok = fit_oscillator(q, p, t, finite_rows(q, p, t))
    assert bool(ok)
    q[:, 0] = np.nan
    _, ok = fit_oscillator(q, p, t, finite_rows(q, p, t))
    assert not bool(ok)


def test_fit_out_of_steps(monkeypatch):
    # A fit that runs out of steps counts as converged only when its next step
    # is round-off. With 10 steps the mw2022 loop's oscillator fit is there
    # (next step 2e-12 of 1 + |omega|) and the random-normal fit is not (1e-2).
    monkeypatch.setattr(_toy, "_MAX_STEPS", 10)

    # A new function, so jit traces it again instead of reusing a cached trace.
    @jax.jit
    def fit(q, p, t):
        return fit_oscillator.__wrapped__(q, p, t, finite_rows(q, p, t))

    ref = np.load(pathlib.Path(__file__).parent / "data" / "o2gf_gala_refs.npz")
    q, p, t = (ref[f"mw2022_loop__{k}"] for k in ("q", "p", "t"))
    _, ok = fit(q, p, t)
    assert bool(ok)
    rng = np.random.default_rng(0)
    q, p, t = rng.normal(size=(50, 3)), rng.normal(size=(50, 3)), np.arange(50.0)
    _, ok = fit(q, p, t)
    assert not bool(ok)
    # In SI omega is about 1e-15 s^-1. The fit is in omega over its seed, so the verdict
    # does not depend on the units.
    _, ok = fit(q * KPC, p * KPC / MYR, t * MYR)
    assert not bool(ok)


def test_oscillator_axis_at_rest_grad_finite():
    # H does not depend on omega_z when the orbit never leaves z = 0, v_z = 0
    q, p, t = integrate(HO, [1.0, 0.5, 0.0], [0.1, -0.2, 0.0], 30.0, 200)
    q, p, t = jnp.asarray(q), jnp.asarray(p), jnp.asarray(t)
    g = jax.jit(
        jax.grad(
            lambda x: fit_oscillator(q.at[0, 0].set(x), p, t, finite_rows(q, p, t))[
                0
            ].omega[0]
        )
    )(1.0)
    assert np.isfinite(g)
