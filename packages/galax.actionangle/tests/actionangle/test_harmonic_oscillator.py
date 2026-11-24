"""Tests for the harmonic-oscillator action-angle transform."""

import jax
import numpy as np

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.potential as gp
from ._helpers import HO as POT, HO_OMEGA as OMEGA
from galax.actionangle import harmonic_oscillator_actions_angles_freqs as haaf
from galax.actionangle._src.harmonic_oscillator import _ho_aaf


def _gala_reference(omega, x, v):
    """Gala's ``harmonic_oscillator_xv_to_aa`` body, on (3, N) arrays.

    Adapted from gala (https://github.com/adrn/gala), MIT licence,
    ``gala/dynamics/actionangle/analyticactionangle.py``.
    """
    omega = omega.reshape(3, 1)
    action = (v**2 + (omega * x) ** 2) / (2.0 * omega)
    angle = np.arctan(-v / omega / x)
    angle[x == 0] = -np.sign(v[x == 0]) * np.pi / 2.0
    angle[x < 0] += np.pi
    return action, angle % (2.0 * np.pi)


def test_round_trip():
    rng = np.random.default_rng(1)
    J = rng.uniform(0.1, 3.0, size=(50, 3))
    th = rng.uniform(0.0, 2 * np.pi, size=(50, 3))
    x = np.sqrt(2 * J / OMEGA) * np.cos(th)
    v = -np.sqrt(2 * J * OMEGA) * np.sin(th)
    J_, th_, freqs = jax.vmap(_ho_aaf, (None, 0, 0))(jnp.asarray(OMEGA), x, v)
    np.testing.assert_allclose(J_, J, rtol=1e-12)
    np.testing.assert_allclose(th_, th, rtol=1e-12)
    np.testing.assert_allclose(freqs, np.broadcast_to(OMEGA, (50, 3)))


def test_matches_gala_formula():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 100))
    v = rng.normal(size=(3, 100))
    J_ref, th_ref = _gala_reference(OMEGA, x, v)
    J, th, _ = jax.vmap(_ho_aaf, in_axes=(None, 1, 1), out_axes=1)(
        jnp.asarray(OMEGA), x, v
    )
    np.testing.assert_allclose(J, J_ref, rtol=1e-12)
    np.testing.assert_allclose(th, th_ref, rtol=1e-12)


def test_grad_finite_at_axis():
    x = jnp.array([0.0, 0.4, 0.0])
    v = jnp.array([0.5, 0.2, -0.3])
    g = jax.jacfwd(lambda x, v: _ho_aaf(jnp.asarray(OMEGA), x, v)[1], argnums=(0, 1))(
        x, v
    )
    assert all(np.all(np.isfinite(gi)) for gi in g)


def test_units():
    res = haaf(POT, u.Q([1.0, 0.5, 0.2], "kpc"), u.Q([5.0, 3.0, 1.0], "kpc/Myr"), 0.0)
    assert str(res.actions.unit) == "kpc2 / Myr"
    assert str(res.angles.unit) == "rad"
    assert str(res.freqs.unit) == "rad / Myr"
    np.testing.assert_allclose(u.ustrip("rad/Myr", res.freqs), OMEGA)


def test_scalar_omega_broadcasts():
    pot = gp.HarmonicOscillatorPotential(omega=u.Q(1.5, "1 / Myr"), units="galactic")
    res = haaf(pot, u.Q([1.0, 0.5, 0.2], "kpc"), u.Q([5.0, 3.0, 1.0], "kpc/Myr"), 0.0)
    np.testing.assert_allclose(u.ustrip("rad/Myr", res.freqs), [1.5, 1.5, 1.5])


def test_jit_vmap_eager_agree():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(8, 3))
    p = rng.normal(size=(8, 3))
    batched = haaf(POT, q, p, 0.0)
    jitted = jax.jit(lambda q, p: haaf(POT, q, p, 0.0))(q, p)
    for i in range(8):
        one = haaf(POT, q[i], p[i], 0.0)
        for a, b, c in zip(*map(jax.tree.leaves, (one, batched, jitted)), strict=True):
            np.testing.assert_allclose(a, b[i], rtol=1e-12)
            np.testing.assert_allclose(a, c[i], rtol=1e-12)


def test_phase_space_coordinate():
    w = gc.PhaseSpaceCoordinate(
        q=u.Q([1.0, 0.5, 0.2], "kpc"),
        p=u.Q([5.0, 3.0, 1.0], "kpc/Myr"),
        t=u.Q(0.0, "Myr"),
    )
    res = haaf(POT, w)
    ref = haaf(POT, u.Q([1.0, 0.5, 0.2], "kpc"), u.Q([5.0, 3.0, 1.0], "kpc/Myr"), 0.0)
    np.testing.assert_allclose(
        u.ustrip("kpc2/Myr", res.actions), u.ustrip("kpc2/Myr", ref.actions), rtol=1e-12
    )
