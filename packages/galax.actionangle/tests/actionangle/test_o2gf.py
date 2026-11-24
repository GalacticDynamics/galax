"""Tests for the public O2GF entry point."""

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from ._helpers import HO, HO_OMEGA, ISO, KPC, MYR, TIGHT, angle_diff, integrate, lmj09
from galax.actionangle import (
    IsochroneToy,
    O2GFResult,
    OscillatorToy,
    harmonic_oscillator_actions_angles_freqs as haaf,
    isochrone_actions_angles_freqs as iaaf,
    o2gf_actions_angles_freqs as o2gf,
)
from galax.actionangle._src._o2gf_nvecs import BOX_STEPS, LOOP_STEPS, n_vectors
from galax.actionangle._src._o2gf_toy import _oscillator_seed
from galax.actionangle._src._parse import parse_w

ISO_TOY = IsochroneToy(m=jnp.asarray(1e11), b=jnp.asarray(2.0))
HO_TOY = OscillatorToy(omega=jnp.asarray(HO_OMEGA))
N_MAX = 6
# The isochrone loop orbit has T_r = 109 Myr: 10 radial periods, 2000 samples.
# Its slowest non-resonant n . Omega (|n| <= 6) needs only 590 Myr to cover 2 pi.
ISO_Q0, ISO_P0, ISO_TMAX = [6.0, 0.0, 1.0], [0.0, 0.13, 0.05], 1100.0
# The slowest oscillator axis has a period of 9 Myr.
HO_Q0, HO_P0, HO_TMAX = [1.0, 0.5, 0.3], [0.1, -0.2, 0.3], 90.0


def _iso_orbit(n=2000):
    return integrate(ISO, ISO_Q0, ISO_P0, ISO_TMAX, n)


def _ho_orbit(q0=HO_Q0, p0=HO_P0, n=1000, tmax=HO_TMAX):
    """Exact oscillator orbit, so integration error does not enter."""
    t = np.linspace(0.0, tmax, n)
    wt = HO_OMEGA * t[:, None]
    q0, p0 = np.asarray(q0), np.asarray(p0)
    q = q0 * np.cos(wt) + p0 / HO_OMEGA * np.sin(wt)
    p = -q0 * HO_OMEGA * np.sin(wt) + p0 * np.cos(wt)
    return q, p, t


def test_isochrone_fixed_toy_exact():
    q, p, t = _iso_orbit()
    res = o2gf(ISO, q, p, t, n_max=N_MAX, toy=ISO_TOY)
    assert isinstance(res, O2GFResult)
    ref = iaaf(ISO, q[0], p[0], t[0])
    J = res.aaf.actions
    np.testing.assert_allclose(J, ref.actions, rtol=1e-10)
    np.testing.assert_allclose(res.aaf.freqs, ref.freqs, rtol=1e-10)
    assert np.max(np.abs(res.sn)) < 1e-10 * np.linalg.norm(J)
    np.testing.assert_allclose(angle_diff(res.aaf.angles, ref.angles), 0, atol=1e-10)
    assert bool(res.is_loop)
    assert int(res.aligned_axis) == 2
    assert int(res.n_nan) == 0
    # Omega_phi = Omega_z in a spherical potential: exact resonances
    assert int(res.n_resonant) > 0
    assert not bool(res.flagged)


def test_isochrone_fitted_toy():
    q, p, t = _iso_orbit()
    res = o2gf(ISO, q, p, t, n_max=N_MAX)
    ref = iaaf(ISO, q[0], p[0], t[0])
    np.testing.assert_allclose(res.aaf.actions, ref.actions, rtol=1e-6)
    np.testing.assert_allclose(res.aaf.freqs, ref.freqs, rtol=1e-6)
    np.testing.assert_allclose(angle_diff(res.aaf.angles, ref.angles), 0, atol=1e-6)
    assert bool(res.fit_converged)


def test_box_orbit_oscillator():
    q, p, t = _ho_orbit()
    res = o2gf(HO, q, p, t, n_max=N_MAX, toy=HO_TOY)
    ref = haaf(HO, q[0], p[0], t[0])
    np.testing.assert_allclose(res.aaf.actions, ref.actions, rtol=1e-10)
    np.testing.assert_allclose(res.aaf.freqs, ref.freqs, rtol=1e-10)
    np.testing.assert_allclose(angle_diff(res.aaf.angles, ref.angles), 0, atol=1e-10)
    assert not bool(res.is_loop)
    assert not bool(res.toy_degenerate)
    assert not bool(res.flagged)
    np.testing.assert_array_equal(res.nvecs, n_vectors(N_MAX, BOX_STEPS))


def test_family_mismatch_raises():
    q, p, t = _ho_orbit()
    with pytest.raises(ValueError, match="family"):
        o2gf(HO, q, p, t, n_max=N_MAX, toy=HO_TOY, family="loop")
    with pytest.raises(ValueError, match="family"):
        o2gf(HO, q, p, t, n_max=N_MAX, toy=ISO_TOY, family="box")
    with pytest.raises(ValueError, match="family"):
        o2gf(HO, q, p, t, n_max=N_MAX, family="tube")


@pytest.mark.parametrize("fam", [pytest.param("loop", marks=pytest.mark.slow), "box"])
def test_classify_auto_matches_fixed(fam):
    if fam == "loop":
        (q, p, t), pot = _iso_orbit(), ISO
    else:
        (q, p, t), pot = _ho_orbit(), HO
    auto = o2gf(pot, q, p, t, n_max=N_MAX)
    fixed = o2gf(pot, q, p, t, n_max=N_MAX, family=fam)
    assert bool(auto.is_loop) == (fam == "loop")
    assert bool(fixed.is_loop) == (fam == "loop")
    for a, b in zip(jax.tree.leaves(auto.aaf), jax.tree.leaves(fixed.aaf), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    n = len(fixed.sn)
    np.testing.assert_allclose(auto.sn[:n], fixed.sn, rtol=1e-12)
    np.testing.assert_array_equal(auto.sn[n:], 0.0)
    N = len(auto.sn)
    # On the exact oscillator box dS_n/dJ is round-off of about 1e-10, which
    # changes with compilation, so allow about 10 times that.
    np.testing.assert_allclose(
        auto.dsn_dj.reshape(3, N)[:, :n],
        fixed.dsn_dj.reshape(3, n),
        rtol=1e-12,
        atol=1e-9,
    )
    loop_n, box_n = auto.nvecs
    np.testing.assert_array_equal(loop_n, n_vectors(N_MAX, LOOP_STEPS))
    np.testing.assert_array_equal(box_n, n_vectors(N_MAX, BOX_STEPS))
    assert int(auto.n_bad_cover) == int(fixed.n_bad_cover)


def test_jit_vmap_eager_agree():
    q1, p1, t1 = _iso_orbit()
    q2, p2, _ = integrate(ISO, [6.0, 1.0, 0.5], [0.02, 0.13, 0.04], ISO_TMAX)
    eager = o2gf(ISO, q1, p1, t1, n_max=N_MAX).aaf
    jitted = jax.jit(lambda q, p, t: o2gf(ISO, q, p, t, n_max=N_MAX).aaf)(q1, p1, t1)
    batch = o2gf(ISO, np.stack([q1, q2]), np.stack([p1, p2]), t1, n_max=N_MAX).aaf
    vmapped = jax.vmap(lambda q, p: o2gf(ISO, q, p, t1, n_max=N_MAX).aaf)(
        jnp.stack([q1, q2]), jnp.stack([p1, p2])
    )
    for e, j, b, v in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(jitted),
        jax.tree.leaves(batch),
        jax.tree.leaves(vmapped),
        strict=True,
    ):
        np.testing.assert_allclose(j, e, rtol=1e-12)
        np.testing.assert_allclose(b[0], e, rtol=1e-12)
        np.testing.assert_allclose(v, b, rtol=1e-12)


@pytest.mark.slow
def test_grad_fd():
    # galax orbits only support forward mode, so jacrev covers the O2GF part:
    # dJ/dw0 = dJ/d(q, p) . d(q, p)/dw0, against jacfwd of the whole chain.
    ts = u.Q(jnp.linspace(0.0, ISO_TMAX, 2000), "Myr")
    t = jnp.asarray(u.ustrip("Myr", ts))

    def orbit(q0):
        w0 = gc.PhaseSpaceCoordinate(
            q=u.Q(q0, "kpc"), p=u.Q(jnp.asarray(ISO_P0), "kpc/Myr"), t=u.Q(0.0, "Myr")
        )
        orb = gd.evaluate_orbit(ISO, w0, ts, integrator=TIGHT)
        return parse_w(ISO, orb)[:2]

    def J_qp(q, p):
        return o2gf(ISO, q, p, t, n_max=N_MAX).aaf.actions

    J = jax.jit(lambda q0: J_qp(*orbit(q0)))
    q0 = jnp.asarray(ISO_Q0)
    jf = jax.jit(jax.jacfwd(J))(q0)
    dq, dp = jax.jit(jax.jacfwd(orbit))(q0)  # (T, 3, 3) each
    gq, gp_ = jax.jit(jax.jacrev(J_qp, argnums=(0, 1)))(*orbit(q0))  # (3, T, 3)
    jr = jnp.einsum("itk,tkj->ij", gq, dq) + jnp.einsum("itk,tkj->ij", gp_, dp)
    np.testing.assert_allclose(jf, jr, rtol=1e-10, atol=1e-14)
    h = 1e-4
    e = jnp.asarray([1.0, 0, 0])
    fd = (J(q0 + h * e) - J(q0 - h * e)) / (2 * h)
    np.testing.assert_allclose(jf[:, 0], fd, rtol=1e-5)


def test_short_orbit_flags_coverage():
    q, p, t = integrate(ISO, ISO_Q0, ISO_P0, 55.0, 200)  # half a radial period
    res = o2gf(ISO, q, p, t, n_max=N_MAX, toy=ISO_TOY)
    assert int(res.n_bad_cover) > 0
    assert bool(res.flagged)


def test_short_window_flagged():
    # Over 1e-7 Myr every n . theta is nearly constant, but no toy angle turns
    # through 2 pi, so the modes are badly covered rather than resonant.
    q, p, t = _ho_orbit(n=200, tmax=1e-7)
    for kw in ({"family": "box"}, {"toy": HO_TOY}):
        res = o2gf(HO, q, p, t, n_max=N_MAX, **kw)
        assert int(res.n_resonant) == 0, kw
        assert int(res.n_bad_cover) == len(res.nvecs), kw
        assert bool(res.flagged), kw


def test_n_max_empty_lattice_raises():
    q, p, t = _ho_orbit()
    # The box lattice has only even components, so n_max = 1 leaves it empty.
    for kw in ({"family": "box"}, {"toy": HO_TOY}, {}):
        with pytest.raises(ValueError, match="n_max"):
            o2gf(HO, q, p, t, n_max=1, **kw)
    with pytest.raises(ValueError, match="n_max"):
        o2gf(ISO, q, p, t, n_max=0, family="loop")


def test_toy_param_jacobians():
    # Gradients with respect to the toy parameters, through the public API.
    q, p, t = _ho_orbit()

    def J(omega):
        return o2gf(
            HO, q, p, t, n_max=N_MAX, toy=OscillatorToy(omega=omega)
        ).aaf.actions

    w = jnp.asarray(1.1 * HO_OMEGA)
    jf = jax.jacfwd(J)(w)
    jr = jax.jacrev(J)(w)
    assert np.all(np.isfinite(jf))
    np.testing.assert_allclose(jr, jf, rtol=1e-8, atol=1e-12)
    h = 1e-6
    fd = jnp.stack(
        [(J(w + h * e) - J(w - h * e)) / (2 * h) for e in jnp.eye(3)], axis=1
    )
    np.testing.assert_allclose(jf, fd, rtol=1e-4, atol=1e-8)


@pytest.mark.slow
def test_planar_loop_orbit():
    pot = gp.MilkyWayPotential2022()
    q, p, t = integrate(pot, [8.0, 0.0, 0.0], [0.02, 0.22, 0.0], 2000.0, 2000)
    res = o2gf(pot, q, p, t, n_max=N_MAX)
    J = np.asarray(res.aaf.actions)
    assert np.all(np.isfinite(J))
    assert bool(res.is_loop)
    assert J[2] < 1e-8 * abs(J[1])
    # theta_z and Omega_z are undefined on a planar orbit
    assert np.isnan(res.aaf.angles[2])
    assert np.isnan(res.aaf.freqs[2])
    assert np.all(np.isfinite(res.aaf.angles[:2]))
    assert np.all(np.isfinite(res.aaf.freqs[:2]))
    g = jax.grad(lambda q_: o2gf(pot, q_, p, t, n_max=N_MAX).aaf.actions.sum())(
        jnp.asarray(q)
    )
    assert np.all(np.isfinite(g))


def test_nan_toy_samples_masked():
    # Two samples get a velocity that is unbound in the toy.
    q, p, t = _iso_orbit()
    p = p.copy()
    p[[500, 1200]] *= 10.0
    res = o2gf(ISO, q, p, t, n_max=N_MAX, toy=ISO_TOY)
    assert int(res.n_nan) == 2
    assert bool(res.flagged)
    for x in (*jax.tree.leaves(res.aaf), res.sn, res.dsn_dj):
        assert np.all(np.isfinite(x))

    g = jax.grad(
        lambda q_: o2gf(ISO, q_, p, t, n_max=N_MAX, toy=ISO_TOY).aaf.actions.sum()
    )(jnp.asarray(q))
    assert np.all(np.isfinite(g))


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"family": "box"}, marks=pytest.mark.slow),
        {"toy": HO_TOY},
        pytest.param({}, marks=pytest.mark.slow),
    ],
)
def test_planar_box_grad_finite(kw):
    q, p, t = _ho_orbit([1.0, 0.5, 0.0], [0.1, -0.2, 0.0])

    def f(q_, p_, kw=kw):
        return o2gf(HO, q_, p_, t, n_max=N_MAX, **kw).aaf.actions.sum()

    gq, gp_ = jax.grad(f, argnums=(0, 1))(jnp.asarray(q), jnp.asarray(p))
    assert np.all(np.isfinite(gq)), kw
    assert np.all(np.isfinite(gp_)), kw
    res = o2gf(HO, q, p, t, n_max=N_MAX, **kw).aaf
    assert float(res.actions[2]) == 0.0, kw
    assert np.isnan(res.angles[2]), kw
    assert np.isnan(res.freqs[2]), kw
    assert np.all(np.isfinite(res.freqs[:2])), kw


def test_batched_toys_match_single():
    orbits = [
        _iso_orbit(),
        integrate(ISO, [6.0, 1.0, 0.5], [0.02, 0.13, 0.04], ISO_TMAX),
        integrate(ISO, [6.0, -1.0, 1.5], [0.03, 0.12, 0.06], ISO_TMAX),
    ]
    q = np.stack([o[0] for o in orbits])
    p = np.stack([o[1] for o in orbits])
    t = orbits[0][2]
    m = jnp.asarray([1e11, 1.1e11, 0.9e11])
    b = jnp.asarray([2.0, 2.2, 1.8])
    res = o2gf(ISO, q, p, t, n_max=N_MAX, toy=IsochroneToy(m=m, b=b))
    # On a flagged orbit the solve is ill-conditioned and batching changes
    # the round-off, so the orbits here are all well covered.
    assert not np.any(res.flagged)
    for i in range(3):
        one = o2gf(ISO, q[i], p[i], t, n_max=N_MAX, toy=IsochroneToy(m=m[i], b=b[i]))
        for a, s in zip(
            jax.tree.leaves(res.aaf), jax.tree.leaves(one.aaf), strict=True
        ):
            # Batching sums in a different order, so a small component is
            # judged against the largest one in its array.
            np.testing.assert_allclose(
                a[i], s, rtol=1e-12, atol=1e-12 * np.max(np.abs(s))
            )
        np.testing.assert_allclose(res.sn[i], one.sn, rtol=1e-10, atol=1e-14)


def _log_box():
    """Integrate a box orbit from rest in a triaxial logarithmic potential."""
    pot = lmj09(0.9, 0.8)
    return pot, integrate(pot, [4.0, 3.0, 2.0], [0.0, 0.0, 0.0], 12000.0)


@pytest.mark.slow
def test_box_forced_to_loop_flagged():
    # On the loop path this orbit gives a J about half the box J, so the
    # result must be flagged however the loop path is chosen.
    pot, (q, p, t) = _log_box()
    auto = o2gf(pot, q, p, t, n_max=N_MAX)
    assert not bool(auto.is_loop)
    assert not bool(auto.family_mismatch)
    loop = o2gf(pot, q, p, t, n_max=N_MAX, family="loop")
    assert bool(loop.family_mismatch)
    assert bool(loop.flagged)
    fixed = o2gf(pot, q, p, t, n_max=N_MAX, toy=loop.toy)
    assert bool(fixed.family_mismatch)
    assert bool(fixed.flagged)


def test_loop_forced_to_box_mismatch():
    q, p, t = _iso_orbit()
    res = o2gf(ISO, q, p, t, n_max=N_MAX, family="box")
    assert bool(res.family_mismatch)
    assert bool(res.flagged)
    for kw in ({"family": "loop"}, {"toy": ISO_TOY}, {}):
        res = o2gf(ISO, q, p, t, n_max=N_MAX, **kw)
        assert not bool(res.family_mismatch), kw
        assert not bool(res.flagged), kw
    qh, ph, th = _ho_orbit()
    for kw in ({"family": "box"}, {"toy": HO_TOY}):
        res = o2gf(HO, qh, ph, th, n_max=N_MAX, **kw)
        assert not bool(res.family_mismatch), kw
        assert not bool(res.flagged), kw


def _tnfw():
    return gp.TriaxialNFWPotential(
        m=u.Q(5.4e11, "Msun"),
        r_s=u.Q(15.0, "kpc"),
        q1=u.Q(0.9, ""),
        q2=u.Q(0.75, ""),
        units="galactic",
    )


def test_collapsed_box_toy_flagged():
    # On this thin box orbit the best variance fit wants omega_z^2 < 0, so the
    # oscillator fit stops at omega_z ~ 0 with the solver converged and the
    # coverage passing. J_z came back as -2e6 kpc^2/Myr, unflagged, while
    # <E_z> / Omega_z is about 2e-3.
    pot = _tnfw()
    kms = u.Q(1.0, "km/s").ustrip("kpc/Myr")
    q, p, t = integrate(
        pot, [10.0, 6.0, 1.0], [5 * kms, 3 * kms, 0.5 * kms], 15000.0, n=4000
    )
    res = o2gf(pot, q, p, t, n_max=8, family="box")
    assert bool(res.fit_converged)
    assert int(res.n_bad_cover) == 0
    assert float(res.toy.omega[2]) < 1e-6
    assert bool(res.toy_degenerate)
    assert bool(res.flagged)
    auto = o2gf(pot, q, p, t, n_max=8)
    assert not bool(auto.is_loop)
    assert bool(auto.toy_degenerate)
    assert bool(auto.flagged)


def test_thin_box_flagged():
    # Near-planar boxes, with z0 = 1e-5 and 1e-4 kpc, and a 3-D box from the
    # same in-plane start. On the thin ones the n_z != 0 modes absorb the
    # in-plane fit residual, and the fitted toy collapses or not depending on
    # round-off, so they must be flagged whatever the round-off. The 3-D box
    # has a smallest toy action fraction of 4e-2 and must stay unflagged.
    pot = lmj09(0.85, 0.7)
    zs = np.array([1e-5, 1e-4, 1.0])
    q0 = np.tile([4.78033055, 0.120120631, 0.0], (3, 1))
    q0[:, 2] = zs
    v0 = np.tile([0.06884186, -0.09733735, 0.0], (3, 1))
    want = np.array([True, True, False])
    for n in (3999, 4000, 4001):
        q, p, t = integrate(pot, q0, v0, 12000.0, n)
        res = o2gf(pot, q, p, t, n_max=6)
        assert not np.any(np.asarray(res.is_loop))
        np.testing.assert_array_equal(np.asarray(res.flagged), want, err_msg=str(n))
        thin = np.asarray(res.thin_box) | np.asarray(res.toy_degenerate)
        np.testing.assert_array_equal(thin, want, err_msg=str(n))
    # 16 draws of 1e-14 relative noise on the 4001-sample orbits.
    rng = np.random.default_rng(0)
    qn = q * (1 + 1e-14 * rng.standard_normal((16, *q.shape)))
    pn = p * (1 + 1e-14 * rng.standard_normal((16, *p.shape)))
    res = o2gf(pot, qn, pn, np.broadcast_to(t, (16, *t.shape)), n_max=6)
    np.testing.assert_array_equal(
        np.asarray(res.flagged), np.broadcast_to(want, (16, 3))
    )
    # The thin-box check is on the box path only, so a near-planar loop is
    # not flagged by it.
    q, p, t = integrate(ISO, [6.0, 0.0, 1e-5], [0.0, 0.13, 0.0], ISO_TMAX)
    for kw in ({}, {"family": "loop"}):
        res = o2gf(ISO, q, p, t, n_max=N_MAX, **kw)
        assert not bool(res.thin_box), kw
        assert not bool(res.flagged), kw


def test_thin_box_audit_reproducers_flagged():
    # The audit's near-planar boxes, which came out with J_z at 2e-3 to 1e-2
    # of the sum of |J| and in-plane actions 4 to 9 percent off, unflagged.
    for pot, q0, v0, tmax in (
        (_tnfw(), [-5.27204, -3.44235, 1.11e-5], [0.0089927, -0.0156182, 0.0], 20000.0),
        (
            lmj09(0.85, 0.7),
            [1.63905, 7.60687, 1.047e-4],
            [0.0245734, -0.0044446, 0.0],
            12000.0,
        ),
    ):
        q, p, t = integrate(pot, np.array([q0]), np.array([v0]), tmax, 5000)
        for n_max in (6, 8):
            res = o2gf(pot, q, p, t, n_max=n_max)
            assert not bool(res.is_loop[0])
            assert bool(res.thin_box[0] | res.toy_degenerate[0]), (pot, n_max)
            assert bool(res.flagged[0]), (pot, n_max)


@pytest.mark.slow
def test_box_scan_collapsed_flagged():
    # The audit's random boxes released near rest in a triaxial NFW: about
    # half collapse an axis. Truth is <E_i> / omega_i at the seed frequency,
    # and a collapsed orbit has some |J_i| more than 30 times its largest.
    pot = _tnfw()
    rng = np.random.default_rng(3)
    n, rmax = 16, 12.0
    q0 = rng.uniform(-rmax, rmax, (n, 3)) * rng.choice([0.03, 0.1, 0.3, 1.0], (n, 3))
    q0[:, 0] = rng.uniform(0.5, 1.0, n) * rmax
    v0 = rng.normal(0, 3, (n, 3)) * u.Q(1.0, "km/s").ustrip("kpc/Myr")
    q, p, t = integrate(pot, q0, v0, 15000.0, 4000)
    res = o2gf(pot, q, p, t, n_max=6, family="box")
    J = np.asarray(res.aaf.actions)
    om = np.stack(
        [
            np.asarray(_oscillator_seed(*x, jnp.ones(len(x[1]), bool)))
            for x in zip(q, t, strict=True)
        ]
    )
    est = np.mean(0.5 * p**2 + 0.5 * (om[:, None] * q) ** 2, axis=1) / om
    collapsed = np.any(np.abs(J) > 30 * est.max(axis=1, keepdims=True), axis=1)
    degenerate = np.asarray(res.toy_degenerate)
    assert collapsed.sum() >= 5
    assert (~collapsed).sum() >= 5
    np.testing.assert_array_equal(degenerate, collapsed)
    assert np.all(np.asarray(res.flagged)[collapsed])
    assert not np.any(np.asarray(res.flagged)[~collapsed])


def test_nan_input_row_fitted_toy():
    # One NaN sample is masked in classification and the toy fit as well as
    # in the solves, so it matches dropping that sample.
    q, p, t = _iso_orbit()
    qn = q.copy()
    qn[500] = np.nan
    res = o2gf(ISO, qn, p, t, n_max=N_MAX)
    ref = o2gf(
        ISO,
        np.delete(q, 500, axis=0),
        np.delete(p, 500, axis=0),
        np.delete(t, 500),
        n_max=N_MAX,
    )
    assert int(res.n_nan) == 1
    assert bool(res.is_loop)
    assert bool(res.fit_converged)
    assert not bool(res.flagged)
    np.testing.assert_allclose(res.aaf.actions, ref.aaf.actions, rtol=1e-8)
    g = jax.grad(lambda q_: o2gf(ISO, q_, p, t, n_max=N_MAX).aaf.actions.sum())(
        jnp.asarray(qn)
    )
    assert np.all(np.isfinite(g))


def test_bad_inputs_raise():
    q, p, t = _iso_orbit()
    toy = IsochroneToy(m=u.Q(1e11, "Msun"), b=u.Q(2.0, "kpc"))
    with pytest.raises(TypeError, match="bare arrays"):
        o2gf(ISO, q, p, t, n_max=N_MAX, toy=toy)
    with pytest.raises(TypeError, match="IsochroneToy or OscillatorToy"):
        o2gf(ISO, q, p, t, n_max=N_MAX, toy=(1e11, 2.0))
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(ISO_Q0, "kpc"), p=u.Q(ISO_P0, "kpc/Myr"), t=u.Q(0.0, "Myr")
    )
    with pytest.raises(ValueError, match="time axis"):
        o2gf(ISO, w0, n_max=N_MAX)
    with pytest.raises(ValueError, match="time axis"):
        o2gf(ISO, q[:1], p[:1], t[:1], n_max=N_MAX)


def test_float32():
    q, p, t = _iso_orbit()
    ref = o2gf(ISO, q, p, t, n_max=N_MAX).aaf
    res = o2gf(
        ISO,
        q.astype(np.float32),
        p.astype(np.float32),
        t.astype(np.float32),
        n_max=N_MAX,
    ).aaf
    for a in jax.tree.leaves(res):
        assert a.dtype == np.float32
    np.testing.assert_allclose(res.actions, ref.actions, rtol=1e-4)
    np.testing.assert_allclose(res.freqs, ref.freqs, rtol=1e-4)
    np.testing.assert_allclose(angle_diff(res.angles, ref.angles), 0, atol=1e-4)


def test_units():
    q, p, t = _iso_orbit()
    plain = o2gf(ISO, q, p, t, n_max=N_MAX)
    assert not isinstance(plain.aaf.actions, u.AbstractQuantity)
    withq = o2gf(ISO, u.Q(q, "kpc"), u.Q(p, "kpc/Myr"), u.Q(t, "Myr"), n_max=N_MAX).aaf
    assert withq.actions.unit == u.unit("kpc2 / Myr")
    assert withq.angles.unit == u.unit("rad")
    assert withq.freqs.unit == u.unit("rad / Myr")
    np.testing.assert_allclose(
        u.ustrip("kpc2/Myr", withq.actions), plain.aaf.actions, rtol=1e-12
    )

    orb = integrate(ISO, ISO_Q0, ISO_P0, ISO_TMAX, orbit=True)
    fromw = o2gf(ISO, orb, n_max=N_MAX).aaf
    assert fromw.actions.unit == u.unit("kpc2 / Myr")
    np.testing.assert_allclose(
        u.ustrip("kpc2/Myr", fromw.actions), plain.aaf.actions, rtol=1e-10
    )


def _mn(units):
    return gp.MiyamotoNagaiPotential(
        m_tot=u.Q(6.8e10, "Msun"), a=u.Q(3.0, "kpc"), b=u.Q(0.28, "kpc"), units=units
    )


def _nfw(units):
    return gp.NFWPotential(m=u.Q(5.4e11, "Msun"), r_s=u.Q(15.0, "kpc"), units=units)


@pytest.mark.parametrize(
    ("mk", "q0", "v0", "tmax"),
    [
        # near Omega_r : Omega_z = 2 : 3, and its fit does not converge
        (_mn, [4.0, 0.0, 1.5], [30.0, 150.0, 60.0], 4000.0),
        (_nfw, [15.0, 0.0, 5.0], [30.0, 120.0, 40.0], 10000.0),
    ],
    ids=["mn", "nfw"],
)
def test_si_matches_galactic(mk, q0, v0, tmax):
    # The toy fit divides the energies by their mean at the seed. Without
    # that, in SI the damping could not shorten the first step and the fit
    # stopped at its seed.
    p0 = u.Q(v0, "km/s").ustrip("kpc/Myr")
    q, p, t = integrate(mk("galactic"), q0, p0, tmax)
    gal = o2gf(mk("galactic"), q, p, t, n_max=N_MAX)
    si = o2gf(mk("si"), q * KPC, p * KPC / MYR, t * MYR, n_max=N_MAX)
    assert bool(si.flagged) == bool(gal.flagged)
    assert bool(si.fit_converged) == bool(gal.fit_converged)
    np.testing.assert_allclose(
        np.asarray(si.aaf.actions) / (KPC**2 / MYR), gal.aaf.actions, rtol=1e-8
    )


def test_sn_units_follow_actions():
    q, p, t = _iso_orbit()
    plain = o2gf(ISO, q, p, t, n_max=N_MAX, toy=ISO_TOY)
    assert not isinstance(plain.sn, u.AbstractQuantity)
    assert not isinstance(plain.aaf.actions, u.AbstractQuantity)
    assert not isinstance(plain.dsn_dj, u.AbstractQuantity)

    orb = integrate(ISO, ISO_Q0, ISO_P0, ISO_TMAX, orbit=True)
    res = o2gf(ISO, orb, n_max=N_MAX, toy=ISO_TOY)
    assert isinstance(res.sn, u.AbstractQuantity)
    assert res.sn.unit == res.aaf.actions.unit
    assert not isinstance(res.dsn_dj, u.AbstractQuantity)


def test_gap_or_shuffle_in_t_flags():
    """A step too long to unwrap the toy angles flags the orbit."""
    q, p, t = _iso_orbit()
    full = o2gf(ISO, q, p, t, n_max=6)
    short = (t < 500) | (t > 520)  # the toy angles step under pi / 2
    r = o2gf(ISO, q[short], p[short], t[short], n_max=6)
    assert not bool(r.flagged)
    np.testing.assert_allclose(r.aaf.freqs, full.aaf.freqs, rtol=1e-6)
    long = (t < 500) | (t > 600)  # theta_r steps by about 5.8 rad
    r = o2gf(ISO, q[long], p[long], t[long], n_max=6)
    assert bool(r.flagged)
    assert int(r.n_bad_cover) > 0
    i = np.random.default_rng(0).permutation(len(t))
    r = o2gf(ISO, q[i], p[i], t[i], n_max=6)
    assert bool(r.flagged)
    i = np.arange(len(t))[::-1]
    r = o2gf(ISO, q[i], p[i], t[i], n_max=6)
    assert not bool(r.flagged)


def test_gap_flags_with_fitted_box_toy():
    """A fitted toy's angles outrun its frequencies; the gap check has margin."""
    pot, (q, p, t) = _log_box()
    keep = np.ones(len(t), bool)
    keep[1884:1891] = False  # predicted step 0.77 pi, actual step 1.06 pi
    r = o2gf(pot, q[keep], p[keep], t[keep], n_max=6)
    assert bool(r.flagged)
    assert int(r.n_bad_cover) > 0


def test_gap_flags_on_disk_loop():
    """On a disk loop the toy's Omega_z is ~3x too slow; the angle rate catches it."""
    pot = gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), b=u.Q(0.28, "kpc"), units="galactic"
    )
    kms = u.Q(1.0, "km/s").ustrip("kpc/Myr")
    q, p, t = integrate(pot, [8.0, 0, 0.3], [10 * kms, 230 * kms, 10 * kms], 6000, 6000)
    assert not bool(o2gf(pot, q, p, t, n_max=6).flagged)
    keep = np.ones(len(t), bool)
    keep[3000:3056] = False  # toy-predicted step 0.42 pi, actual theta_z step 1.0 pi
    r = o2gf(pot, q[keep], p[keep], t[keep], n_max=6)
    assert bool(r.flagged)
    assert int(r.n_bad_cover) > 0


def test_t_without_time_axis_raises_and_equal_times_flag():
    """A scalar t raises; equal sample times leave Omega undetermined, flagged."""
    q, p, t = _iso_orbit()
    for bad in (0.0, np.array([0.0])):
        with pytest.raises(ValueError, match="time axis"):
            o2gf(ISO, q, p, bad, n_max=6)
    r = o2gf(ISO, q, p, np.full_like(t, 5.0), n_max=6)
    assert bool(r.flagged)
    # A single phase-space point still raises the documented ValueError.
    for t1 in (0.0, t[:1]):
        with pytest.raises(ValueError, match="time axis of at least 2 samples"):
            o2gf(ISO, q[0], p[0], t1, n_max=6)
