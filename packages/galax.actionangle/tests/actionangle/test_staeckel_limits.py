"""Tests for the Staeckel-fudge turning points (find_limits)."""

import decimal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import galax.potential as gp
from ._helpers import (
    _GRAD_POTS,
    AXIS_W6,
    D2,
    MN_D2,
    _assert_lam_between,
    _assert_limits_close,
    _D2_pipeline,
    _F_rows,
    _kk,
    _lam_roots,
    _limits_fn,
    _live_limits,
    _mn,
    _near_circular_w6s,
    _nfw,
    _phi,
    _setup,
    _vc,
    _w6s,
    bisect,
    kk_nu_max_decimal,
    kk_turning_points,
)
from galax.actionangle._src import _staeckel_limits as limits
from galax.actionangle._src._staeckel_limits import (
    _first_step,
    _noise_snap,
)
from galax.actionangle._src._staeckel_prolate import (
    integrals_of_motion,
    make_F,
)


def test_limits_match_kk_exact():
    pot = _kk()
    w6s = _w6s(160, seed=7)
    E = jax.vmap(lambda w: integrals_of_motion(_phi(pot), w, D2).E)(w6s)
    w6s = w6s[np.asarray(E) < 0][:100]  # bound (Phi_inf = 0 for KK)
    assert len(w6s) == 100
    D2s = jnp.full(len(w6s), D2)
    lim = _limits_fn(pot)(w6s, D2s)
    ref = np.array([kk_turning_points(pot, w) for w in np.asarray(w6s)])
    assert not np.any(lim.shell | lim.planar | lim.snapped)
    # measured 3.5e-15 (lambda) and 3.7e-14 (nu): 10x margin
    np.testing.assert_allclose(lim.lam_min, ref[:, 0], rtol=4e-14, atol=0)
    np.testing.assert_allclose(lim.lam_max, ref[:, 1], rtol=4e-14, atol=0)
    nu_scale = np.maximum(ref[:, 2], 1e-4 * D2)
    np.testing.assert_array_less(np.abs(lim.nu_max - ref[:, 2]), 4e-13 * nu_scale)


def _mixed_w6s(n, vc, rmax, seed):
    """Make generic points plus planar, near-planar, edge and L_z = 0 ones."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.3, rmax, n)
    phi_ang = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-0.3, 0.3, n) * rmax
    v = rng.normal(0.0, 0.25 * vc, (n, 3))
    v[:, 1] += 0.8 * vc
    k = n // 5
    z[:k] *= 1e-7  # nu_max ~ 0: no absolute floor in the solver
    v[:k, 2] *= 1e-6
    z[k : k + 5] = 0.0  # exactly planar
    v[k : k + 5, 2] = 0.0
    z[k + 5 : k + 10] = 0.0  # z = 0, v_R = 0: lambda edge
    v[k + 5 : k + 10, 0] = 0.0
    phi_ang[k + 10 : k + 15] = 0.0  # L_z = 0
    v[k + 10 : k + 15, 1] = 0.0
    v[k + 15 : k + 35] *= 0.05  # near-radial: lam_min << lam (log-tau lane)
    c, s = np.cos(phi_ang), np.sin(phi_ang)
    vx, vy = v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c
    return jnp.asarray(np.stack([R * c, R * s, z, vx, vy, v[:, 2]], axis=1))


_REGRESSION_POTS = {
    "mn": (_mn, 0.25, 15.0),
    "isochrone": (
        lambda: gp.IsochronePotential(
            m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
        ),
        0.2,
        15.0,
    ),
    "mw2022": (gp.MilkyWayPotential2022, 0.22, 20.0),
}


@pytest.mark.parametrize("name", list(_REGRESSION_POTS))
def test_limits_convergence_regression(name, monkeypatch):
    mk, vc, rmax = _REGRESSION_POTS[name]
    pot = mk()
    w6s = _mixed_w6s(200, vc, rmax, seed=3)
    D2s = _D2_pipeline(pot, w6s)
    f16 = _limits_fn(pot)
    lim16 = f16(w6s, D2s)
    _assert_lam_between(pot, w6s, D2s, lim16)
    monkeypatch.setattr(limits, "N_ROOT_ITER", 60)
    assert _limits_fn(pot) is not f16  # the cache keys on module constants
    lim60 = _limits_fn(pot)(w6s, D2s)
    # most points bound and solved; NaN pattern unchanged
    assert np.isfinite(lim16.lam_max).mean() > 0.9
    for k in ("lam_min", "lam_max", "nu_max"):
        np.testing.assert_array_equal(
            np.isnan(getattr(lim16, k)), np.isnan(getattr(lim60, k)), err_msg=k
        )
    # the MINIMUM_RANGE collapses may differ only on lanes at their threshold
    # (roundoff); the values agree regardless (next line)
    thr = 2 * limits.MINIMUM_RANGE * np.asarray(D2s)
    for flag, width in (
        ("shell", lambda x: x.lam_max - x.lam_min),
        ("planar", lambda x: x.nu_max),
    ):
        flip = np.asarray(getattr(lim16, flag) != getattr(lim60, flag))
        w = np.maximum(np.asarray(width(lim16)), np.asarray(width(lim60)))
        assert np.all(w[flip] < thr[flip]), (flag, np.flatnonzero(flip))
    _assert_limits_close(lim16, lim60.lam_min, lim60.lam_max, lim60.nu_max, D2s)
    for lim in (lim16, lim60):  # lam_min >= D2 (log-tau lane round trip)
        ok = np.isfinite(lim.lam_min)
        np.testing.assert_array_less(
            np.asarray(D2s)[ok], np.nextafter(lim.lam_min[ok], np.inf)
        )
    assert np.any(lim16.planar)  # the planar block is exercised
    assert np.any(np.asarray(lim16.nu_max) > 0)


@pytest.mark.parametrize("semi", [False, True], ids=["lam_min", "lam_max"])
def test_quadratic_first_step_fallbacks(semi):
    sign = 1.0 if semi else -1.0
    lam = 10.0
    a, b = (0.0, 1.0) if semi else (8.0, lam)
    mid = 0.5 * (a + b)

    def step(f, df, d2f, a=a, b=b):
        args = (jnp.asarray(v) for v in (lam, f, df, d2f, a, b))
        return _first_step(*args, sign=sign, lam_pos=lam if semi else None)

    # accepted: f=1, f'=0, f''=-2 -> tau* = lam -/+ 1, inside, not clipped
    x = step(1.0, 0.0, -2.0)
    exp = np.log1p(1 / lam) / (1 + np.log1p(1 / lam)) if semi else lam - 1.0
    np.testing.assert_allclose(x, exp, rtol=1e-14)
    # accepted but within eps (b - a) of an end: clipped
    if semi:  # s* ~ 3e-6 < eps
        np.testing.assert_allclose(step(1.0, 0.0, -2e9), 1e-3, rtol=1e-14)
    else:  # tau* = lam - 1, just below b
        bb = lam - 1.0 + 1e-6
        x = step(1.0, 0.0, -2.0, b=bb)
        np.testing.assert_allclose(x, bb - 1e-3 * (bb - a), rtol=1e-14)

    # 1. disc < 0
    assert step(-1.0, 0.0, -1.0) == mid
    # 2. f'' >= 0, with disc >= 0 and the trial inside the bracket
    df = -0.1 if semi else 2.0
    assert step(1.0, df, 1e-3) == mid
    assert step(1.0, df, 0.0) == mid
    # 3. NaN f'' at lambda = D2 (on the axis inside the foci)
    _, iom, F = _setup(w6=AXIS_W6)
    assert iom.lam == D2
    d2f = F.d2F_lam(iom.lam)
    assert np.isnan(d2f)
    assert step(1.0, 0.0, d2f) == mid
    # 4. trial outside the bracket
    if semi:  # f < 0, f' < 0: tau* below lam_pos
        assert step(-1.0, -3.0, -1.0) == mid
    else:  # tau* = lam - 1 < a
        assert step(1.0, 0.0, -2.0, a=lam - 0.5) == lam - 0.25


def test_noise_snap_threshold():
    """Snap iff f'' < 0 and |f''| h^2 < SNAP_NOISE eps (S (|lam - D2| + h) + F0).

    h = (sqrt(disc) + |f'|) / |f''| is the model's far-root distance from
    lam. With S = 1, lam - D2 = 1, F0 = 0, f'' = -2 the bound is
    h^2 < 4 eps (1 + h), i.e. h < ~2.98e-8.
    """
    eps = np.finfo(float).eps
    h_max = 2 * np.sqrt(eps)  # 4 eps (1 + h) ~ 4 eps

    def snap(f, df, d2f=-2.0, F0=0.0):
        args = (f, df, d2f, 1.0, 1.0, F0)
        return bool(_noise_snap(*(jnp.asarray(v) for v in args)))

    assert snap(0.0, 0.0)
    # at f' = 0: h = sqrt(2 f |f''|) / |f''| = sqrt(f)
    assert snap((0.99 * h_max) ** 2, 0.0)
    assert not snap((1.01 * h_max) ** 2, 0.0)
    # at f = 0: h = 2 |f'| / |f''| = |f'|
    assert snap(0.0, 0.99 * h_max)
    assert not snap(0.0, 1.01 * h_max)
    # the L_z term widens it: F0 = 1 doubles the noise
    assert snap((1.2 * h_max) ** 2, 0.0, F0=1.0)
    assert not snap(0.0, 0.0, 2.0)  # convex: a minimum of F, not a shell
    assert not snap(0.0, 0.0, 0.0)
    assert not snap(0.0, 0.0, np.nan)  # f'' at D2


def test_log_potential_bound_by_default():
    pot = gp.LogarithmicPotential(
        v_c=u.Q(220.0, "km/s"), r_s=u.Q(1.0, "kpc"), units="galactic"
    )
    w6s = _w6s(30, seed=5)
    D2s = jnp.full(len(w6s), 4.0)
    lim = _limits_fn(pot)(w6s, D2s)
    for k in ("lam_min", "lam_max", "nu_max"):
        assert np.all(np.isfinite(getattr(lim, k))), k
    assert np.all(lim.lam_max > lim.lam_min)
    lim_inf = _limits_fn(pot, phi_inf=jnp.inf)(w6s, D2s)
    _assert_limits_close(lim_inf, lim.lam_min, lim.lam_max, lim.nu_max, D2s)


def test_find_limits_flags():
    """L_z = 0 pinning, the MINIMUM_RANGE collapse and the lambda edge, on KK."""
    pot = _kk()
    w6s = jnp.asarray(
        [
            [4.0, 0.0, 1.0, 0.05, 0.0, 0.08],  # L_z = 0 tube (lam_min pinned)
            [4.0, 0.0, 0.0, 0.03, 0.15, 0.0],  # exactly planar
            AXIS_W6.tolist(),  # on the axis inside the foci
            # z = 0, x v_x + y v_y = 0 (v_R = 0): the lambda edge
            [3.0, 0.0, 0.0, 0.0, 0.12, 0.05],
            [1.0, 2.0, 0.0, -0.1, 0.05, 0.03],
        ]
    )
    lim = _limits_fn(pot)(w6s, jnp.full(len(w6s), D2))
    assert lim.planar[1]
    assert lim.nu_max[1] == 0
    for k in ("lam_min", "lam_max", "nu_max"):
        assert np.all(np.isfinite(getattr(lim, k))), (k, getattr(lim, k))
    assert np.all(lim.lam_min >= D2)
    assert np.all(lim.nu_max <= D2)
    # L_z = 0: either the lam_min lane is pinned to D2, or it solved above D2
    assert (lim.lam_min[0] == D2) == lim.lmin_pinned[0]
    # the edge: exactly one lambda limit is the phase point's lambda
    lam = jax.vmap(lambda w: integrals_of_motion(_phi(pot), w, D2).lam)(w6s[3:])
    lmin, lmax = lim.lam_min[3:], lim.lam_max[3:]
    assert np.all((lmin == lam) != (lmax == lam))
    assert np.all(lmin < lmax)
    assert not np.any(lim.snapped[3:] | lim.shell[3:])


@pytest.mark.parametrize("name", ["mn", "mw2022"])
def test_nu_turning_point_falls_back_to_absnu(name):
    """At a nu turning point (nu-dot = 0), nu_max = |nu|.

    F(|nu|) = 0 there to roundoff, so the nu lane often has no bracket and
    nu_max takes the C++ fallback |nu| (not NaN, not the planar collapse).
    """
    pot = _GRAD_POTS[name]()
    phi = _phi(pot)
    rng = np.random.default_rng(4)
    n = 60
    R, z = rng.uniform(1.0, 12.0, n), rng.uniform(0.05, 2.0, n)
    vc = np.asarray(_vc(phi, R))
    w6s = jnp.asarray(np.stack([R, 0 * R, z, 0.2 * vc, 0.9 * vc, 0 * R], 1))
    D2s = jnp.full(n, MN_D2)
    iom = jax.vmap(lambda w, d: integrals_of_motion(phi, w, d))(w6s, D2s)
    # nu-dot = 0: R |nu| v_R = z dmn v_z
    w6s = w6s.at[:, 5].set(w6s[:, 0] * iom.absnu * w6s[:, 3] / (w6s[:, 2] * iom.dmn))
    iom = jax.vmap(lambda w, d: integrals_of_motion(phi, w, d))(w6s, D2s)
    lim = _limits_fn(pot)(w6s, D2s)
    assert not np.any(lim.planar)
    np.testing.assert_allclose(lim.nu_max, iom.absnu, rtol=1e-9, atol=0)


@pytest.mark.parametrize("name", ["mn", "mw2022"])
def test_lam_min_not_above_lam_pos_near_circular(name):
    """Near-circular orbits (e = 1e-7): a solved lam_min never exceeds lam_pos.

    exp(log tau) can round a root at the bracket's upper end an ulp above
    lam_pos; the clamp keeps lam_min in [lmin_a, lam_pos]. Every lane is
    finite (no NaN from a bracket lost in F's roundoff).
    """
    pot = _GRAD_POTS[name]()
    w6s = _near_circular_w6s(pot, 1e-7)
    for D2s in (jnp.full(len(w6s), MN_D2), _D2_pipeline(pot, w6s)):
        lim = _limits_fn(pot)(w6s, D2s)
        for k in ("lam_min", "lam_max", "nu_max", "lam_pos"):
            assert np.all(np.isfinite(getattr(lim, k))), k
        free = ~(lim.shell | lim.snapped | lim.lmin_pinned)
        assert free.sum() > 100
        assert np.all(lim.lam_min[free] <= lim.lam_pos[free])


@pytest.mark.parametrize("name", ["mn", "mw2022"])
@pytest.mark.parametrize("e", [0.0, 1e-8])
def test_circular_orbits_are_finite_shells(name, e):
    """(Near-)circular orbits, one batch of 310 radii: every lane a finite shell.

    The double root is fixed only to ~sqrt(eps) by F's roundoff, so the noise
    snap (model width and vertex offset < 1e-7 lam) takes it; gala's
    ``F(lam_pos) < 0`` clause left 3-14 of 310 lanes per set NaN.
    """
    pot = _GRAD_POTS[name]()
    w6s = _near_circular_w6s(pot, e)
    for D2s in (jnp.full(len(w6s), MN_D2), _D2_pipeline(pot, w6s)):
        lim = _limits_fn(pot)(w6s, D2s)
        for k in ("lam_min", "lam_max", "nu_max"):
            assert np.all(np.isfinite(getattr(lim, k))), k
        assert np.mean(lim.shell) > 0.99
        ref_lo, ref_hi = _lam_roots(pot, w6s, D2s)
        np.testing.assert_allclose(lim.lam_min, ref_lo, rtol=2e-7, atol=0)
        np.testing.assert_allclose(lim.lam_max, ref_hi, rtol=2e-7, atol=0)


def test_near_circular_pipeline_delta_roots():
    """MN, e = 1e-5, pipeline Delta: converged, unsnapped roots (<= 1e-8).

    gala's F(lam_pos) < 0 clause snapped the R = 0.65 lane, w/lam = 2.1e-5
    (2.1e-5 error): a roundoff-level F's sign is no shell test.
    """
    pot = _mn()
    w6s = _near_circular_w6s(pot, 1e-5, draw=1)
    D2s = _D2_pipeline(pot, w6s)
    lim = _limits_fn(pot)(w6s, D2s)
    assert not np.any(lim.snapped | lim.shell)
    ref_lo, ref_hi = _lam_roots(pot, w6s, D2s)
    np.testing.assert_allclose(lim.lam_min, ref_lo, rtol=1e-8, atol=0)
    np.testing.assert_allclose(lim.lam_max, ref_hi, rtol=1e-8, atol=0)


def test_split_at_the_vertex_near_the_foci():
    """Near the foci with lam at F's vertex: the lambda brackets still split.

    Near-circular KK lanes at phases near 0 and pi (lam mid-orbit, at F's
    maximum), at the worst radii of a 64-phase scan (w/lam 1e-9 to 3e-8).
    F(v) - F(lam) is ~1e-28 there, far below F's roundoff (~1e-18): the
    split's F(v) > F(lam_pos) failed by roundoff on ~0.4% of such lanes
    (KK, MN), the unsplit lam_max bracket's clipped first step was far from
    the root, and lam_max collapsed onto lam, half the extent off. Against
    KK's exact roots, every unsnapped lane is within 0.1 of its extent
    (roundoff-limited here: measured <= 2.4e-2).
    """
    pot = _kk()
    phi = _phi(pot)
    ph = np.pi * np.array([0.0, 0.03, -0.03, 1.0, 0.97, 1.03])
    w6s = jnp.asarray(
        [
            [R, 0.0, 0.0, e * vc * np.cos(p), vc * (1 + e * np.sin(p)), 0.3 * e * vc]
            for R, e in ((0.02, 1e-5), (0.0796, 1e-5), (0.2, 1e-6), (0.3, 1e-6))
            for vc in [float(_vc(phi, R))]
            for p in ph
        ]
    )
    lim = _limits_fn(pot)(w6s, jnp.full(len(w6s), D2))
    ref = np.array([kk_turning_points(pot, w, exact=True) for w in np.asarray(w6s)])
    ext = ref[:, 1] - ref[:, 0]
    np.testing.assert_array_equal(lim.shell, lim.snapped)  # shells by design only
    ok = ~np.asarray(lim.shell)
    assert ok.sum() >= 20
    for k, j in (("lam_min", 0), ("lam_max", 1)):
        err = np.abs(np.asarray(getattr(lim, k)) - ref[:, j])[ok]
        np.testing.assert_array_less(err, 0.1 * ext[ok], err_msg=k)


def test_axis_inside_foci_shells():
    """R = 0 inside the foci, moving along the axis: lambda = D2 at all times.

    F'' is NaN at D2 (0/0); the model takes it at the nudged lambda, and
    F'(D2) = 0 to roundoff there, so the lane is a shell at D2 (snapped, or
    the edge's lam with lam_min pinned to D2: L_z = 0). Its lam_max lane is
    not solved, so lam_pos is lam.
    """
    rng = np.random.default_rng(1)
    z = rng.uniform(-1.5, 1.5, 12)  # |z| < Delta = 2, and stays inside
    w6s = jnp.asarray(np.stack([0 * z, 0 * z, z, 0 * z, 0 * z, 0.01 * np.sign(z)], 1))
    lim = _limits_fn(_mn())(w6s, jnp.full(12, MN_D2))
    np.testing.assert_array_equal(lim.lam_min, MN_D2)
    np.testing.assert_array_equal(lim.lam_max, MN_D2)
    np.testing.assert_array_equal(lim.lam_pos, MN_D2)
    assert np.all(lim.shell)


def test_lam_min_not_below_D2_near_lz0():
    """Near-radial, L_z ~ 1e-9 orbits on MN (real Delta): lam_min ~ D2, never below."""
    pot = _mn()
    rng = np.random.default_rng(11)
    n = 300
    R, z = rng.uniform(0.5, 15.0, n), rng.uniform(-4.5, 4.5, n)
    v = rng.normal(0.0, 0.025, (n, 3))
    v[:, 1] = 1e-9  # tiny L_z: the root sits ulps above D2 = exp(log D2)
    w6s = jnp.asarray(np.stack([R, 0 * R, z, v[:, 0], v[:, 1], v[:, 2]], axis=1))
    D2s = _D2_pipeline(pot, w6s)
    lim = _limits_fn(pot)(w6s, D2s)
    ok = np.isfinite(lim.lam_min)
    assert ok.mean() > 0.9
    np.testing.assert_array_less(
        np.asarray(D2s)[ok], np.nextafter(lim.lam_min[ok], np.inf)
    )
    assert np.all(lim.lam_min[ok] <= lim.lam_pos[ok])


def test_near_shell_turning_point_roots(monkeypatch):
    """Near-circular orbits, incl. at a lambda turning point: converged roots.

    Near a turning point F(lam) ~ 0, so lam_pos is almost the other root and
    the far bracket end's |F| dwarfs F near the root: before the vertex split
    ~11% of these lanes were up to 6e-6 short after 16 iterations.
    """
    pot = _mn()
    phi = _phi(pot)
    rows = []  # at (and 1e-3 v_R off) a turning point, from the shells
    for R in (3.0, 4.0, 7.0):
        v = float(_vc(phi, R))
        for rel in (1e-3, 1e-4, 3e-5, 1e-5):
            for s in (1.0, -1.0):
                rows.append([R, 0.0, 0.0, 0.0, v * (1 + s * rel), 0.0])
                rows.append([R, 0.0, 0.0, 1e-3 * rel * v, v * (1 + s * rel), 0.0])
    rng = np.random.default_rng(0)  # random epicyclic phase, e = 1e-2, 1e-3
    for amp in (1e-2, 1e-3):
        R, z, ang = (
            rng.uniform(2, 12, 100),
            rng.uniform(-0.3, 0.3, 100),
            rng.uniform(0, 2 * np.pi, 100),
        )
        v = np.asarray(_vc(phi, R))
        vz = 0.05 * v * rng.normal(size=100)
        dv = amp * v
        rows += np.stack(
            [R, 0 * R, z, dv * np.cos(ang), v + dv * np.sin(ang) / 1.4, vz], 1
        ).tolist()
    w6s = jnp.asarray(rows)
    D2s = jnp.full(len(rows), MN_D2)
    lim16 = _limits_fn(pot)(w6s, D2s)
    monkeypatch.setattr(limits, "N_ROOT_ITER", 60)
    lim60 = _limits_fn(pot)(w6s, D2s)
    _assert_limits_close(lim16, lim60.lam_min, lim60.lam_max, lim60.nu_max, D2s)
    assert not np.any(lim16.snapped)  # was < 10 of 248 with gala's F < 0 clause
    _assert_lam_between(pot, w6s, D2s, lim16)

    # and the converged roots are F's roots, on every lane (bisection;
    # near-shell roots are only defined to ~eps |F| / |F'(root)|, hence 1e-9)
    assert not np.any(lim60.snapped[:48])
    lo, hi = _lam_roots(pot, w6s[:48], D2s[:48], span=0.1)
    np.testing.assert_allclose(lim60.lam_min[:48], lo, rtol=1e-9)
    np.testing.assert_allclose(lim60.lam_max[:48], hi, rtol=1e-9)


def test_lam_pos_finite_where_F_flat_and_negative():
    """F(lam) < 0 with F'(lam) = 0 (a maximum below 0, by roundoff).

    The edge's Newton step -F/F' is inf there; Limits.lam_pos is exported, so
    it must stay finite: lam_pos = lam, and both lambda limits are lam.
    """
    phi = _phi(_mn())
    w6 = jnp.asarray([5.0, 0.0, 0.0, 0.0, 0.2, 0.0])  # z = 0, v_R = 0: F(lam) = 0
    iom = integrals_of_motion(phi, w6, MN_D2)
    iom = iom._replace(E=iom.E - 1e-12)  # F(lam) = -1e-12 lam (lam - D2) < 0
    F = make_F(phi, iom)._replace(dF_lam=jnp.zeros_like)
    lim = limits.find_limits(F, iom, limits.default_phi_inf(phi, iom))
    assert F.F_lam(iom.lam) < 0
    assert lim.lam_pos == iom.lam
    assert lim.lam_min == iom.lam
    assert lim.lam_max == iom.lam
    # F(lam) > 0 with F' = 0 is no edge (an inf Newton step before): solved
    w6 = w6.at[3].set(0.02)  # v_R != 0: F(lam) > 0
    iom = integrals_of_motion(phi, w6, MN_D2)
    F = make_F(phi, iom)._replace(dF_lam=jnp.zeros_like)
    lim = limits.find_limits(F, iom, limits.default_phi_inf(phi, iom))
    assert F.F_lam(iom.lam) > 0
    assert lim.lam_min < iom.lam < lim.lam_max


# Slow lanes: (w6, D2; None = the pipeline Delta). Each converges within
# N_ROOT_ITER = 16, and at least one per potential needs more than 13.
_VC = {"kk": 0.18633180930882928, "mn": 0.21363830992234828}
_SLOW = {
    "kk": [
        # R ~ 1e-8 |z| inside the foci (lambda = D2): F'' NaN at lambda
        (
            [
                2.2455415895233283e-08,
                0.0,
                0.0224554158952258,
                -0.0011898344546856154,
                0.0024270060125430523,
                0.0006663731828401483,
            ],
            D2,
        ),
        (
            [
                1.999041651677654e-08,
                0.0,
                0.019990416516769877,
                -0.0026286978797531247,
                -0.0011428619908152267,
                0.00137959107053803,
            ],
            D2,
        ),
        # the centre with the pipeline Delta (1e-4 r): lambda_max ~ 1e24 lambda
        ([0.0, 0.0, 1e-12, *(np.array([0.3, 0.2, 0.4]) * _VC["kk"])], None),
        ([1e-12, 0.0, 0.0, *(np.array([0.3, 0.2, 0.4]) * _VC["kk"])], None),
    ],
    "mn": [
        # near the axis outside the foci, near-radial: lam_min just above D2
        (
            [
                0.002688551260061316,
                0.0,
                26.885512510994783,
                -0.0013264164495941722,
                -0.0010965626737614864,
                -0.08190822149955775,
            ],
            None,
        ),
        ([1e-12, 0.0, 0.0, *(np.array([0.3, 0.2, 0.4]) * _VC["mn"])], None),
        ([1e-12, 0.0, 0.0, *(np.array([0.3, 0.0, 0.4]) * _VC["mn"])], None),
    ],
    "nfw": [
        # a cusp at the centre, fixed Delta: F's quadratic model under-estimates
        # the lambda_max root ~1e5-fold
        (
            [
                1.295146030901508e-06,
                0.0,
                -4.3828865007367356e-07,
                -0.02814195888690673,
                0.019069903403572227,
                -0.029917216205698198,
            ],
            56.25,
        ),
        ([1e-6, 0.0, 0.0, 0.0722, 0.0481, 0.0963], 56.25),
    ],
}


@pytest.mark.parametrize("name", ["kk", "mn", "nfw"])
def test_slow_lanes_converge_in_budget(name, monkeypatch):
    """Slow lanes: converged at N_ROOT_ITER = 16, the same as at 60.

    Near the axis inside the foci (KK, also against its exact roots), at
    the centre with the pipeline Delta, near-radial lam_min just above D2,
    and a cusp. Tolerance 3e-11 of the orbit's lambda extent lam_max - D2
    (4 ulp of lambda at least): the lambda limits of these orbits are within
    ~1e-6 of D2, where 3e-11 relative to lambda says nothing. A 13-iteration
    budget fails it.
    """
    pot = _GRAD_POTS[name]()
    w6s = jnp.asarray([w for w, _ in _SLOW[name]])
    pipe = np.asarray(_D2_pipeline(pot, w6s))
    D2s = jnp.asarray(
        [pipe[i] if d is None else d for i, (_, d) in enumerate(_SLOW[name])]
    )
    lim16 = _limits_fn(pot)(w6s, D2s)
    monkeypatch.setattr(limits, "N_ROOT_ITER", 60)
    lim60 = _limits_fn(pot)(w6s, D2s)
    lam = np.asarray(lim60.lam_max)
    tol = np.maximum(3e-11 * (lam - np.asarray(D2s)), 4 * np.finfo(float).eps * lam)
    for k in ("lam_min", "lam_max"):
        assert np.all(np.isfinite(getattr(lim16, k))), k
        np.testing.assert_array_less(
            np.abs(np.asarray(getattr(lim16, k)) - getattr(lim60, k)), tol, err_msg=k
        )
    nu_tol = 3e-10 * np.maximum(np.asarray(lim60.nu_max), 1e-4 * np.asarray(D2s))
    np.testing.assert_array_less(np.abs(lim16.nu_max - lim60.nu_max), nu_tol + 1e-300)
    if name == "kk":  # inside the foci: the exact roots
        ref = np.array([kk_turning_points(pot, w) for w in np.asarray(w6s[:2])])
        for k, j in (("lam_min", 0), ("lam_max", 1)):
            np.testing.assert_array_less(
                np.abs(np.asarray(getattr(lim16, k))[:2] - ref[:, j]),
                tol[:2],
                err_msg=k,
            )


def _nu_bisect(pot, w6s, D2s):
    """nu_max by bisection of F_nu on [|nu|, D2], to the last ulp."""
    phi = _phi(pot)
    f = jax.jit(
        jax.vmap(lambda w, d, t: make_F(phi, integrals_of_motion(phi, w, d)).F_nu(t))
    )
    iom = jax.vmap(lambda w, d: integrals_of_motion(phi, w, d))(w6s, D2s)
    lo, hi = np.asarray(iom.absnu), np.asarray(D2s)
    return bisect(lambda t: np.asarray(f(w6s, D2s, jnp.asarray(t))), lo, hi, n_iter=400)


@pytest.mark.parametrize("name", ["nfw", "mw2022"])
def test_nu_max_near_a_cusp(name):
    """nu_max near a cuspy centre (fixed Delta): to 1e-8 of max(nu, 1e-4 D2).

    The nu bracket [|nu|, D2] spans decades there and F ~ a + b sqrt(tau)
    (Phi ~ r): in tau the solver only bisected (NFW: D2 / 2^15, against
    ~5e-4; MW2022 up to 10% off). It runs in log tau.
    """
    if name == "nfw":
        pot, d2 = _nfw(), 56.25
        w6s = [
            [
                3.5405919774327843e-06,
                0.0,
                -8.58212123918845e-07,
                -0.019784153453073952,
                -0.017007136829650944,
                0.018740946686217803,
            ],
            [
                2.5447273543029476e-06,
                0.0,
                6.075798585052109e-07,
                -0.04696692668393462,
                -0.13803063588216094,
                -0.02700560287124308,
            ],
            [1e-8, 0.0, 1e-8, 0.0212, 0.0212, 0.0212],
        ]
    else:
        pot, d2 = gp.MilkyWayPotential2022(), 4.0
        w6s = [
            [
                2.0256113669210302e-05,
                0.0,
                -1.5413893370878444e-05,
                0.025789695631357575,
                -0.04573692569612911,
                -0.11943352652162724,
            ],
            [
                6.881635172721498e-06,
                0.0,
                -2.589439226101709e-06,
                -0.005859224697080791,
                -0.008398595333846522,
                0.12207549902479263,
            ],
            [
                5.4529896351803646e-05,
                0.0,
                -7.060186521287817e-06,
                -0.06735933273910436,
                -0.005172915158274547,
                0.1215164746126365,
            ],
        ]
    w6s = jnp.asarray(w6s)
    D2s = jnp.full(len(w6s), d2)
    lim = _limits_fn(pot)(w6s, D2s)
    ref = _nu_bisect(pot, w6s, D2s)
    assert not np.any(lim.planar)
    np.testing.assert_array_less(
        np.abs(lim.nu_max - ref), 1e-8 * np.maximum(ref, 1e-4 * d2)
    )


def test_nu_max_near_D2_kk():
    """nu_max within 0.3% of D2 (KK, L_z = 0 and L_z != 0): KK's exact roots.

    The bracket's upper end is nu_upper (D2, or D2 (1 - D2_SIDE) at L_z =
    0), where F is tiny next to F(|nu|): inverse-quadratic steps crept
    towards it and 16 iterations left nu_max 4.2e-4 and 7.4e-5 relative
    off. The bracket is now closed from a ladder towards nu_upper. Gate
    4e-13 of max(nu_max, 1e-4 D2), as in `test_limits_match_kk_exact`
    (a 60-iteration solve: 1.3e-14 and 2e-16).
    """
    pot = _kk()
    w6s = jnp.asarray(
        [
            [
                357.433831832236,
                -56.0603401950373,
                -52.965631125202506,
                -0.014313338780077369,
                0.002244920793380785,
                0.005730489874327654,
            ],
            [
                0.5088026585917871,
                0.0,
                -0.009527732399219275,
                -0.03717906562285368,
                -0.008142090861444243,
                -0.47020274307675547,
            ],
        ]
    )
    lim = _limits_fn(pot)(w6s, jnp.full(len(w6s), D2))
    ref = np.array([kk_turning_points(pot, w, exact=True)[2] for w in np.asarray(w6s)])
    assert np.all(ref > 0.99 * D2)
    np.testing.assert_array_less(
        np.abs(lim.nu_max - ref), 4e-13 * np.maximum(ref, 1e-4 * D2)
    )


def test_nu_max_at_D2_kk_values_and_tangents():
    """nu_max = D2 (1 - 10^-k), k = 8..11 (KK, L_z = 0 and L_z ~ 2e-10).

    There F ~ c (tau - nu_max)(tau - D2) - L_z^2 tau / 2, and a ladder
    towards nu_upper ending at 10^-3 of D2 - |nu| left the root in the
    last segment, where 16 iterations crept towards nu_upper: nu_max was
    D2 - nu_max off (k >= 8), and its tangent ~(nu_upper - D2) / (nu_max -
    D2) of the right one (k = 8: 5e-6 against -49.4). The ladder now runs
    to 10^-15. nu_max within 1e-5 (D2 - nu_max) + 16 ulp of KK's exact root
    (measured <= 2.7e-6, and 6 ulp on macOS, 13 on Linux CI); its d/dv_R
    within 1e-3 (measured <= 1.3e-4) of central differences of the decimal
    root.
    """
    pot = _kk()
    w6s = jnp.asarray(
        [
            [2.0, 0.0, 1.0, 0.007200217607260861, 0.0, 0.22997202246502482],
            [2.0, 0.0, 1.0, 0.007200216087170534, 0.0, 0.22997202251261747],
            [0.5, 0.0, 3.0, -0.030380349691969857, 0.0, 0.06458363675135051],
            [5.0, 0.0, 5.0, -0.018424247937944126, 0.0, 0.09997006364742952],
            [2.0, 0.0, 1.0, 0.0072002160288155625, 2.30084711e-10, 0.2299720225144445],
            [2.0, 0.0, 1.0, 0.007200214121720883, 2.30084711e-10, 0.22997202257415392],
            [5.0, 0.0, 5.0, -0.018424248497154544, 1.01653660e-10, 0.09997006354436835],
        ]
    )
    n = len(w6s)
    lim = _limits_fn(pot)(w6s, jnp.full(n, D2))
    ref = np.array([kk_turning_points(pot, w, exact=True)[2] for w in np.asarray(w6s)])
    gap = D2 - ref
    assert np.all((gap > 5e-12 * D2) & (gap < 2e-8 * D2))
    assert not np.any(lim.nu_pinned | lim.planar)
    np.testing.assert_array_less(
        np.abs(lim.nu_max - ref), 1e-5 * gap + 16 * np.spacing(D2)
    )

    def nu_of(w, t):
        return jax.jvp(
            lambda x: _live_limits(pot, x, jnp.asarray(D2))[0].nu_max, (w,), (t,)
        )

    T = jnp.zeros((n, 6)).at[:, 3].set(1.0)
    _, J = jax.jit(jax.vmap(nu_of))(w6s, T)
    D = decimal.Decimal
    for w, Ji, g in zip(np.asarray(w6s), np.asarray(J), gap, strict=True):
        h = D(1e-10 * g)  # nu moves by ~50 h << D2 - nu_max
        wp, wm = [D(float(x)) for x in w], [D(float(x)) for x in w]
        wp[3] += h
        wm[3] -= h
        fd = float((kk_nu_max_decimal(pot, wp) - kk_nu_max_decimal(pot, wm)) / (2 * h))
        assert abs(Ji - fd) < 1e-3 * abs(fd), (Ji, fd)


def test_nu_max_just_above_tiny_absnu():
    """Log potential, fixed Delta: nu_max ~1e-15 D2, 1.6% and 1.2% above |nu|.

    With the relative planar test nu_max this small is solved, on a bracket
    [|nu|, nu_upper] of ~35 decades in log tau, and 16 iterations left these
    lanes 1.4% and 1.2% off. The bracket is also closed from |nu|
    NU_LADDER_LO: nu_max within 1e-4 of a bisection of F (measured 4.1e-6).
    """
    pot = gp.LogarithmicPotential(
        v_c=u.Q(220.0, "km/s"), r_s=u.Q(1.0, "kpc"), units="galactic"
    )
    w6s = jnp.asarray(
        [
            [5.193759239476192e-04, -2.8369768256752076e-03, -8.509305871543512e-08,
             5.960742649443681e-04, 2.6191251301744666e-04, -3.1282270457868534e-10],
            [-7.106587972695781e-05, 9.475153686437968e-05, 1.1248827861701403e-07,
             -2.940145878903825e-05, -2.20517858841831e-05, 1.4433793351485774e-12],
        ]
    )  # fmt: skip
    D2s = jnp.full(2, 4.0)
    lim = _limits_fn(pot)(w6s, D2s)
    assert not np.any(lim.planar)
    np.testing.assert_allclose(lim.nu_max, _nu_bisect(pot, w6s, D2s), rtol=1e-4)


def test_snap_independent_of_batch_size():
    """MN near-circular (e = 1e-7) and circular orbits, pipeline Delta.

    The snap decisions (snapped, shell) are the same in one batch of 310
    and one lane at a time: the noise snap's F(lam) comes from the momenta,
    not from F's roundoff, which changes with the compiled batch shape (the
    fixed 1e-7 lam width flipped 6 of these 310 lanes).
    """
    pot = _mn()
    w6s = jnp.concat([_near_circular_w6s(pot, 1e-7), _near_circular_w6s(pot, 0.0)])
    D2s = _D2_pipeline(pot, w6s)
    big = _limits_fn(pot)(w6s, D2s)
    one = [_limits_fn(pot)(w6s[i : i + 1], D2s[i : i + 1]) for i in range(len(w6s))]
    for k in ("snapped", "shell"):
        np.testing.assert_array_equal(
            np.concat([getattr(o, k) for o in one]), getattr(big, k), err_msg=k
        )
    assert 0 < np.mean(big.snapped[:310]) < 1  # the e = 1e-7 set straddles it
    assert np.all(big.snapped[310:])


def test_no_snap_deep_core_or_cusp():
    """Deep inside the foci and at a cusp, small lambda extents are resolved.

    F's roundoff scales with lam - D2 there, so the noise snap leaves these
    alone; a width relative to lam (1e-7 lam) snapped them (NFW: lam_max
    56.25 against 56.398).
    """
    pot = _nfw()
    vc = 0.24067585494452795  # at 15 kpc
    w6s = jnp.asarray(
        [
            [R, 0.0, 0.0, 0.3 * vc, vphi * vc, 0.4 * vc]
            for R in (1e-6, 1e-4, 1e-2)
            for vphi in (0.2, 0.0)
        ]
    )
    D2s = jnp.full(len(w6s), 56.25)
    lim = _limits_fn(pot)(w6s, D2s)
    assert not np.any(lim.snapped | lim.shell)
    iom = jax.vmap(lambda w, d: integrals_of_motion(_phi(pot), w, d))(w6s, D2s)
    assert np.all(lim.lam_max - iom.lam > 1e-3 * (iom.lam - 56.25 + 1e-12))
    np.testing.assert_allclose(lim.lam_max[0], 56.398, rtol=1e-4)


def test_lz_squared_underflow():
    """R = z = 1e-300: L_z^2 underflows to 0, and is treated as L_z = 0.

    F(D2) = -L_z^2 D2 / 2 is then -0, which the solver took for a converged
    root at nu_max = D2. MN, fixed Delta: nu_max matches the bisection of
    F_nu, never D2.
    """
    pot = _mn()
    vc = _VC["mn"]
    w6s = jnp.asarray([[1e-300, 0.0, 1e-300, 0.1 * vc, 0.1 * vc, 0.1 * vc]])
    D2s = jnp.full(1, 16.0)
    lim = _limits_fn(pot)(w6s, D2s)
    assert lim.nu_max[0] != 16.0
    np.testing.assert_allclose(lim.nu_max, _nu_bisect(pot, w6s, D2s), rtol=1e-8)


def _deep_core_w6s(name, n, r_frac, seed=0):
    """Random phase points at r = r_frac a, with velocities ~ v_c(r) (deep core)."""
    rng = np.random.default_rng(seed)
    r = r_frac * 3.0  # a = 3 kpc (both potentials)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    v = rng.normal(size=(n, 3)) * _VC[name] * r_frac
    return jnp.asarray(np.concat([d * r, v], axis=1))


@pytest.mark.parametrize("name", ["kk", "mn"])
def test_deep_core_lam_min_finite(name):
    """Deep cores, pipeline Delta (r = 1e-6, 1e-5 a): every limit is finite.

    At the lambda edge with F(lam) > 0 the Newton step points at the limit
    being assigned; added to lam_pos it crossed the whole (tiny) extent, past
    lam_min or below D2, and the lam_min lane lost its bracket: NaN on ~12%
    of these lanes. Like gala, the lanes then resolve or collapse to shells.
    """
    pot = _GRAD_POTS[name]()
    w6s = jnp.concat([_deep_core_w6s(name, 24, f) for f in (1e-6, 1e-5)])
    D2s = _D2_pipeline(pot, w6s)
    lim = _limits_fn(pot)(w6s, D2s)
    for k in ("lam_min", "lam_max", "nu_max"):
        assert np.all(np.isfinite(getattr(lim, k))), (
            k,
            np.flatnonzero(~np.isfinite(getattr(lim, k))),
        )
    _assert_lam_between(pot, w6s, D2s, lim)


def test_edge_limit_deep_core_kk():
    """Deep KK core (r = 10^-4.5, 1e-4 a, exact Delta): 1e-4 of the extent.

    Extents here are ~1e-9 lam, so a phase point within lam MINIMUM_RANGE
    of a turning point is at gala's edge, which sets that limit to lam: up
    to 10% of the extent off (measured) against KK's exact roots. The edge
    is relative to lam - D2 here; these lanes are solved (<~3e-6; from F's
    quadratic model at lam, with the edge relative to lam: <~1.3e-5).

    The tangents of the lanes near a turning point match central
    differences of the exact roots (step 1e-4 |v|) to 2e-2 of the extent per
    |v| (measured <~9e-3, the differences' own noise; at lam, up to 0.4).
    """
    pot = _kk()
    w6s = jnp.concat([_deep_core_w6s("kk", 24, f) for f in (10**-4.5, 1e-4)])
    lim = _limits_fn(pot)(w6s, jnp.full(len(w6s), D2))
    ref = np.array([kk_turning_points(pot, w) for w in np.asarray(w6s)])
    ext = ref[:, 1] - ref[:, 0]
    for k, j in (("lam_min", 0), ("lam_max", 1)):
        err = np.abs(np.asarray(getattr(lim, k)) - ref[:, j])
        np.testing.assert_array_less(err, 1e-4 * ext, err_msg=k)

    lam = np.asarray(jax.vmap(lambda w: integrals_of_motion(_phi(pot), w, D2).lam)(w6s))
    edge = np.min(np.abs(ref[:, :2] - lam[:, None]), axis=1) < 1e-10 * lam
    assert edge.sum() >= 4

    def lims(w):
        lim, _ = _live_limits(pot, w, jnp.asarray(D2))
        return jnp.stack([lim.lam_min, lim.lam_max])

    J = np.asarray(jax.jit(jax.vmap(jax.jacfwd(lims)))(w6s[edge]))[:, :, 3:]
    for w, Ji, e in zip(np.asarray(w6s[edge]), J, ext[edge], strict=True):
        v = np.linalg.norm(w[3:])
        h = 1e-4 * v
        for k in range(3):
            wp, wm = w.copy(), w.copy()
            wp[3 + k] += h
            wm[3 + k] -= h
            fd = (
                np.array(kk_turning_points(pot, wp)[:2])
                - np.array(kk_turning_points(pot, wm)[:2])
            ) / (2 * h)
            np.testing.assert_array_less(np.abs(Ji[:, k] - fd) * v, 2e-2 * e)


def test_deep_core_fixed_delta_not_collapsed_kk():
    """Deep KK core, exact Delta (r = 10^-5.5, 1e-5 a): no lane collapses.

    lam - D2 is 7e-12..9e-10 here, below gala's D2 MINIMUM_RANGE = 8e-10,
    and the extents are about as large: its shell test (lam_max - lam_min
    < D2 MINIMUM_RANGE) collapsed 34 of these 48 lanes onto lam, and its
    planar test (nu_max < D2 MINIMUM_RANGE) set 47 nu_max (~z^2, 2e-12..1e-9)
    to 0, O(1) errors. Both tests are relative now: lam_max - lam_min <=
    (lam - D2) MINIMUM_RANGE, and nu_max <= MINIMUM_RANGE D2 (lam - D2) /
    lam, (z/R)^2 <= MINIMUM_RANGE. Against KK's exact roots the limits are
    within 1e-3 of the extent (measured 1.4e-4, F's roundoff) and nu_max
    within 4e-3 relative (measured 3.9e-4).
    """
    pot = _kk()
    w6s = jnp.concat([_deep_core_w6s("kk", 24, f) for f in (10**-5.5, 1e-5)])
    lim = _limits_fn(pot)(w6s, jnp.full(len(w6s), D2))
    ref = np.array([kk_turning_points(pot, w, exact=True) for w in np.asarray(w6s)])
    assert not np.any(lim.shell | lim.planar | lim.snapped)
    ext = ref[:, 1] - ref[:, 0]
    for k, j in (("lam_min", 0), ("lam_max", 1)):
        err = np.abs(np.asarray(getattr(lim, k)) - ref[:, j])
        np.testing.assert_array_less(err, 1e-3 * ext, err_msg=k)
    np.testing.assert_allclose(lim.nu_max, ref[:, 2], rtol=4e-3, atol=0)


def test_edge_lam_min_near_axis_kk():
    """R ~ 1e-6..1e-5 inside the foci: lam - D2 below lam MINIMUM_RANGE ~ 8e-10.

    Every such point is at gala's lambda edge (relative to lam; it is
    relative to lam - D2 here, and these lanes are solved). lam_min - D2
    (1e-12..2e-10, set by L_z) is now right to 2 ulp of D2 against KK's
    exact roots; lam itself (gala) was up to ~70x lam_min - D2. With L_z ~
    1e-12 the root is D2 to an ulp. With L_z = 0 it is D2 exactly (F(D2) =
    0); the model's root fell an ulp or a few below D2 (3 of 120 lanes,
    with the edge relative to lam): an edge lam_min at or below D2 is
    pinned there (``lmin_pinned``), with D2's tangent. Their tangents stay
    finite.
    """
    pot = _kk()
    rng = np.random.default_rng(5)
    n = 16
    R = 10 ** rng.uniform(-6, -5, n)
    z = rng.uniform(0.3, 2.5, n)
    v = rng.normal(0, 0.3 * _VC["kk"], (n, 3))
    v[:, 1] = 10 ** rng.uniform(-7.3, -6.7, n) / R  # L_z ~ 1e-7
    v[n // 2 :, 1] = 1e-6  # L_z ~ 1e-12: lam_min is D2 to an ulp
    w6s = jnp.asarray(np.stack([R, 0 * R, z, v[:, 0], v[:, 1], v[:, 2]], 1))
    lim = _limits_fn(pot)(w6s, jnp.full(n, D2))
    ref = np.array([kk_turning_points(pot, w) for w in np.asarray(w6s)])
    np.testing.assert_array_less(
        np.abs(np.asarray(lim.lam_min) - ref[:, 0]), 2 * np.spacing(D2) + 1e-300
    )
    assert np.all(np.asarray(lim.lam_min) >= D2)

    # L_z = 0 at R = 1e-5 (every third at v_R = 0): lam_min = D2
    m = 48
    z = rng.uniform(-2.5, 2.5, m)
    vR = rng.normal(0, 0.05, m) * (np.arange(m) % 3 != 0)
    w6s = jnp.asarray(
        np.stack([np.full(m, 1e-5), 0 * z, z, vR, 0 * z, rng.normal(0, 0.1, m)], 1)
    )
    lim = _limits_fn(pot)(w6s, jnp.full(m, D2))
    lmin = np.asarray(lim.lam_min)
    assert np.all(lmin >= D2), np.flatnonzero(lmin < D2)

    def lmin_of(w, d2):
        return _live_limits(pot, w, d2)[0].lam_min

    # tangents: finite; a lam_min at D2 is pinned there (d/dD2 = 1, d/dw = 0)
    d2s = jnp.full(m, D2)
    J = np.asarray(jax.jit(jax.vmap(jax.jacfwd(lmin_of)))(w6s, d2s))
    Jd = np.asarray(jax.jit(jax.vmap(jax.grad(lmin_of, argnums=1)))(w6s, d2s))
    assert np.all(np.isfinite(J))
    at = lmin == D2
    assert at.sum() >= 3
    np.testing.assert_array_equal(J[at], 0.0)
    np.testing.assert_array_equal(Jd[at], 1.0)


def _lam_bisect(pot, w6s, D2s, n_grid=400):
    """(lam_min, lam_max) by bisection of F_lam, to the last ulp.

    Split at F's maximum on the grid D2 + (lam - D2) 10^(-8..4), which
    spans a core orbit's extent.
    """
    phi = _phi(pot)
    f = _F_rows(pot, "F_lam")
    iom = jax.vmap(lambda w, d: integrals_of_motion(phi, w, d))(w6s, D2s)
    lam, d2 = np.asarray(iom.lam), np.asarray(D2s)
    T = d2[:, None] + (lam - d2)[:, None] * np.logspace(-8, 4, n_grid)
    Fg = np.asarray(f(w6s, D2s, jnp.asarray(T)))
    rows = np.arange(len(lam))
    top = T[rows, np.argmax(np.where(np.isnan(Fg), -np.inf, Fg), axis=1)]

    def g(t):
        return np.asarray(f(w6s, D2s, jnp.asarray(t)))

    assert np.all(g(top) > 0)
    # F(D2) = -L_z^2 D2 / 2 < 0 just above D2
    return bisect(g, np.nextafter(d2, np.inf), top), bisect(g, top, T[:, -1])


def test_edge_limit_fixed_delta_cusp():
    """NFW core with a fixed Delta (r = 1e-5..1e-3, D2 = 4): 1e-5 of the extent.

    lam - D2 ~ 3e-12..2e-7 lam there, and the extent is about as large, so
    an edge test relative to lam (|dx| < lam MINIMUM_RANGE) put most of
    these lanes at the edge, and the model's root is poor on a cusp (F''
    ~ (tau - D2)^-1/2): on these lanes up to 1.8e-4 of the extent
    (lam_max) and 1.3e-4 (lam_min), and up to 4.8e-3 on a wider sweep. The
    test is relative to lam - D2 (measured now <= 2.1e-6, an ulp of lam,
    F's roundoff), against a bisection of F, to 2 ulp on the smallest
    extents (4e4 ulp; shells under gala's D2 MINIMUM_RANGE).
    """
    pot = _nfw()
    rng = np.random.default_rng(0)
    n = 48
    r = 10 ** rng.uniform(-5, -3, n)
    q, v = rng.normal(size=(2, n, 3))
    q *= (r / np.linalg.norm(q, axis=1))[:, None]
    vc = np.asarray(_vc(_phi(pot), r))
    v *= (rng.uniform(0.1, 1.2, n) * vc / np.linalg.norm(v, axis=1))[:, None]
    w6s = jnp.asarray(np.concat([q, v], 1))
    D2s = jnp.full(n, 4.0)
    lim = _limits_fn(pot)(w6s, D2s)
    lo, hi = _lam_bisect(pot, w6s, D2s)
    # no lane collapses (gala's D2 MINIMUM_RANGE shell test took 8 of them)
    assert not np.any(lim.shell | lim.snapped)
    for k, ref in (("lam_min", lo), ("lam_max", hi)):
        err = np.abs(np.asarray(getattr(lim, k)) - ref)
        np.testing.assert_array_less(
            err, 1e-5 * (hi - lo) + 2 * np.spacing(4.0), err_msg=k
        )


def test_lam_at_D2_core_lanes_nfw():
    """NFW, fixed Delta: R ~ 1e-12..3e-9 and |z| ~ 1e-6..3e-5 inside the foci.

    lam - D2 ~ R^2 rounds to 0 there (lam = D2), while lam_max - D2 ~
    R_max^2 is 6 to 2e5 ulp of D2. F''(lam) is NaN at D2, the nudge D2
    MINIMUM_RANGE overshot lam_max, and the lam_max lane had no bracket:
    NaN on 29 of these 32 lanes. It now starts from D2 (1 + D2_SIDE) and
    is closed from the sampled points (an extent within D2_SIDE of D2 is
    snapped): every lane is finite and lam_max within 4 ulp of D2 of a
    bisection of F above D2 (measured 1), lam_min D2. Tangents are finite,
    and on the lanes with extents over 1e4 ulp (FD is roundoff below that)
    the v_x tangent matches central differences to 2e-2 (measured 1e-2).
    """
    pot, d2, n = _nfw(), 4.0, 32
    phi = _phi(pot)
    rng = np.random.default_rng(2)
    r = 10 ** rng.uniform(-6, -4.5, n)
    q = rng.normal(size=(n, 3))
    q *= (r / np.linalg.norm(q, axis=1))[:, None]
    q[:, :2] *= 10 ** rng.uniform(-6, -4, n)[:, None]
    v = rng.normal(size=(n, 3))
    s = rng.uniform(0.05, 1.0, n) * np.asarray(_vc(phi, r))
    v *= (s / np.linalg.norm(v, axis=1))[:, None]
    w6s = jnp.asarray(np.concat([q, v], 1))
    D2s = jnp.full(n, d2)
    lam = jax.vmap(lambda w: integrals_of_motion(phi, w, d2).lam)(w6s)
    np.testing.assert_array_equal(lam, d2)
    lim = _limits_fn(pot)(w6s, D2s)

    # lam_max: the end of the first run of F > 0 above D2, on a log grid
    f = _F_rows(pot, "F_lam")
    T = d2 + d2 * np.logspace(-15.6, -6, 800) * np.ones((n, 1))
    pos = np.asarray(f(w6s, D2s, jnp.asarray(T))) > 0
    k = np.argmax((np.arange(800) > np.argmax(pos, 1)[:, None]) & ~pos, 1)
    rows = np.arange(n)
    ref = bisect(
        lambda t: np.asarray(f(w6s, D2s, jnp.asarray(t))), T[rows, k - 1], T[rows, k]
    )
    np.testing.assert_array_equal(lim.lam_min, d2)
    np.testing.assert_array_less(np.abs(lim.lam_max - ref), 4 * np.spacing(d2))

    def lmax(w, t):
        def lm(x):
            return _live_limits(pot, x, jnp.asarray(d2))[0].lam_max

        return jax.jvp(lm, (w,), (t,))

    t = jnp.zeros((n, 6)).at[:, 3].set(1.0)
    _, J = jax.jit(jax.vmap(lmax))(w6s, t)
    assert np.all(np.isfinite(J))

    # step 1e-2 |v_x|: the extent moves by ~2e-2 of itself, >= 200 ulp
    big = np.asarray(lim.lam_max - d2) > 1e4 * np.spacing(d2)
    assert big.sum() >= 8
    h = 1e-2 * np.abs(np.asarray(w6s[:, 3]))
    dw = jnp.asarray(h)[:, None] * t
    lp, lm = (_limits_fn(pot)(w6s + sg * dw, D2s).lam_max for sg in (1, -1))
    fd = np.asarray(lp - lm) / (2 * h)
    np.testing.assert_allclose(np.asarray(J)[big], fd[big], rtol=2e-2)


_NO_POS_W6 = [  # NFW, fixed Delta (D2 = 4), lam = D2: no F > 0 above D2
    [1.2411698862505264e-11, -1.2004018969039545e-11, -1.5254875816743483e-06,
     -9.476711721890513e-07, 1.05960991822735e-06, 1.3192020084028797e-06],
    [5.212575037157114e-13, 1.2888163397340344e-12, -1.652809500653819e-07,
     -9.688979520216365e-07, 2.5304664769262925e-06, -2.094423496719529e-06],
    [-1.5347478254630572e-12, -3.131075688277551e-12, 1.8360930007210213e-07,
     6.31296507366625e-09, 6.647528959125674e-08, -8.74009031420406e-09],
    [1.0274401048669117e-09, -7.675037878823473e-09, 3.848209674238355e-07,
     -8.684522873188362e-07, -1.5597438923415701e-06, 1.8007517444767532e-06],
    [3.293086635069785e-10, -2.708268809613961e-10, 1.5224925092123688e-06,
     -2.6809460746737086e-07, 7.646065213755888e-07, 5.512143602622589e-07],
]  # fmt: skip


def test_no_pos_lanes_snapped_nfw():
    """NFW, fixed Delta, lam = D2: no sampled point has F > 0.

    F is negative at each of the first 200 ulp above D2 (the test checks 8, D2
    (1 + D2_SIDE) among them), so the lambda extent is under an ulp. The lane
    had no F > 0 start and lam_max was NaN (105 of 1500 lanes of a fixed
    Delta core sweep). These lanes are snapped: lam_min = lam_max = D2.
    """
    pot, d2 = _nfw(), 4.0
    w6s = jnp.asarray(_NO_POS_W6)
    D2s = jnp.full(len(w6s), d2)
    lim = _limits_fn(pot)(w6s, D2s)
    T = d2 + np.spacing(d2) * np.arange(1, 9) * np.ones((len(w6s), 1))
    assert np.all(np.asarray(_F_rows(pot, "F_lam")(w6s, D2s, jnp.asarray(T))) <= 0)
    assert np.all(lim.snapped)
    np.testing.assert_array_equal(lim.lam_min, d2)
    np.testing.assert_array_equal(lim.lam_max, d2)


_EDGE_NUDGE_W6 = {  # near-circular, at a lam_min turning point (r ~ 1e-6..1e-5)
    "nfw": [
        [-5.0519113983329e-06, 5.3450111090405695e-06, 1.0330250002113305e-08,
         -0.0002587591965280235, -0.00024456095119293296, 2.416667289880595e-09],
    ],
    "hern": [
        [-1.3674897815347233e-06, 3.863469226370118e-07, 1.5809439964260176e-09,
         -0.00013293339384811372, -0.00047052296056709854, 1.6250236832292452e-11],
        [1.2027215122862353e-06, -2.7074807517278614e-06, -2.9705817235635016e-09,
         0.0006277433263121288, 0.00027885714609539417, 2.6155485391777668e-11],
        [9.174958989789957e-06, 9.498282639206984e-06, -1.3720962810951252e-08,
         -0.001052569135014696, 0.0010167394927559639, 6.080394703091064e-12],
    ],
}  # fmt: skip


@pytest.mark.parametrize("name", ["nfw", "hern"])
def test_edge_nudge_above_roundoff(name):
    """Near-circular cusp cores at a lam_min turning point, pipeline Delta.

    lam is within (lam - D2) MINIMUM_RANGE of lam_min (the edge) and F'
    there is small (~1e-18): the nudge lam MINIMUM_RANGE moved F by less
    than its roundoff, F(lam_pos) rounded to <= 0, and the lam_max lane had
    no bracket (NaN on all four lanes; then snapped, its whole extent off,
    as a lane with no F > 0 start). The nudge is now at least
    SNAP_NOISE F's roundoff over |F'|, short of half the vertex offset
    (also off the edge where F(lam) is within that of 0: its sign changed
    with the compiled program): both limits are within 1e-8 of the extent
    of a bisection of F (measured 4e-9).
    """
    if name == "nfw":
        pot = _nfw()
    else:
        pot = gp.HernquistPotential(
            m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
        )
    w6s = jnp.asarray(_EDGE_NUDGE_W6[name])
    D2s = _D2_pipeline(pot, w6s)
    lim = _limits_fn(pot)(w6s, D2s)
    lo, hi = _lam_bisect(pot, w6s, D2s)
    for k, ref in (("lam_min", lo), ("lam_max", hi)):
        err = np.abs(np.asarray(getattr(lim, k)) - ref)
        np.testing.assert_array_less(err, 1e-8 * (hi - lo), err_msg=k)


_OFF_EDGE_W6 = [  # Hernquist, near-circular, r ~ 3e-7..3e-6, pipeline Delta
    [1.0853151219378354e-06, 8.032645940553953e-07, -1.5970436640872182e-09,
     -0.00028851197624275566, 0.0003898172532888746, -5.495405236195217e-12],
    [1.2195384004059707e-06, 6.277563458500451e-08, 1.5538071844421246e-09,
     -2.4234032992708675e-05, 0.00047081453714165847, -3.084254100646906e-10],
    [3.627917048217131e-07, -4.666108706852441e-08, -8.430182789197643e-10,
     4.100957675224276e-05, 0.0003188509612439539, -3.470661811627384e-13],
    [-2.895364640955438e-07, -3.09608078499815e-07, 8.414310948206339e-10,
     0.00023775044323190065, -0.0002223378420596576, -4.075648894990571e-10],
]  # fmt: skip


def test_off_edge_nudge_hern():
    """Near-circular Hernquist cores off the lambda edge, pipeline Delta.

    F(lam) is within its roundoff of 0 but the lanes are not at the edge.
    Without the nudge of lam_pos off the edge as well, the lam_max lane had
    no F > 0 start and these lanes were snapped, lam_max off by the whole
    extent, in this batch and one lane at a time (it changed 64 of 5760
    near-circular lanes at r = 1e-8..1e-2). Both limits are within 1e-8 of
    the extent of a bisection of F (measured 2e-9).
    """
    pot = gp.HernquistPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
    )
    w6s = jnp.asarray(_OFF_EDGE_W6)
    D2s = _D2_pipeline(pot, w6s)
    lim = _limits_fn(pot)(w6s, D2s)
    assert not np.any(lim.snapped | lim.shell)
    lo, hi = _lam_bisect(pot, w6s, D2s)
    for k, ref in (("lam_min", lo), ("lam_max", hi)):
        err = np.abs(np.asarray(getattr(lim, k)) - ref)
        np.testing.assert_array_less(err, 1e-8 * (hi - lo), err_msg=k)


def test_find_limits_has_no_sort():
    """The ladder brackets use masked argmin/argmax, not argsort (perf).

    An argsort per side under vmap cost ~2-3% of the value pass; the
    sort-free form is bitwise the same (pinned by the convergence and
    regression tests).
    """
    _, iom, F = _setup(_mn())
    jaxpr = jax.make_jaxpr(lambda: limits.find_limits(F, iom, jnp.asarray(0.0)))()
    assert "sort" not in str(jaxpr)
