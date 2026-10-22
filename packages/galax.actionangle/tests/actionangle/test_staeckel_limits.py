"""Tests for the Staeckel-fudge turning points (find_limits)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import galax.potential as gp
from ._helpers import (
    AXIS_W6,
    D2,
    MN_D2,
    _assert_lam_between,
    _assert_limits_close,
    _D2_pipeline,
    _kk,
    _lam_roots,
    _limits_fn,
    _mn,
    _near_circular_w6s,
    _phi,
    _setup,
    _w6s,
    bisect,
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
    lim16 = _limits_fn(pot)(w6s, D2s)
    _assert_lam_between(pot, w6s, D2s, lim16)
    monkeypatch.setattr(limits, "N_ROOT_ITER", 60)
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


@pytest.mark.parametrize("pot", [_kk(), _mn()], ids=["kk", "mn"])
def test_edge_branch_vR0_z0(pot):
    w6s = jnp.asarray(
        [[3.0, 0.0, 0.0, 0.0, 0.12, 0.05], [1.0, 2.0, 0.0, -0.1, 0.05, 0.03]]
    )  # x v_x + y v_y = 0: v_R = 0
    phi = _phi(pot)
    lim = _limits_fn(pot)(w6s, jnp.full(2, D2))
    for i, w6 in enumerate(w6s):
        iom = integrals_of_motion(phi, w6, D2)
        lam = float(iom.lam)
        lmin, lmax = float(lim.lam_min[i]), float(lim.lam_max[i])
        assert np.isfinite(lmin)
        assert np.isfinite(lmax)
        assert (lmin == lam) != (lmax == lam), (lmin, lam, lmax)
        assert lmin < lmax
        assert not lim.snapped[i]
        assert not lim.shell[i]


def test_find_limits_flags():
    """L_z = 0 pinning and the MINIMUM_RANGE collapse, on KK."""
    pot = _kk()
    w6s = jnp.asarray(
        [
            [4.0, 0.0, 1.0, 0.05, 0.0, 0.08],  # L_z = 0 tube (lam_min pinned)
            [4.0, 0.0, 0.0, 0.03, 0.15, 0.0],  # exactly planar
            AXIS_W6.tolist(),  # on the axis inside the foci
        ]
    )
    lim = _limits_fn(pot)(w6s, jnp.full(3, D2))
    assert lim.planar[1]
    assert lim.nu_max[1] == 0
    for k in ("lam_min", "lam_max", "nu_max"):
        assert np.all(np.isfinite(getattr(lim, k))), (k, getattr(lim, k))
    assert np.all(lim.lam_min >= D2)
    assert np.all(lim.nu_max <= D2)
    # L_z = 0: either the lam_min lane is pinned to D2, or it solved above D2
    assert (lim.lam_min[0] == D2) == lim.lmin_pinned[0]


@pytest.mark.parametrize("name", ["mn", "mw2022"])
def test_nu_turning_point_falls_back_to_absnu(name):
    """At a nu turning point (nu-dot = 0), nu_max = |nu|.

    F(|nu|) = 0 there to roundoff, so the nu lane often has no bracket and
    nu_max takes the C++ fallback |nu| (not NaN, not the planar collapse).
    """
    pot = {"mn": _mn, "mw2022": gp.MilkyWayPotential2022}[name]()
    phi = _phi(pot)
    rng = np.random.default_rng(4)
    n = 60
    R, z = rng.uniform(1.0, 12.0, n), rng.uniform(0.05, 2.0, n)
    vc = np.sqrt(R * np.asarray(jax.vmap(jax.grad(lambda r: phi(r, 0.0 * r)))(R)))
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
    pot = {"mn": _mn, "mw2022": gp.MilkyWayPotential2022}[name]()
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
    pot = {"mn": _mn, "mw2022": gp.MilkyWayPotential2022}[name]()
    w6s = _near_circular_w6s(pot, e)
    for D2s in (jnp.full(len(w6s), MN_D2), _D2_pipeline(pot, w6s)):
        lim = _limits_fn(pot)(w6s, D2s)
        for k in ("lam_min", "lam_max", "nu_max"):
            assert np.all(np.isfinite(getattr(lim, k))), k
        assert np.mean(lim.shell) > 0.99
        ref_lo, ref_hi = _lam_roots(pot, w6s, D2s)
        np.testing.assert_allclose(lim.lam_min, ref_lo, rtol=2e-7, atol=0)
        np.testing.assert_allclose(lim.lam_max, ref_hi, rtol=2e-7, atol=0)


def test_limits_cache_sees_monkeypatched_constant(monkeypatch):
    """The `_limits_fn` cache keys on module constants read at trace time."""
    pot = _mn()
    w6s = _near_circular_w6s(pot, 0.0)
    D2s = jnp.full(len(w6s), MN_D2)
    assert np.mean(_limits_fn(pot)(w6s, D2s).snapped) > 0.99  # 1.0
    monkeypatch.setattr(limits, "SNAP_NOISE", 0.0)
    assert np.mean(_limits_fn(pot)(w6s, D2s).snapped) < 0.6


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

    def vc(R):
        return jnp.sqrt(R * jax.grad(lambda r: phi(r, jnp.asarray(0.0)))(R))

    rows = []  # at (and 1e-3 v_R off) a turning point, from the shells
    for R in (3.0, 4.0, 7.0):
        v = float(vc(jnp.asarray(R)))
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
        v = np.asarray(jax.vmap(vc)(jnp.asarray(R)))
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


def test_F_tolerates_nonfinite_tau():
    """Frozen/invalid lanes may carry NaN/inf tau: no raise, no NaN gradients."""
    w6 = _w6s(1)[0]
    pot = _kk()
    phi = _phi(pot)

    def valid_sum(w):
        iom = integrals_of_motion(phi, w, D2)
        F = make_F(phi, iom)
        tot = 0.0
        for t in (jnp.nan, jnp.inf, -jnp.inf, -1.0):
            R3, z3, mult3, I3c = F.lane_coords(jnp.stack([0.5 * iom.absnu, 20.0, t]))
            tot = tot + (phi(R3, z3) * mult3 + I3c)[:2].sum()
            t3 = jnp.asarray([t, iom.absnu * 0.5, 20.0])
            tot = tot + F.F_lam(t3)[2] + F.F_nu(t3)[1] + F.dF_lam(t3)[2]
            tot = tot + F.dF_nu(t3)[1] + F.d2F_lam(t3)[2]
            tot = tot + F.F_nu_u(jnp.asarray([t, 0.3]))[1]
            s = jnp.asarray([t, 0.5])
            tot = tot + F.F_semi(s, iom.lam, jnp.asarray(0.0))[1]
        return tot

    for jac in (jax.jacfwd, jax.jacrev):
        g = jac(valid_sum)(w6)
        assert np.all(np.isfinite(g)), g
    # and the invalid lanes themselves are flagged, not silently finite
    _, _, F = _setup()
    for t in (jnp.nan, jnp.inf, -1.0):
        assert np.isnan(F.F_lam(jnp.asarray(t)))
        assert np.isnan(F.F_nu(jnp.asarray(t)))


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
    pot = {"kk": _kk, "mn": _mn, "nfw": gp.NFWPotential}[name]
    pot = (
        pot()
        if name != "nfw"
        else pot(m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), units="galactic")
    )
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
        pot, d2 = (
            gp.NFWPotential(
                m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), units="galactic"
            ),
            56.25,
        )
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
    pot = gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), units="galactic")
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
    _, iom, _ = (
        None,
        jax.vmap(lambda w, d: integrals_of_motion(_phi(pot), w, d))(w6s, D2s),
        None,
    )
    assert np.all(lim.lam_max - iom.lam > 1e-3 * (iom.lam - 56.25 + 1e-12))
    np.testing.assert_allclose(lim.lam_max[0], 56.398, rtol=1e-4)


def test_lz_squared_underflow():
    """R = z = 1e-300: L_z^2 underflows to 0, and is treated as L_z = 0.

    F(D2) = -L_z^2 D2 / 2 is then -0, which the solver took for a converged
    root at nu_max = D2. MN, fixed Delta: nu_max matches the bisection of
    F_nu (or is NaN), never D2.
    """
    pot = _mn()
    vc = _VC["mn"]
    w6s = jnp.asarray([[1e-300, 0.0, 1e-300, 0.1 * vc, 0.1 * vc, 0.1 * vc]])
    D2s = jnp.full(1, 16.0)
    lim = _limits_fn(pot)(w6s, D2s)
    assert lim.nu_max[0] != 16.0
    if np.isfinite(lim.nu_max[0]):
        ref = _nu_bisect(pot, w6s, D2s)
        np.testing.assert_allclose(lim.nu_max, ref, rtol=1e-8)
