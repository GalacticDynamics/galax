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
    _jac_both,
    _kk,
    _kk_F,
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
    attach_tangents,
    default_phi_inf,
    find_limits,
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


def test_near_shell_lane_not_snapped_kepler():
    """A near-circular Kepler orbit in the plane (D2 = 4) is not snapped.

    Its extent is 3.1e-4 at lam = 99.2. The roundoff scale S of F at lam
    takes its I and Phi terms from the phi call at lam. Taken from D2 (1 +
    D2_SIDE) instead, the lane snapped to 99.20946, half its extent from
    either limit (mpmath).
    """
    pot = gp.KeplerPotential(m_tot=u.Q(1e11, "Msun"), units="galactic")
    w6s = jnp.asarray([[9.757533623071872, 0, 0, 1.4906015819851504e-07,
                        0.21471581123911318, 0]])  # fmt: skip
    ref, ext = (99.20938657429292, 99.20969285106199), 3.0628e-4
    modes = {
        k: _modes(_phi(pot), 4.0, w6s, k) for k in ("lam_min", "lam_max", "snapped")
    }
    for mode in modes["snapped"]:
        assert not np.any(modes["snapped"][mode]), mode
        for k, r in zip(("lam_min", "lam_max"), ref, strict=True):
            np.testing.assert_allclose(modes[k][mode], r, atol=1e-4 * ext, rtol=0)


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
    lanes 1.4% and 1.2% off. The bracket is also closed from 1.1 |nu| (the
    first nu point below the ladder): nu_max within 1e-4 of a bisection of F
    (measured 4.1e-6).
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


def test_lz0_lam_within_D2_side_not_snapped():
    """L_z = 0 with D2 < lam <= D2 (1 + D2_SIDE): F(lam_pos) is NaN there.

    lam_pos = lam is at or below lam_lower, and with F''(lam) > 0 there is no
    split. The lane was snapped to lam_min = lam_max = lam. It now starts
    from D2 (1 + D2_SIDE), as at lam = D2. Plummer (1e11, 1 kpc) at R = 1e-9
    (lam = D2, unchanged) and 1e-8, 2.03e-8, 2.79e-8 Delta (in the band),
    against mpmath; and a Plummer minus a smaller Plummer whose band lane is
    unbound (lam_max NaN, it was lam).
    """

    def plum(m, b):
        return gp.PlummerPotential(
            m_tot=u.Q(m, "Msun"), r_s=u.Q(b, "kpc"), units="galactic"
        )

    pot, d2 = plum(1e11, 1.0), 1.5665770046181047
    R = [1.251629739427002e-09, 1.251629739427002e-08,
         2.540808371036814e-08, 3.492046973001335e-08]  # fmt: skip
    w6s = jnp.asarray(
        [[r, 0, 0.5569, -0.4787551904160538, 0, -0.6383402538880718] for r in R]
    )
    D2s = jnp.full(4, d2)
    lam = jax.vmap(lambda a: integrals_of_motion(_phi(pot), a, d2).lam)(w6s)
    assert lam[0] == d2
    assert np.all((lam[1:] > d2) & (lam[1:] <= d2 * (1 + limits.D2_SIDE)))
    lim = _limits_fn(pot)(w6s, D2s)
    assert not np.any(lim.snapped | lim.shell)
    ref = [29.272928125277541, 29.272928183656834,
           29.27292825046873, 29.272928299766855]  # fmt: skip
    np.testing.assert_allclose(lim.lam_max, ref, rtol=1e-13, atol=0)
    np.testing.assert_allclose(lim.lam_min, d2, rtol=1e-15, atol=0)

    hole = gp.CompositePotential(a=plum(1e11, 5.0), b=plum(-6e10, 1.0))
    w6 = [2.2117974176508188e-08, 0.0, 0.5568904825309939,
          -0.02714827994671754, 0.0, -0.02436039854983938]  # fmt: skip
    lim = _limits_fn(hole)(jnp.asarray([w6]), jnp.full(1, d2))
    assert not lim.snapped[0]
    assert np.isnan(lim.lam_max[0])
    np.testing.assert_allclose(lim.lam_min[0], d2, rtol=1e-15, atol=0)


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


# ---------------------------------------------------------------------------
# L_z = 0 on the z axis: the box/tube decision at a double zero of F at D2.
# Inside the foci (lam = D2) F_nu ~ (tau - D2)^2 near D2, and outside them
# (|nu| = D2) so is F_lam; F at D2 (1 -+ D2_SIDE) is then below its roundoff.


def _first_root(f, start, end, n_grid=300, stop=1e-6):
    """First root of f walking from ``start`` towards ``end`` (rows), else ``end``.

    The grid start + (end - start) g, g = 10^(-14..-0.3) and 1 - 10^(log10
    stop..-1), stops ``stop`` (end - start) short of ``end``: where F has a
    double zero at ``end``, F nearer it is roundoff. Bisected between the
    last grid point (or ``start``) and the first with F < 0.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    lo = np.log10(stop)
    g = np.concat([np.logspace(-14, -0.3, n_grid), 1 - np.logspace(lo, -1, 100)])
    g = np.sort(g)
    T = start[:, None] + (end - start)[:, None] * g
    neg = np.asarray(f(T.T)).T < 0
    i = np.argmax(neg, axis=1)
    rows = np.arange(len(start))
    prev = np.where(i > 0, T[rows, np.maximum(i - 1, 0)], start)
    root = bisect(f, prev, T[rows, i])
    return np.where(neg.any(axis=1), root, end)


def _axis_refs(pot, w6s, D2s, stop_lam=1e-6):
    """(lam_min, nu_max) as first roots of F (see `_first_root`), in numpy."""
    phi = _phi(pot)
    iom = jax.vmap(lambda w, d: integrals_of_motion(phi, w, d))(w6s, D2s)
    lam, absnu, d2 = (np.asarray(x) for x in (iom.lam, iom.absnu, D2s))

    def on(name):
        f = _F_rows(pot, name)
        return lambda t: np.asarray(f(w6s, D2s, jnp.asarray(t).T)).T

    on_lam, on_nu = on("F_lam"), on("F_nu")
    return _first_root(on_lam, lam, d2, stop=stop_lam), _first_root(on_nu, absnu, d2)


def _axis_w6s(n, *, inside, seed=5, Delta=2.0):
    """L_z = 0 points on the z axis (R = 0, 1e-12, 1e-9), at rest or moving."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.05, 0.975, n) if inside else rng.uniform(1.025, 3.0, n)
    z *= Delta * rng.choice([-1.0, 1.0], n)
    R = rng.choice([0.0, 0.0, 1e-12, 1e-9], n)
    k = np.arange(n) % 5
    vR = np.array([0, 0, 1e-6, 1e-3, 0.3])[k] * rng.normal(size=n)
    vz = np.array([0, 1e-3, 0.05, 0.2, 0.1])[k] * rng.normal(size=n)
    return jnp.asarray(np.stack([R, 0 * R, z, vR, 0 * R, vz], 1))


def test_nu_max_on_axis_at_rest_kk():
    """KK at rest on the axis inside the foci (exact D2 = 8): nu_max = z^2.

    lam = D2 there, so F_nu has a double zero at D2 and F(D2 (1 - D2_SIDE))
    ~ 1e-31 is roundoff. Its sign decided the box/tube pin: in one batch of
    27 the lanes z = 2.2..2.7 had nu_max pinned to D2, one lane at a time
    they had z^2. nu_max is now pinned to D2 only where no sampled point
    in (|nu|, D2) has F < 0.
    """
    pot = _kk()
    z = np.linspace(0.1, 2.7, 27)
    w6s = jnp.asarray(np.stack([0 * z, 0 * z, z, 0 * z, 0 * z, 0 * z], 1))
    D2s = jnp.full(len(z), D2)
    big = _limits_fn(pot)(w6s, D2s)
    one = [_limits_fn(pot)(w6s[i : i + 1], D2s[i : i + 1]) for i in range(len(z))]
    for lim in (big, jax.tree.map(lambda *a: np.concat(a), *one)):
        np.testing.assert_allclose(lim.nu_max, z**2, rtol=1e-12, atol=0)
        assert not np.any(lim.nu_pinned | lim.planar)


@pytest.mark.parametrize("name", ["kk", "mn", "nfw", "mw2022"])
def test_axis_lz0_inside_foci(name):
    """L_z = 0 on the axis inside the foci (fixed D2 = 4): nu_max, every lane.

    The audit repros (NFW [0,0,0.5] at rest: 4.0 against 0.25; MW2022
    [0,0,1.4] at v_z 0.05: 4.0 against 2.106; MN [0,0,1.9] at rest: 4.0
    against 3.61) and 40 random points, at rest or moving (v_z, a tiny
    v_R), so the root is not always at the start point. Before, 3 to 16
    of the 40 had nu_max pinned to D2. nu_max is the first root of F_nu
    above |nu| (a bisection, `_axis_refs`) to 1e-9 of max(nu_max, 1e-4 D2),
    the same in one lane at a time, and lam_min = D2.
    """
    pot = _GRAD_POTS[name]()
    rep = [[0, 0, 0.5, 0, 0, 0], [0, 0, 1.4, 0, 0, 0.05], [0, 0, 1.9, 0, 0, 0]]
    w6s = jnp.concat([jnp.asarray(rep, dtype=float), _axis_w6s(40, inside=True)])
    n = len(w6s)
    D2s = jnp.full(n, 4.0)
    lim = _limits_fn(pot)(w6s, D2s)
    _, ref = _axis_refs(pot, w6s, D2s)
    assert np.all(ref < 4.0)  # every lane has a root below D2: none is pinned
    err = np.abs(np.asarray(lim.nu_max) - ref)
    np.testing.assert_array_less(err, 1e-9 * np.maximum(ref, 4e-4))
    assert not np.any(lim.nu_pinned)
    np.testing.assert_array_equal(lim.lam_min, 4.0)
    for i in (0, 1, 2, 5, 9):
        one = _limits_fn(pot)(w6s[i : i + 1], D2s[i : i + 1])
        np.testing.assert_allclose(one.nu_max[0], lim.nu_max[i], rtol=1e-12, err_msg=i)


@pytest.mark.parametrize("name", ["kk", "mn", "nfw"])
def test_axis_lz0_at_rest_nu_max_modes_agree(name):
    """At rest on the axis inside the foci, nu_max is |nu| in every mode.

    With d sigma/dt = 0 the point is its own nu turning point and F(|nu|)
    is 0 up to roundoff. Its sign changed with the compile: in the
    43-lane batch of `test_axis_lz0_inside_foci` lanes 3 (kk, nfw) and 28
    (mn, nfw) read F(|nu|) > 0 and solved to nu_max ~ 4.0 against 2.53
    and 2.92. The at-rest lanes of that batch are checked in the batch,
    one lane at a time under jit, and in eager mode.
    """
    pot = _GRAD_POTS[name]()
    phi = _phi(pot)
    rep = [[0, 0, 0.5, 0, 0, 0], [0, 0, 1.4, 0, 0, 0.05], [0, 0, 1.9, 0, 0, 0]]
    w6s = jnp.concat([jnp.asarray(rep, dtype=float), _axis_w6s(40, inside=True)])
    d2 = jnp.asarray(4.0)
    rest = np.flatnonzero(np.all(np.asarray(w6s[:, 3:]) == 0, axis=1))
    assert {3, 28} <= set(rest.tolist())

    def one(w6, d):
        iom = integrals_of_motion(phi, w6, d)
        lim = find_limits(make_F(phi, iom), iom, default_phi_inf(phi, iom))
        return jnp.stack([lim.nu_max, iom.absnu])

    batch = _limits_fn(pot)(w6s, jnp.full(len(w6s), 4.0)).nu_max[rest]
    jone = jax.jit(one)
    absnu = np.asarray([one(w6s[i], d2)[1] for i in rest])
    for mode, out in (
        ("batch", np.asarray(batch)),
        ("jit", np.asarray([jone(w6s[i], d2)[0] for i in rest])),
        ("eager", np.asarray([one(w6s[i], d2)[0] for i in rest])),
    ):
        np.testing.assert_allclose(out, absnu, rtol=1e-12, err_msg=mode)


def _modes(phi, d2, w6s, key):
    """Return ``key`` of the limits of ``w6s`` in each mode.

    The modes: a jitted batch, one lane at a time under jit, and eager.
    """

    def one(w6, d):
        iom = integrals_of_motion(phi, w6, d)
        return find_limits(make_F(phi, iom), iom, default_phi_inf(phi, iom))

    d = jnp.asarray(d2)
    jone = jax.jit(one)
    return {
        "batch": np.asarray(getattr(jax.jit(jax.vmap(one, (0, None)))(w6s, d), key)),
        "jit": np.asarray([getattr(jone(w, d), key) for w in w6s]),
        "eager": np.asarray([getattr(one(w, d), key) for w in w6s]),
    }


def _phi_ring_axis(R, z):
    """MN-like disk plus a flattened Plummer core (G = 1)."""
    disk = -1.0 / jnp.sqrt(R * R + (1.0 + jnp.sqrt(z * z + 0.05**2)) ** 2)
    return disk - 1.0 / jnp.sqrt(R * R + (z / 1.5) ** 2 + 0.25)


def _phi_disk_masses(R, z):
    """MN disk plus two Plummer masses on the z axis at z = +-1 (G = 1)."""
    disk = -1.0 / jnp.sqrt(R * R + (1.0 + jnp.sqrt(z * z + 0.01)) ** 2)
    up = -0.5 / jnp.sqrt(R * R + (z - 1.0) ** 2 + 0.04)
    return disk + up - 0.5 / jnp.sqrt(R * R + (z + 1.0) ** 2 + 0.04)


@pytest.mark.parametrize(
    ("phi", "d2", "w6", "ref"),
    [
        (_phi_ring_axis, 4.0, [5.162040716333926, 0, 3.223754822995957, 0, 0, 0], 4.0),
        (
            _phi_ring_axis,
            4.0,
            [5.162040716333926, 0, 3.223754822995957, 0, 0.02, 0],
            3.9864911975596359,
        ),
        (
            _phi_disk_masses,
            0.5,
            [0.2931708082449846, 0, 0.29035368631704117, 0, 0, 0],
            0.5,
        ),
    ],
    ids=["ring_axis", "ring_axis_Lz", "disk_masses"],
)
def test_sigdot0_lower_nu_turning_point(phi, d2, w6, ref):
    """At rest at a lower nu turning point, nu_max is the turning point above.

    With d sigma/dt = 0 the point is at a nu turning point. Every such point
    was given nu_max = |nu|, but here F_nu > 0 from |nu| up (mpmath: F(|nu|
    (1 + 1e-6)) = 7.1e-7 on the first), so |nu| is the lower turning point.
    nu_max was |nu| (1.039 and 0.0703) where it is D2 (L_z = 0, F > 0 up to
    D2) or the mpmath root (L_z = 0.10, 3.98649...), and a v_z of 1e-12
    already gave D2. The same in a batch, one lane at a time under jit and
    in eager mode.
    """
    w6s = jnp.asarray([w6], dtype=float)
    for mode, out in _modes(phi, d2, w6s, "nu_max").items():
        np.testing.assert_allclose(out[0], ref, rtol=1e-10, err_msg=mode)


@pytest.mark.parametrize(
    ("d2", "z"),
    [(4.0, 2.0 * (1 - 3e-8)), (4.0, 2.0 * (1 - 3e-9)), (1.0, 1 - 1e-8)],
)
def test_sigdot0_unresolved_is_upper_nu_turning_point(d2, z):
    """At rest on the axis just inside a focus, nu_max is |nu| in every mode.

    KK, D2 = 4 or 1, |nu| within 1e-7 D2 of D2. The point is its upper nu
    turning point, but F at t_tp (between |nu| and D2) is a positive
    number below its roundoff bound. Read as F > 0 it made |nu| a lower
    turning point and nu_max D2. Only a resolved F(t_tp) > 0 does that.
    """
    phi = _phi(_kk())
    w6s = jnp.asarray([[0.0, 0, z, 0, 0, 0]])
    absnu = float(integrals_of_motion(phi, w6s[0], jnp.asarray(d2)).absnu)
    assert absnu < d2
    for mode, out in _modes(phi, d2, w6s, "nu_max").items():
        np.testing.assert_allclose(out[0], absnu, rtol=1e-12, err_msg=mode)


@pytest.mark.parametrize(
    ("pot", "d2", "w6", "ref", "rtol"),
    [
        (
            lambda: gp.IsochronePotential(
                m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
            ),
            1e-4,
            [1e-10, 0, 0.007, 0, 0, 0.000846805067150659],
            1.0000987539159073e-4,
            2e-3,
        ),
        (
            lambda: gp.PlummerPotential(
                m_tot=u.Q(1e11, "Msun"), r_s=u.Q(1.0, "kpc"), units="galactic"
            ),
            1e-4,
            [1e-10, 0, 0.007, 0, 4.789550856972426e-07, 0.0047900298120581225],
            1.0001068469640094e-4,
            5e-4,
        ),
        (
            _mn,
            1.0,
            [1e-08, 0, 0.01, 0, 2.2393428509463645e-07, 0.11196714254731822],
            1.0008175072588148,
            1e-6,
        ),
    ],
    ids=["isochrone", "plummer", "mn"],
)
def test_axis_inside_foci_lam_max_converged(pot, d2, w6, ref, rtol):
    """lam_max on the axis inside the foci with a small v_R or L_z.

    R ~ 1e-8 Delta, so lam rounds to D2 or an ulp above it, and v_z carries
    the orbit past a focus (lam_max - D2 ~ 1e-4 to 1e-3 D2). F'(D2) > 0 is
    resolved, so the model's root above lam is within an ulp of lam and the
    ladder above was empty; in eager mode the isochrone lane starts from D2
    (1 + D2_SIDE) with F rounding to 0 at the first points above. The lanes
    were closed only by lam_pos e^8 (or not at all), and 16 iterations left
    lam_max 7.8e-2 (isochrone), 1.6e-2 (Plummer) and 2.2e-6 (MN) of the
    extent off. Against mpmath to ``rtol`` of the extent lam_max - D2, and
    the modes agree to it. F is within its roundoff over +-3e-3 (isochrone)
    and +-7e-5 (Plummer, on macOS) of the extent around the root, hence their
    rtol. On Linux CI the Plummer batch lane landed 1.2e-4 off, so its rtol
    is 5e-4, still 30 times closer than before the fix.
    """
    w6s = jnp.asarray([w6], dtype=float)
    outs = _modes(_phi(pot()), d2, w6s, "lam_max")
    ext = ref - d2
    for mode, out in outs.items():
        assert abs(out[0] - ref) < rtol * ext, (mode, out[0])
    vals = [o[0] for o in outs.values()]
    assert max(vals) - min(vals) < rtol * ext, vals


def _kk_first_root_decimal(pot, w6, which):
    """KK's lam_min or nu_max (``which``) at L_z = 0, in 40-digit decimals.

    The first root of F walking from lam (or |nu|) towards D2, else D2: the
    grid start + (D2 - start) g, g = 10^(-20..0) and 1 - 10^(-15..-0.3),
    then a bisection to past 40 digits. Needs R > 0.
    """
    D = decimal.Decimal
    g = np.concat([np.logspace(-20, -0.3, 300), 1 - np.logspace(-15, -0.3, 200)])
    g = np.sort(g)
    with decimal.localcontext(prec=40):
        F, d2, lam, nu = _kk_F(
            pot, w6, D, lambda a: a.sqrt(), lambda a, b: (a * a + b * b).sqrt()
        )
        start = nu if which == "nu_max" else lam
        prev = start
        for gk in g:
            t = start + (d2 - start) * D(float(gk))
            if F(t) < 0:
                a, b = prev, t
                for _ in range(170):  # 2^-170 of the bracket: past 40 digits
                    mid = (a + b) / 2
                    a, b = (mid, b) if F(mid) > 0 else (a, mid)
                return (a + b) / 2
            prev = t
        return d2


def _jvp_fd_kk(pot, w6s, key, cols, h=1e-12):
    """(jvp of limit ``key``, decimal central differences) along w6 ``cols``."""
    D = decimal.Decimal

    def lim_of(w, t):
        return jax.jvp(
            lambda x: getattr(_live_limits(pot, x, jnp.asarray(D2))[0], key), (w,), (t,)
        )[1]

    out = []
    for c in cols:
        T = jnp.zeros_like(w6s).at[:, c].set(1.0)
        J = np.asarray(jax.jit(jax.vmap(lim_of))(w6s, T))
        fd = []
        for w in np.asarray(w6s):
            wp, wm = [D(float(x)) for x in w], [D(float(x)) for x in w]
            wp[c] += D(h)
            wm[c] -= D(h)
            rp, rm = (_kk_first_root_decimal(pot, x, key) for x in (wp, wm))
            fd.append(float((rp - rm) / (2 * D(h))))
        out.append((J, np.asarray(fd)))
    return out


def test_axis_lz0_inside_foci_tangents_kk():
    """KK (D2 = 8), R = 1e-12 inside the foci (lam = D2), moving: nu_max tangents.

    These lanes had nu_max pinned to D2 (and its tangent, 0 at a fixed
    Delta) by the sign of a roundoff-level F. The d/dz, d/dv_R and d/dv_z
    tangents now match central differences of KK's decimal root to 1e-6.
    """
    pot = _kk()
    w6s = jnp.asarray(
        [
            [1e-12, 0.0, 2.3, 0.0, 0.0, 0.05],
            [1e-12, 0.0, -2.5, 1e-6, 0.0, 0.01],
            [1e-12, 0.0, 1.2, 1e-3, 0.0, -0.08],
            [1e-12, 0.0, 2.6, 0.0, 0.0, 1e-3],
        ]
    )
    lim = _limits_fn(pot)(w6s, jnp.full(len(w6s), D2))
    ref = np.array(
        [float(_kk_first_root_decimal(pot, w, "nu_max")) for w in np.asarray(w6s)]
    )
    np.testing.assert_allclose(lim.nu_max, ref, rtol=1e-12, atol=0)
    assert not np.any(lim.nu_pinned)
    for J, fd in _jvp_fd_kk(pot, w6s, "nu_max", (2, 3, 5)):
        np.testing.assert_allclose(J, fd, rtol=1e-6, atol=1e-9 * np.max(np.abs(fd)))


def test_axis_lz0_outside_foci_kk():
    """KK (D2 = 8), R = 1e-12 on the axis outside the foci (|nu| = D2): lam_min.

    The mirror case: F_lam has a double zero at D2 and F(D2 (1 + D2_SIDE))
    is roundoff. A small v_R puts lam_min ~1e-6 (lam - D2) above D2, below
    the ladder towards D2 (its last point was 10^-3 of the way): the lane
    started from D2 (1 + D2_SIDE) on a roundoff F and stayed there (up to
    2e-5 relative off), or was pinned to D2 by that F's sign. lam_min is now
    KK's decimal first root below lam (D2 where F > 0 down to D2: at rest)
    to 1e-10 relative, pinned exactly where that root is D2, and nu_max =
    D2. Tangents of the solved lanes match central differences to 1e-5.
    With v_R ~ 1e-6..1e-3, lam_min - D2 is ~2e-13..3e-5 D2, and the ladder
    point at 10^-9 closes the smallest brackets: lam_min to 1e-13 (measured
    1.3e-15; the ladder ending at 10^-6 left these 3.6e-11 off).
    """
    pot = _kk()
    rng = np.random.default_rng(3)
    n = 20
    z = rng.uniform(3.0, 6.0, n) * rng.choice([-1.0, 1.0], n)
    vR = np.array([0, 1e-4, 1e-3, 3e-2, 0.3])[np.arange(n) % 5] * rng.normal(size=n)
    vz = rng.normal(0, 0.2, n) * (np.arange(n) % 5 != 0)
    w6s = jnp.asarray(np.stack([np.full(n, 1e-12), 0 * z, z, vR, 0 * z, vz], 1))
    lim = _limits_fn(pot)(w6s, jnp.full(n, D2))
    ref = np.array(
        [float(_kk_first_root_decimal(pot, w, "lam_min")) for w in np.asarray(w6s)]
    )
    np.testing.assert_allclose(lim.lam_min, ref, rtol=1e-10, atol=0)
    np.testing.assert_array_equal(lim.lmin_pinned, ref == D2)
    np.testing.assert_array_equal(lim.nu_max, D2)
    near = np.flatnonzero((ref > D2) & (ref < D2 * (1 + 1e-4)))
    assert len(near) >= 4
    for J, fd in _jvp_fd_kk(pot, w6s[near[:4]], "lam_min", (2, 3, 5)):
        np.testing.assert_allclose(J, fd, rtol=1e-5, atol=1e-9 * np.max(np.abs(fd)))

    rng = np.random.default_rng(3)
    z = rng.uniform(3.0, 6.0, n) * rng.choice([-1.0, 1.0], n)
    vR = np.array([1e-6, 1e-5, 1e-4, 1e-3])[np.arange(n) % 4] * rng.normal(size=n)
    vz = rng.normal(0, 0.2, n)
    w6s = jnp.asarray(np.stack([np.full(n, 1e-12), 0 * z, z, vR, 0 * z, vz], 1))
    lim = _limits_fn(pot)(w6s, jnp.full(n, D2))
    ref = np.array(
        [float(_kk_first_root_decimal(pot, w, "lam_min")) for w in np.asarray(w6s)]
    )
    assert np.all(ref > D2)
    np.testing.assert_allclose(lim.lam_min, ref, rtol=1e-13, atol=0)


@pytest.mark.parametrize("name", ["kk", "isochrone"])
def test_axis_lz0_pipeline_delta(name):
    """L_z = 0 along the axis (R = v_R = 0) with the pipeline Delta (1e-4 r).

    |z| >> Delta, and the orbit runs along the axis through the centre, so
    lam_min = nu_max = D2 exactly. D2 ~ 1e-8 lam, and F near D2 is roundoff
    on both branches. A roundoff F(D2 (1 + D2_SIDE)) < 0 left lam_min
    solved on a roundoff root ~1e-6 D2 above D2 (isochrone, 2 of 80 axis
    lanes). nu_max is pinned there with F(D2 (1 - D2_SIDE)) < 0 by
    roundoff on some lanes, and its value must be D2 (0 is the F(0) <= 0
    pin). Both are D2 to 1e-14.
    """
    pot = (
        _kk()
        if name == "kk"
        else gp.IsochronePotential(
            m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
        )
    )
    w6s = jnp.concat([_axis_w6s(40, inside=k, seed=9) for k in (True, False)])
    w6s = w6s[(w6s[:, 0] == 0) & (w6s[:, 3] == 0)]
    assert len(w6s) >= 16
    D2s = _D2_pipeline(pot, w6s)
    lim = _limits_fn(pot)(w6s, D2s)
    np.testing.assert_allclose(lim.lam_min, D2s, rtol=1e-14, atol=0)
    np.testing.assert_allclose(lim.nu_max, D2s, rtol=1e-14, atol=0)


def test_axis_lz0_box_past_the_foci_pinned():
    """L_z = 0 along the axis fast enough to pass the foci: nu_max = D2, pinned.

    KK (D2 = 8) at R = 0 and 1e-12 inside the foci, v_z = f v_th (v_th just
    reaches the focus, f > 1), and the NFW lane outside the foci of the
    review. F < 0 nowhere below D2, but F at the ladder points nearest D2
    is roundoff there (a double zero on the axis), and a roundoff F < 0 at
    one of them unpinned nu_max (26 of 36 KK lanes; NFW 4 - 1.6e-14) with a
    spurious tangent up to 3e-6 (5.4e-6 against 0 for NFW). Only an F below
    -SNAP_NOISE x its roundoff unpins. nu_max is D2, pinned, with D2's
    tangent (0 at a fixed Delta).
    """
    pot = _kk()
    phi = _phi(pot)
    Df = float(np.sqrt(D2))

    def P(z):
        return float(phi(jnp.asarray(0.0), jnp.asarray(z)))

    rows = [
        [R, 0.0, z, 0.0, 0.0, f * np.sqrt(2 * (P(Df) - P(z)))]
        for z in (0.3, 1.0, 1.7, 2.2, 2.5, 2.7)
        for f in (1.00001, 1.001, 1.5)
        for R in (0.0, 1e-12)
    ]
    cases = [(pot, jnp.asarray(rows), D2)]
    cases.append((_nfw(), jnp.asarray([[3.4e-7, 0, 5.2756, -5.3e-9, 0, 0]]), 4.0))
    for p, w6s, d2 in cases:
        lim = _limits_fn(p)(w6s, jnp.full(len(w6s), d2))
        np.testing.assert_array_equal(lim.nu_max, d2)
        assert np.all(lim.nu_pinned)

        def nu_of(w, t, p=p, d2=d2):
            return jax.jvp(
                lambda x: _live_limits(p, x, jnp.asarray(d2))[0].nu_max, (w,), (t,)
            )[1]

        t = jnp.zeros_like(w6s).at[:, jnp.asarray([2, 3, 5])].set(1.0)
        J = np.asarray(jax.jit(jax.vmap(nu_of))(w6s, t))  # z + v_R + v_z
        np.testing.assert_array_equal(J, 0.0)

    # At rest just inside a focus (|nu| = D2 (1 - 6.6e-8), fixed D2 = 4): F
    # between |nu| and D2 is not resolved, and the lane is at its nu turning
    # point (d sigma/dt = 0): nu_max = |nu|, not the D2 pin (6.6e-8 off).
    w = jnp.asarray([[1.2647547397054988e-07, 0.0, -1.9999999948424243, 0, 0, 0]])
    lim = _limits_fn(pot)(w, jnp.full(1, 4.0))
    absnu = integrals_of_motion(phi, w[0], jnp.asarray(4.0)).absnu
    assert float(lim.nu_max[0]) == float(absnu) < 4.0 * (1 - 6e-8)
    assert not lim.nu_pinned[0]


def test_axis_lz0_inside_foci_lam_max():
    """On the axis inside the foci with v_R = 0, an orbit that passes a focus.

    lam = D2 and F_lam has a double zero there: F(D2) = 0 and F'(D2) ~ v_R^2
    is roundoff. A roundoff F' <= 0 set the lam_max edge, and lam_max came
    out D2 (flagged shell) by the sign of that roundoff, so it changed with
    jit, the batch and the AD mode (Hernquist below: D2 in jvp, the root in
    vjp). F' within its roundoff now counts as 0 at D2, and F'' > 0 above
    D2 leaves lam_max solved.

    KK (D2 = 8) at R = 0 and 1e-12, v_z = f v_th (v_th just reaches the
    focus): lam_max is KK's decimal root at R = 1e-12 to 1e-9 of the extent
    (measured 4e-10; F's roundoff near D2 limits f = 1.001), lam_min = D2
    pinned; f = 0.5 stays inside the foci (lam_max = D2, shell). MN and NFW
    at D2 = 0.25 (the audit repros) against mpmath to 1e-12. Hernquist at
    D2 = 1e6: the values against mpmath; at z = 999 jacfwd = jacrev and the
    z, v_z and D2 tangents against mpmath central differences to 1e-6.
    """
    pot = _kk()
    phi = _phi(pot)
    Df = float(np.sqrt(D2))

    def P(z):
        return float(phi(jnp.asarray(0.0), jnp.asarray(z)))

    grid = [(z, f, R) for z in (0.3, 1.0, 2.2, 2.5) for f in (0.5, 1.001, 1.5)
            for R in (0.0, 1e-12)]  # fmt: skip
    w6s = jnp.asarray(
        [[R, 0.0, z, 0.0, 0.0, f * np.sqrt(2 * (P(Df) - P(z)))] for z, f, R in grid]
    )
    lim = _limits_fn(pot)(w6s, jnp.full(len(w6s), D2))
    inside = np.array([f < 1 for _, f, _ in grid])
    ref = np.array(
        [
            D2 if k else kk_turning_points(pot, w.at[0].set(1e-12), exact=True)[1]
            for w, k in zip(w6s, inside, strict=True)
        ]
    )
    np.testing.assert_array_less(
        np.abs(np.asarray(lim.lam_max) - ref), 1e-9 * (ref - D2) + 1e-15 * ref
    )
    np.testing.assert_array_equal(lim.shell, inside)
    np.testing.assert_array_equal(lim.lam_min, D2)
    assert np.all(lim.lmin_pinned[~inside])

    hern = gp.HernquistPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
    )
    cases = [  # (pot, w6, D2, mpmath lam_max)
        (_mn(), [0, 0, 0.01517230543172058, 0, 0, 0.15294396559678883], 0.25,
         2.0858244810666755),
        (_nfw(), [0, 0, -0.05, 0, 0, 0.2321470551386121], 0.25, 9.423453619130255),
        (hern, [0, 0, 999.0, 0, 0, 0.00899400614742016], 1e6, 1205302.5378224468),
        # F'(D2) = -2.1e-11 by roundoff: 16 times the roundoff of the terms
        # left after I_3 - f_nu cancels their lam Phi(0, 0)
        (hern, [0, 0, 831.129945041045, 0, 0, 0.01830386526917468], 1e6,
         1220947.9004422508),
    ]  # fmt: skip
    for p, w6, d2, ref in cases:
        lim = _limits_fn(p)(jnp.asarray([w6], dtype=float), jnp.full(1, d2))
        np.testing.assert_allclose(lim.lam_max, ref, rtol=1e-12, atol=0)
        assert not lim.shell[0]

    def lmax(w7):
        out = _live_limits(hern, w7[:6], w7[6])[0].lam_max[None]
        return out, out

    w7 = jnp.asarray([[*cases[2][1], 1e6]], dtype=float)
    Jf, Jr, (val,) = _jac_both(lmax, w7)
    np.testing.assert_allclose(val, cases[2][3], rtol=1e-12)
    cd = [2952.881377, 53050739.53, -0.15083246]  # mpmath d/dz, d/dv_z, d/dD2
    for J in (Jf, Jr):
        np.testing.assert_allclose(J[0, 0, [2, 5, 6]], cd, rtol=1e-6)


@pytest.mark.parametrize("name", ["kk", "mn", "nfw"])
def test_axis_lz0_outside_foci_on_axis(name):
    """R = 0 on the axis outside the foci (|nu| = D2 exactly), small v_R.

    F_lam has a double zero at D2, so F(D2 (1 + D2_SIDE)) is roundoff and
    only the lam_min ladder sees the root, ~1e-7 to 1e-5 (lam - D2) above
    D2. Most of these lanes are unsplit (F'' > 0 at lam): the ladder runs
    from lam_pos there, it must reach 10^-9, and a closed ladder unpins.
    lam_min is the first root of F_lam below lam (a bisection) to 1e-10,
    never pinned.
    """
    pot = _GRAD_POTS[name]()
    rng = np.random.default_rng(4)
    n = 24
    z = rng.uniform(2.3, 6.0, n) * rng.choice([-1.0, 1.0], n)
    vR = np.array([1e-4, 3e-4, 1e-3, 3e-3])[np.arange(n) % 4] * rng.choice([-1, 1], n)
    vz = rng.normal(0, 0.2, n)
    w6s = jnp.asarray(np.stack([0 * z, 0 * z, z, vR, 0 * z, vz], 1))
    D2s = jnp.full(n, 4.0)
    lim = _limits_fn(pot)(w6s, D2s)
    ref, _ = _axis_refs(pot, w6s, D2s, stop_lam=1e-10)
    assert np.all(ref > 4.0 * (1 + 1e-9))
    np.testing.assert_allclose(lim.lam_min, ref, rtol=1e-10, atol=0)
    assert not np.any(lim.lmin_pinned)


def test_lz0_lam_min_past_a_thin_interval_above_D2():
    """L_z = 0 near the plane: lam_min is the orbit's root, not D2.

    MW2022 (D2 = 4, through (R, z) = (8, z)) and a Kepler potential (the
    pipeline's D2 ~ 1e-6, and D2 = 4 on a lane whose bracket is not split
    at the model's vertex). Phi on the axis is deep there, so F' (D2) > 0:
    F > 0 on a thin interval just above D2 (to D2 (1 + 1e-4) for MW2022),
    then a resolved F < 0, then the orbit's interval. lam_min was pinned to
    D2 from the sign of F at D2 (1 + D2_SIDE) alone, 11% of the extent off
    for MW2022 (a leapfrog orbit stays in lambda in [6, 75.5]). A resolved
    F < 0 at a lam_min ladder point below lam_pos now unpins it, and the
    ladder closes its bracket. lam_min matches mpmath to 1e-10, and its
    tangent (forward and reverse) is the implicit-function one: that of the
    same lane with L_z = 1e-8, which never pinned.
    """
    mw = gp.MilkyWayPotential2022()
    kep = gp.KeplerPotential(m_tot=u.Q(1e11, "Msun"), units="galactic")
    w_kep = [10.463750588903896, 0, 0.10574665884686271, -0.14056295339392177]
    w_kep4 = [17.482806289322035, 0, 0.013598888553341138, 0.08041371469391963]
    cases = [  # (pot, w6 without v_phi and v_z, v_z, D2, mpmath lam_min)
        (mw, [8.0, 0, 0.1, 0.05], 0.15, 4.0, 11.3271074460904),
        (mw, [8.0, 0, 0.01, 0.05], 0.15, 4.0, 11.4752444742309),
        (mw, [8.0, 0, 0.001, 0.05], 0.15, 4.0, 11.4904091184200),
        (kep, w_kep, -0.1302914689383773, 1.0950125874264389e-06, 5.35567957519090),
        (kep, w_kep4, 0.20057040957663766, 4.0, 221.192273715420),
    ]
    for pot, w4, vz, d2, ref in cases:
        w6s = jnp.asarray([[*w4, vy, vz] for vy in (0.0, 1.25e-9)])
        lim = _limits_fn(pot)(w6s, jnp.full(2, d2))
        np.testing.assert_allclose(lim.lam_min, ref, rtol=1e-10, atol=0)
        assert not np.any(lim.lmin_pinned)

        def lmin(w, pot=pot, d2=d2):
            out = _live_limits(pot, w, jnp.asarray(d2))[0].lam_min[None]
            return out, out

        Jf, Jr, _ = _jac_both(lmin, w6s)
        for J in (Jf[:, 0], Jr[:, 0]):
            scale = np.max(np.abs(J))
            np.testing.assert_allclose(J[0], J[1], rtol=1e-6, atol=1e-6 * scale)
            assert np.abs(J[0, 2]) > 1e-3 * scale  # d/dz: not D2's (0 here)


def test_lam_max_unsplit_first_sign_change():
    """lam_max is the first root above lambda where the bracket is unsplit.

    Hernquist near escape with an explicit D2 = 100, about as large as the
    orbit. The model's vertex v = 75.8 is below D2, so the bracket is not
    split, and it was closed only from lam_pos e^{8, 24, 72}: F < 0 on
    [138.9, ~170] lies between lam_pos and the first of them, F > 0 again
    beyond, and lam_max came out 2.05e10. The points above lam_pos now run
    from lam (lam + 2 x LADDER^k, x the model's root offset), and the
    first sign change among them closes the bracket: lam_max matches
    mpmath (138.856833198752) to 1e-10.
    """
    pot = gp.HernquistPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
    )
    w6s = jnp.asarray([[4.973335, 0, -0.521347, 0.05219, -0.112326, -0.336409]])
    lim = _limits_fn(pot)(w6s, jnp.full(1, 100.0))
    np.testing.assert_allclose(lim.lam_max, 138.856833198752, rtol=1e-10, atol=0)


def test_limits_past_an_f_negative_window_near_lambda():
    """A thin F < 0 window next to the root nearest lambda is sampled.

    Explicit Delta about as large as the orbit, so F has a second F > 0
    interval past the orbit's own: below it for lam_min, above it for
    lam_max. The ladders sampled only the model's reflected roots and then
    jumped a decade, landed in the second interval, and the solver returned
    one of its roots. One point past the model's near root r, by half of
    |r - lam|, now sits in the window. The MW2022 lane is unsplit with
    L_z != 0, and its closed ladder bracket was dropped for [D2, lam_pos],
    which holds 3 roots (lam_min 100.07). mpmath references (40 digits).
    """
    hern = gp.HernquistPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
    )
    mn = gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), b=u.Q(0.3, "kpc"), units="galactic"
    )
    cases = [  # (pot, w6, D2, limit, mpmath value)
        (
            gp.MilkyWayPotential2022(),
            [4.075266578863329, 0, 0.8240321068264767, -0.024817494243921958,
             -0.01251758738776556, -0.6110165987399514],
            100.0, "lam_min", 112.7346585,
        ),
        (
            hern,
            [18.894869675577276, 0, -4.823990626536554, 0.05802367801518065,
             0.00038785863377274217, 0.1438111671754796],
            100.0, "lam_min", 263.7846028720386,
        ),
        (
            hern,
            [18.65628895362441, 0, 4.494568711533599, -0.0339735542814824,
             0.0563704288427556, 0.19232527093290414],
            1000.0, "lam_max", 1389.123359763941,
        ),
        (
            mn,
            [19.122272660950564, 0, -1.7741339472406503, 0.02570697565065356,
             -0.02705187527699842, -0.2102831850711349],
            1000.0, "lam_max", 1411.418848043493,
        ),
    ]  # fmt: skip
    for pot, w6, d2, key, ref in cases:
        lim = _limits_fn(pot)(jnp.asarray([w6]), jnp.full(1, d2))
        np.testing.assert_allclose(getattr(lim, key)[0], ref, rtol=1e-9, atol=0)


def test_nu_max_with_a_central_maximum():
    """Where F_nu(0) <= 0 (Phi with a central maximum), nu_max is still solved.

    A Plummer minus a smaller Plummer, D2 = 4. The orbit cannot reach z = 0,
    so nu has a lower turning point above 0, but its upper one is F's first
    root above |nu| (mpmath), or D2 at L_z = 0 with F > 0 up to D2. These
    lanes returned |nu| (1.80 and 0.444). The second is at rest at its
    lower nu turning point. L_z = 0 and 1e-12.
    """

    def plum(m, b):
        return gp.PlummerPotential(
            m_tot=u.Q(m, "Msun"), r_s=u.Q(b, "kpc"), units="galactic"
        )

    pot = gp.CompositePotential(a=plum(1e11, 5.0), b=plum(-6e10, 1.0))
    w0 = [
        [5.5896985901513954e-08, 0, -1.3422723627667397,
         -0.026867858810200755, 0, -0.26718606210411033],
        [1.9447909322026145, 0, 0.9572308227454791, 0, 0, 0],
    ]  # fmt: skip
    w = np.array(w0 + w0)
    w[2:, 4] = 1e-12 / w[2:, 0]
    ref = [3.9891496145906316, 4.0, 3.989149609796149, 4.0]
    for mode, out in _modes(_phi(pot), 4.0, jnp.asarray(w), "nu_max").items():
        np.testing.assert_allclose(out, ref, rtol=1e-12, err_msg=mode)


def _phi_two_masses(R, z):
    """Two Plummer masses on the z axis at z = +-1 (G = 1)."""
    up = -1.0 / jnp.sqrt(R * R + (z - 1.0) ** 2 + 0.09)
    return up - 1.0 / jnp.sqrt(R * R + (z + 1.0) ** 2 + 0.09)


def _phi_prolate_plummer(R, z):
    """Return a prolate Plummer potential, q = 2 (G = 1)."""
    return -1.0 / jnp.sqrt(R * R + (z / 2) ** 2 + 1)


_RA = [3.883080032266311, 0, 5.20671047239665]


@pytest.mark.parametrize(
    ("phi", "d2", "w6", "ref"),
    [
        (_phi_ring_axis, 4.0, [*_RA, 0, 0.0077200440504543635, 0], 3.998445207170379),
        (
            _phi_ring_axis,
            4.0,
            [*_RA, 0.03, 0.0077200440504543635, -0.02],
            3.998478868394697,
        ),
        (_phi_ring_axis, 4.0, [*_RA, 0.03, 0.1, 0.05], 3.721688838882465),
        (_phi_ring_axis, 4.0, [*_RA, 0.03, 0, 0.05], 4.0),
        (
            _phi_two_masses,
            1.0,
            [
                0.12609821309054342,
                0,
                -0.038952617904076316,
                0.14483789129858154,
                0.10654971950305686,
                0.04882263017348399,
            ],
            0.9959697810408036,
        ),
        (
            _phi_two_masses,
            1.0,
            [
                0.6839536842853515,
                0,
                0.04630171496729963,
                -0.07586557605376086,
                0,
                -0.04049597461043365,
            ],
            1.0,
        ),
        (
            _phi_prolate_plummer,
            0.25,
            [2.1083042079120875, 0, -0.02778348161741031, 0, 0.380501488423569, 0],
            0.14839987944960248,
        ),
    ],  # fmt: skip
    ids=["ring_rest", "ring", "ring_Lz", "ring_lz0", "masses", "masses_lz0", "prolate"],
)
def test_nu_max_orbit_off_the_plane(phi, d2, w6, ref):
    """F_nu(0) <= 0 (the orbit does not reach z = 0): nu_max is still solved.

    nu_max was |nu| on every lane with F_nu(0) <= 0 (audit 14), e.g. 2.48
    against 3.998 on the first (at rest) and 1.5e-3 against 0.996 on
    "masses". The reference is F's first root above |nu| (mpmath), or D2 at
    L_z = 0 with F > 0 up to D2. The lane and its mirror (z, v_z -> -z,
    -v_z; Phi is even in z) in a batch, under jit and in eager mode.
    """
    w6s = jnp.asarray([w6, np.multiply(w6, [1, 1, -1, 1, 1, -1])], dtype=float)
    for mode, out in _modes(phi, d2, w6s, "nu_max").items():
        np.testing.assert_allclose(out, ref, rtol=1e-12, err_msg=mode)


@pytest.mark.parametrize(
    ("d2", "w6", "ref"),
    [
        (
            0.25,
            [
                1.1869683036367835,
                0,
                -0.21032637199236656,
                0.09337479171778351,
                0.0012108692288520333,
                0.04639114505105796,
            ],
            0.007214840236538335,
        ),
        (
            0.25,
            [
                7.676036953159323,
                0,
                0.015662184617673137,
                -0.03605977759283699,
                0,
                0.05593927305756374,
            ],
            0.006973595413511447,
        ),
        (
            1.0,
            [
                4.320600693905262,
                0,
                0.34847232439432957,
                0.11054666611120789,
                0,
                0.07750959941448504,
            ],
            0.02184684351381971,
        ),
        (
            0.25,
            [
                1.7467593482373056,
                0,
                -0.0011957049546290665,
                0.19873806007815203,
                0,
                0.2156288388990217,
            ],
            0.006243894345593461,
        ),
        (
            0.25,
            [
                0.8690913226514466,
                0,
                -0.4722919273820335,
                0.07580410169462314,
                -0.10023974303965753,
                0.07865796120622766,
            ],
            0.06625904278156256,
        ),
        (
            1.0,
            [
                1.7553287209613389,
                0,
                0.02098682800800216,
                0.280574218355157,
                0,
                -0.2827972989883237,
            ],
            0.057453210236599245,
        ),
        (
            1.0,
            [1.4549748254424164, 0, -4.619105499334229, 0, 0.14834531246687996, 0],
            0.9178338505063943,
        ),
    ],  # fmt: skip
    ids=["Lz", "lz0", "lz0_d2_1", "lz0_1e6", "pow_1_4", "pow_3_4", "rest"],
)
def test_nu_max_first_root_below_a_wide_window(d2, w6, ref):
    """nu_max is the first root above |nu|, below a wide F_nu < 0 window.

    Disk plus flattened core. F_nu < 0 from the first root (1.11 |nu|,
    6.7e3 |nu| and 3.6 |nu|) to well below D2, then F > 0 again. The nu
    ladder had no point between 1.1 |nu| and 0.9 nu_upper and closed past
    the window: nu_max 0.25, 0.25 (the L_z = 0 pin) and 1.0 against 7.2e-3,
    7.0e-3 and 2.2e-2 (mpmath). On the last lane only the 1e6 |nu| point
    falls in the window: a ladder that capped every multiple lost it. On
    "pow_1_4" (window 1.41 to 5.3 |nu|) and "pow_3_4" (542 to 7.7e3 |nu|)
    the window holds only the r^(1/4) or the r^(3/4) point. "rest" is at
    rest (sigma dot = 0) with its root at 1.013 |nu|, and the r^1 point
    must stay just below the ladder's first point, not at |nu|. The lane
    and its mirror in z, in a batch, under jit and in eager mode.
    """
    w6s = jnp.asarray([w6, np.multiply(w6, [1, 1, -1, 1, 1, -1])], dtype=float)
    for mode, out in _modes(_phi_ring_axis, d2, w6s, "nu_max").items():
        np.testing.assert_allclose(out, ref, rtol=1e-12, err_msg=mode)


def test_axis_cusp_centre_window_above_D2():
    """At a cusp centre on the axis (R = 0, small Delta) lam_max is D2.

    F_lam ~ -c D2 (tau - D2)^(3/2) + v_z^2 (tau - D2)^2 / 2 there: F < 0
    on (D2, D2 + x*), so the first root above D2 is D2 itself. With x*
    below D2 MINIMUM_RANGE, F'' at D2 (1 + MINIMUM_RANGE) is > 0 and the
    lane was solved past the window, decades out (NFW: 46571 at D2 =
    1e-8). A resolved F < 0 at D2 (1 + D2_SIDE) or at a lam_min ladder
    point below lam_pos now keeps lam_max at D2 (a shell). The audit
    repros (R = z = 0, v_R = 0) and lanes just off the origin with a small
    v_R; references from a first-crossing mpmath scan. lam_max moves with
    D2 alone: jacfwd = jacrev = (0, ..., 0, 1) in (w6, D2).
    """
    hern = gp.HernquistPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
    )
    cases = [  # (pot, w6, D2)
        (_nfw(), [0, 0, 0, 0, 0, 0.6970209698127537], 1e-8),
        (_nfw(), [0, 0, 0, 0, 0, 0.3872338721181965], 1e-10),
        (hern, [0, 0, 0, 0, 0, 0.6036378668283111], 1e-10),
        (gp.MilkyWayPotential2022(), [0, 0, 0, 0, 0, 0.7845619420066827], 1e-12),
        (_nfw(), [1e-16, 0, 1e-10, 7.744677442351023e-10, 0, 0.46468064654106134],
         1e-8),
        (hern, [1e-17, 0, 1e-11, 6.707087409186689e-10, 0, 0.603637866826802],
         1e-10),
        (gp.MilkyWayPotential2022(),
         [1e-18, 0, 1e-12, 8.71735491116532e-10, 0, 0.7845619420048787], 1e-12),
    ]  # fmt: skip
    for pot, w6, d2 in cases:
        lim = _limits_fn(pot)(jnp.asarray([w6], dtype=float), jnp.full(1, d2))
        assert float(lim.lam_max[0]) == float(lim.lam_min[0]) == d2
        assert lim.shell[0]

    def lmax(w7, pot):
        out = _live_limits(pot, w7[:6], w7[6])[0].lam_max[None]
        return out, out

    for pot, w6, d2 in cases[:2]:
        w7 = jnp.asarray([[*w6, d2]], dtype=float)
        Jf, Jr, _ = _jac_both(lambda w, pot=pot: lmax(w, pot), w7)
        for J in (Jf, Jr):
            np.testing.assert_array_equal(J[0, 0], [0, 0, 0, 0, 0, 0, 1])


def test_axis_window_tiny_Lz_lam_at_D2():
    """With a tiny L_z and lam rounded to D2, lam_max is F's first root above lam.

    NFW, D2 = 1e-8, R = 1e-13 and v_z = 0.58, so lam - D2 = R^2 is far
    below an ulp of D2. Under mpmath F < 0 is resolved at D2 (1 + 1e-12)
    for v_phi up to 1e-6 (L_z up to 1e-19). For v_phi = 0, 1e-9 and 1e-7
    F is also negative from D2 (1 + 1e-16) up, so lam_max is D2. For
    v_phi = 1e-6 F > 0 is resolved from D2 (1 + 1e-17) to D2 (1 + 1e-13),
    and lam_max is the root at D2 (1 + 3.0549627e-13). The window test
    was gated on L_z = 0, and lam_max was solved past the window to 2558
    on all three L_z != 0 lanes. With v_phi = 1e-5 F has no window and
    lam_max stays 2558. Checked in a batch, single-lane jit and eager.
    Where lam_max is D2 it moves with D2 alone: jacfwd = jacrev = (0,
    ..., 0, 1) in (w6, D2).
    """
    pot, d2 = _nfw(), 1e-8
    w6s = jnp.asarray(
        [[1.0000000000000002e-13, 0, 0, 0, vp, 0.5808508081772947]
         for vp in (0.0, 1e-9, 1e-7, 1e-6, 1e-5)]
    )  # fmt: skip
    D2s = jnp.full(len(w6s), d2)
    phi = _phi(pot)

    def one(w6, d):
        iom = integrals_of_motion(phi, w6, d)
        return find_limits(make_F(phi, iom), iom, default_phi_inf(phi, iom))

    jone = jax.jit(one)
    batch = _limits_fn(pot)(w6s, D2s)
    for mode, out in (
        ("batch", batch.lam_max),
        ("jit", [jone(w, d).lam_max for w, d in zip(w6s, D2s, strict=True)]),
        ("eager", [one(w, d).lam_max for w, d in zip(w6s, D2s, strict=True)]),
    ):
        lm = np.ravel(out)
        np.testing.assert_array_equal(lm[:3], d2, err_msg=mode)
        ref = d2 + d2 * 3.0549627e-13
        assert abs(lm[3] - ref) <= 2 * np.spacing(d2), (mode, lm[3])
        np.testing.assert_allclose(lm[4], 2558.1225031437, rtol=1e-12, err_msg=mode)
    np.testing.assert_array_equal(batch.lam_min, d2)

    def lmax(w7):
        out = _live_limits(pot, w7[:6], w7[6])[0].lam_max[None]
        return out, out

    w7 = jnp.concat([w6s[:3], D2s[:3, None]], axis=1)
    Jf, Jr, _ = _jac_both(lmax, w7)
    for J in (Jf, Jr):
        np.testing.assert_array_equal(J[:, 0], np.eye(7)[[6] * 3])


def test_lam_near_turning_point_modes_agree():
    """Two lanes where F(lam) and F(lam_pos) are roundoff agree across modes.

    MN disk + ring + Plummer bulge, D2 = 4, both lanes at z = 0 with a tiny
    v_R. F(lam) = (lam - |nu|)^2 lamdot^2 / (8 lam) from the momenta is
    exact, while the sampled F(lam) and F(lam_pos) are within F's
    roundoff, and their signs changed with the compile.
    - R = 10.04: lam is 1.2e-9 above lam_min, with a resolved F < 0 window
      below it. The tiny-step edge test read the sampled F(lam): 0 in
      eager (edge) and 6.6e-14 under jit (not edge). Off the edge the
      lam_min ladder skipped the window and lam_min was 65.78 under jit.
    - R = 4.16: F has a minimum of 3.3e-15 near lam, with no root there.
      A roundoff F(lam_pos) < 0 in a 16-lane batch left the lane without
      an F > 0 start, and it was snapped to lam (21.30).
    The edge test now uses the exact F(lam), and F(lam_pos) is the model
    value from it where the two agree to F's roundoff. References from
    mpmath roots of F.
    """
    G = 4.498502151469554e-12

    def phi(R, z):
        disk = -G * 5e10 / jnp.sqrt(R * R + (3.0 + jnp.sqrt(z * z + 0.3 * 0.3)) ** 2)
        ring = (
            -G
            * 3e10
            / jnp.sqrt((jnp.sqrt(R * R + 0.5 * 0.5) - 8.0) ** 2 + z * z + 1.0 * 1.0)
        )
        bulge = -G * 1e10 / jnp.sqrt(R * R + z * z + 0.5 * 0.5)
        return disk + ring + bulge

    w6s = jnp.asarray(
        [[4.1593096487701064, 0, 0, -4.235495604904709e-09, 0.043061244042003385,
          -8.615916431061039e-08],
         [10.040366133232038, 0, 0, 7.760317371643708e-10, 0.5069629456373822,
          -0.00023675317378422987]]
    )  # fmt: skip
    ref = np.asarray(
        [[4.2567029035846589, 103.48409251660775], [104.80895208817217, np.nan]]
    )

    def one(w6, d):
        iom = integrals_of_motion(phi, w6, d)
        lim = find_limits(make_F(phi, iom), iom, default_phi_inf(phi, iom))
        return jnp.stack([lim.lam_min, lim.lam_max])

    d2 = jnp.asarray(4.0)
    jone = jax.jit(one)
    batch = jax.jit(jax.vmap(one))(jnp.tile(w6s, (8, 1)), jnp.full(16, 4.0))
    for mode, out in (
        ("batch", np.asarray(batch)),
        ("jit", np.tile([np.asarray(jone(w, d2)) for w in w6s], (8, 1))),
        ("eager", np.tile([np.asarray(one(w, d2)) for w in w6s], (8, 1))),
    ):
        np.testing.assert_allclose(out, np.tile(ref, (8, 1)), rtol=1e-12, err_msg=mode)


def _units_phi(L, V):
    """Hernquist + MN + NFW in kpc/Myr, for lengths in units of L kpc, speeds V."""
    G = 4.498502151469554e-12

    def phi(R, z):
        R, z = R / L, z / L
        r = jnp.sqrt(R * R + z * z)
        h = -G * 5e10 / (r + 1.0)
        m = -G * 5e10 / jnp.sqrt(R * R + (3.0 + jnp.sqrt(z * z + 0.09)) ** 2)
        x = r / 15.0
        xs = jnp.where(x > 0, x, 1.0)
        n = -G * 1e12 / 15.0 * jnp.where(x > 0, jnp.log1p(xs) / xs, 1.0)
        return (h + m + n) * V * V

    return phi


def test_limits_independent_of_units():
    """The limits do not depend on the length and velocity units.

    The log lanes ran in log tau: Chandrupatla's stop |b - a| <= 4 eps |x|
    is then ~380 ulp of tau in SI (log tau ~ 94), and lam_max's tau =
    lam_pos exp(l) was formed as exp(log lam_pos + l), with l rounded to
    the ulp of log lam_pos. Near the foci (lam_min - D2 ~ 350 ulp of D2)
    lam_min in SI was off by 2.1e-3 of the extent and lam_max by 2.2e-4.
    The lanes now run in log(tau / sqrt(a b)) and tau = lam_pos exp(l).

    One near-foci lane (D2 = 100) in kpc, SI (m, m/s), Gpc and AU against
    mpmath: both lambda limits to 2e-5 of the extent (~2 ulp of D2; they
    are 1 ulp off at most), nu_max to the reference's 10 digits. The same
    analytic potential on 40 mixed lanes: SI, Gpc and AU equal kpc to 2e-5
    of the extent (lambda; measured 8e-6, 1 ulp) and 3e-13 D2 (nu_max;
    measured up to 1.1e-13 D2 on one lane on Linux and Windows CI).
    The lane's lam_min tangent: jacfwd = jacrev in kpc and SI. (SI's is not
    kpc's scaled to 1e-6: lam_min - D2 is known to 1 ulp in ~350 there, and
    F' at the root moves with it.)
    """
    units = [("kpc", 1.0, 1.0), ("SI", 3.0856775814913673e19, 977.792221e3),
             ("Gpc", 1e-6, 1.0), ("AU", 2.0626480624709636e8, 1.0)]  # fmt: skip
    w = np.array([4.188678314424233e-05, 0.0, 0.0035152183957328598,
                  -1.5916141289500283e-05, 1.822234762123215e-05,
                  2.394652289527858e-06])  # fmt: skip
    d2 = 100.0
    mp_ref = (4.991024175e-12, 1.758329793e-9, 1.235676046e-5)  # minus D2, nu
    ext = mp_ref[1] - mp_ref[0]

    rng = np.random.default_rng(2)
    R = rng.uniform(0.01, 15, 40)
    z = rng.uniform(-4, 4, 40)
    v = rng.normal(0, 0.06, (40, 3)) + np.array([0, 0.15, 0])
    R[::5], v[::5, 1] = 0.0, 0.0  # on the axis
    W1 = np.stack([R, 0 * R, z, *v.T], 1)
    W1[1::5, :3] *= 1e-3  # cores
    W1[1::5, 3:] *= 1e-3
    W1 = np.concat([W1, w[None]])
    D1 = np.concat([np.array([4.0, 0.25, 100.0, 1e-4])[np.arange(40) % 4], [d2]])

    def limits_of(phi):
        def one(w6, d):
            iom = integrals_of_motion(phi, w6, d)
            return find_limits(make_F(phi, iom), iom, default_phi_inf(phi, iom))

        return jax.jit(jax.vmap(one))

    res = {}
    for name, L, V in units:
        s = np.array([L, L, L, V, V, V])
        lim = limits_of(_units_phi(L, V))(jnp.asarray(W1 * s), jnp.asarray(D1 * L * L))
        out = np.stack([lim.lam_min, lim.lam_max, lim.nu_max], 1) / (L * L)
        res[name] = out
        got = (out[-1, 0] - d2, out[-1, 1] - d2, out[-1, 2])
        assert abs(got[0] - mp_ref[0]) < 2e-5 * ext, (name, got)
        assert abs(got[1] - mp_ref[1]) < 2e-5 * ext, (name, got)
        np.testing.assert_allclose(got[2], mp_ref[2], rtol=1e-9)
    base = res["kpc"]
    assert np.all(np.isfinite(base))
    ext_all = base[:, 1] - base[:, 0]
    for name, *_ in units[1:]:
        err = np.abs(res[name] - base)
        bad = ~(err[:, :2] <= 2e-5 * ext_all[:, None])
        assert not np.any(bad), (name, np.argwhere(bad), err[bad])
        np.testing.assert_array_less(err[:, 2], 3e-13 * D1)
        # nu_max in ulp of itself, median over the lanes: 14 to 40 with the
        # scaled nu lane, 84 in SI in log tau (30 Gpc, 43 AU). The low nu
        # ladder points that replace unused multiples of |nu| change the
        # bracket, and with it the last ulp, on lanes with nu_max well
        # below D2: 69 in SI (converged; 40 iterations give the same). The
        # largest error stays 2.4e-14 D2.
        nu_ulp = err[:, 2] / np.spacing(np.abs(base[:, 2]) + 1e-300)
        assert np.median(nu_ulp) <= 80, (name, np.median(nu_ulp))

    def lmin_fn(L, V):
        phi = _units_phi(L, V)

        def f(w6):
            iom = integrals_of_motion(phi, w6, jnp.asarray(d2 * L * L))
            iomz = jax.lax.stop_gradient(iom)
            F, Fz = make_F(phi, iom), make_F(phi, iomz)
            lim = find_limits(Fz, iomz, default_phi_inf(phi, iomz))
            out = attach_tangents(lim, F, Fz).lam_min[None]
            return out, out

        return f

    for _, L, V in units[:2]:
        s = np.array([L, L, L, V, V, V])
        _jac_both(lmin_fn(L, V), jnp.asarray([w * s]))  # asserts jacfwd = jacrev


def _phi_hole(R, z):
    """Plummer sphere minus a small dense one: Phi rises to the centre."""
    r2 = R * R + z * z
    return -1.0 / jnp.sqrt(r2 + 4.0) + 0.3 / jnp.sqrt(r2 + 0.09)


def _phi_caps(R, z):
    """Plummer sphere plus two Plummer caps on the z axis at z = +-1.5."""
    zc = jnp.sqrt(z * z + 1e-6) - 1.5
    return -1.0 / jnp.sqrt(R * R + z * z + 1.0) - 0.5 / jnp.sqrt(R * R + zc**2 + 0.01)


def _phi_z3(R, z):
    """Core plus two heavier masses on the z axis at z = +-2."""
    p = lambda m, z0: -m / jnp.sqrt(R * R + (z - z0) ** 2 + 0.04)
    return p(1.0, 0.0) + p(2.0, 2.0) + p(2.0, -2.0)


def _phi_tori(R, z):
    """Two rings at R = 1, z = +-0.5, and a wide Plummer sphere."""
    p = lambda z0: -1.0 / jnp.sqrt((R - 1) ** 2 + (z - z0) ** 2 + 0.1)
    return p(0.5) + p(-0.5) - 0.2 / jnp.sqrt(R * R + z * z + 4.0)


def _phi_ring(R, z):
    """Ring at R = 2 in the plane plus a weak core."""
    ring = -1.0 / jnp.sqrt((R - 2) ** 2 + z * z + 0.25)
    return ring - 0.1 / jnp.sqrt(R * R + z * z + 0.01)


def _phi_ring_cusp(R, z):
    """MN disk, a ring with a cusp in R at R = 8 kpc, and a Plummer bulge."""
    G = 4.498502151469554e-12
    disk = -G * 5e10 / jnp.sqrt(R * R + (3.0 + jnp.sqrt(z * z + 0.09)) ** 2)
    ring = -G * 3e10 / jnp.sqrt((R - 8.0) ** 2 + z * z + 1.0)
    return disk + ring - G * 1e10 / jnp.sqrt(R * R + z * z + 0.25)


@pytest.mark.parametrize(
    ("phi", "d2", "w6", "ref"),
    [
        (_phi_hole, 1e-8, [1.7036517434375906, 0, 0.03167294523075253, 0, 0, 0], 1e-8),
        (_phi_hole, 1e-8, [0.2708489085430357, 0, 0.034352492118084825, 0, 0, 0], 1e-8),
        (
            _phi_hole,
            1e-5,
            [0.0033332872855137744, 0, 0.2268301293320112, 0, 0, 0],
            1e-5,
        ),
        (_phi_caps, 1.0, [0.03770074127017039, 0, 10.0, 0, 0, 0], 1.0),
        (
            _phi_hole,
            5.1536750210142465e-05,
            [0.0020185501976153187, 0, -0.524102544895999, 0, 0, 0],
            5.1536750210142465e-05,
        ),
    ],
    ids=["hole_a", "hole_b", "hole_axis", "caps", "hole_tp_neg"],
)
def test_sigdot0_lower_turning_point_from_the_ladder(phi, d2, w6, ref):
    """At rest at L_z = 0, a ladder point decides a lower nu turning point.

    F(t_tp) is within its roundoff (t_tp is 1e-8 (D2 - |nu|) above |nu|),
    but F at the lowest nu ladder point where it is resolved is positive:
    |nu| is the lower turning point and nu_max is pinned to D2 (mpmath: F
    > 0 from |nu| up to D2). nu_max was |nu|: 3.5e-12, 1.6e-10, 9.9978e-6
    and 0.999986. On "hole_tp_neg" F(t_tp) reads as a resolved negative
    (mpmath: +1.0e-27; Phi's own terms cancel in the hole), and the ladder
    decides there too. In a batch, under jit and in eager mode.
    """
    w6s = jnp.asarray([w6], dtype=float)
    for mode, out in _modes(phi, d2, w6s, "nu_max").items():
        np.testing.assert_allclose(out[0], ref, rtol=1e-12, err_msg=mode)


@pytest.mark.parametrize(
    ("phi", "d2", "w6", "ref"),
    [
        (
            _phi_ring_axis,
            0.25,
            [
                1.1869683036367835,
                0,
                0.0,
                0.09337479171778351,
                0.0012108692288520333,
                0.04639114505105796,
            ],
            6.977657918354794e-05,
        ),
        (
            _phi_ring_axis,
            1.0,
            [4.320600693905262, 0, 0.0, 0.11054666611120789, 0, 0.07750959941448504],
            0.0061042437979218316,
        ),
        (
            _phi_ring_axis,
            0.25,
            [7.676036953159323, 0, 0.0, -0.03605977759283699, 0, 0.05593927305756374],
            0.006757723491600656,
        ),
        (
            _phi_z3,
            1.0,
            [
                0.5324943898986754,
                0,
                0.0,
                0.05539382331785269,
                0.030644766266727838,
                -0.5575650908617041,
            ],
            0.09428234770490002,
        ),
        (
            _phi_z3,
            1.0,
            [
                0.5324943898986754,
                0,
                -0.0035,
                0.05539382331785269,
                0.030644766266727838,
                -0.5575650908617041,
            ],
            0.09426151260668054,
        ),
        (
            _phi_ring_axis,
            0.0004910480828476595,
            [3.3617198705627174, 0, 0.0, 0.08242376999388613, 0, 0.15592888162796129],
            4.203709176680851e-05,
        ),
    ],
    ids=["dd_a", "dd_b", "dd_lz0", "z3", "z3_off_plane", "dd_mid"],
)
def test_nu_max_first_root_from_the_plane(phi, d2, w6, ref):
    """nu_max is the first root above |nu| for a start at or near z = 0.

    At z = 0, |nu| is floored at ~1e-20 D2 and every multiple of it in
    the old NU_LADDER_LO fell below nu_tiny, so the ladder had no point between
    ~1e-8 D2 and 0.9 nu_upper and closed past an F_nu < 0 window (2.9e-4
    to 0.86 D2 on the first): nu_max 0.25, 1.0, 0.25 (the L_z = 0 pin)
    and 0.998 against the mpmath roots. Those slots now spread over [nu_tiny,
    0.9 nu_upper] in log tau. "z3_off_plane" (z = -0.0035, |nu| ~ 1e-5 D2)
    had its points at most 5.4e3 |nu|, a factor 17.5 below the ladder,
    and the window (0.095 to 0.79 D2) fell in that gap. "dd_mid" needed the
    point at the old 0.5 fraction (with 0.75 there nu_max was D2). In a batch,
    under jit and in eager mode.
    """
    w6s = jnp.asarray([w6], dtype=float)
    for mode, out in _modes(phi, d2, w6s, "nu_max").items():
        np.testing.assert_allclose(out[0], ref, rtol=1e-10, err_msg=mode)


@pytest.mark.parametrize(
    ("phi", "d2", "w6", "ref"),
    [
        (
            _phi_z3,
            647.7319535900526,
            [0.20237680701349695, 0, 0.0, 0.03502971936004276, 0, 0.05844847693061718],
            8.0496541409231722e-05,
        ),
        (
            _phi_z3,
            238.8633478973877,
            [
                0.9856637522452779,
                0,
                0.0,
                0.0565706790405401,
                0.14115509934877382,
                0.030551671258382374,
            ],
            0.001940844180191316,
        ),
        (
            _phi_z3,
            41.2497964760393,
            [
                1.5820757638279277,
                0,
                0.0,
                0.0006361428685030012,
                0.003122707932938607,
                -0.019753863467588242,
            ],
            0.019966888708486143,
        ),
        (
            _phi_z3,
            300.0,
            [
                0.49042431538549974,
                0,
                -0.003924519013740204,
                0.4440762479517389,
                0.08549407349104579,
                -0.16684979892417498,
            ],
            0.0047904044868971185,
        ),
        (
            _phi_z3,
            3000.0,
            [
                0.7178938961671308,
                0,
                -0.0029529949499714193,
                0.41521429302018337,
                0,
                -0.61472134594804,
            ],
            0.48799109081272564,
        ),
        (
            _phi_z3,
            74.9802735318495,
            [
                0.0071128519392046575,
                0,
                0.36201580601619116,
                -0.5155580189490159,
                0,
                0.6851006587747338,
            ],
            0.17421888450758401,
        ),
        (
            _phi_z3,
            1.118388092713613,
            [
                0.6733326677949062,
                0,
                -7.678810742451194e-05,
                0.00032472699362460844,
                0.02048123995657541,
                -0.30800366570268434,
            ],
            0.066025637699876821,
        ),
        (
            _phi_ring_axis,
            1.9741084200717924,
            [
                11.10238054813454,
                0,
                -0.5436611635086825,
                0.06017119730913215,
                0,
                0.020058065535427258,
            ],
            0.026317625774305301,
        ),
        (
            _phi_ring_axis,
            0.0034964378579946725,
            [
                0.6075926833537346,
                0,
                -1.2007283567266303e-06,
                -0.48888662416359907,
                0.010331857302568179,
                0.26470479025526983,
            ],
            0.00014761178460237233,
        ),
        (
            _phi_ring_axis,
            0.670683707218127,
            [
                3.270928999331523,
                0,
                1.6765123948867012e-05,
                -0.00120538070208931,
                0.047885321428484125,
                -0.1488950047561167,
            ],
            0.024416137670745191,
        ),
    ],
    ids=[
        "z3_a",
        "z3_b",
        "z3_c",
        "z3_F2e",
        "z3_3000",
        "z3_mid",
        "z3_small_z",
        "dd_a",
        "dd_b",
        "dd_c",
    ],
)
def test_nu_max_window_between_spread_points(phi, d2, w6, ref):
    """nu_max is the first root above |nu| where the F < 0 window is decades wide.

    The spread points had gaps of 1.2 to 5 decades in log tau: at z = 0 the
    1e12 |nu| multiple was never below nu_tiny, and at t / |nu| ~ 1e2 to 1e3
    the multiples and the r^p points clustered. These windows (1e-7 to 4e-3
    D2 on "z3_a", 0.5 decade on "dd_a") fell in a gap, and the lane closed on
    the far root: nu_max 5.42, 5.24, 3.48, 5.88, 6.33, 6.50, 1.118, 1.974,
    3.49e-3 and 0.661. The points are now b (t / b)^q, at most 1.1 b and 100 b on
    the first two and at least t 10^-2.5, t 10^-1.3 and t 10^-0.9 on the
    last three (see find_limits). Against mpmath bisection, in a batch,
    under jit and in eager mode.
    """
    w6s = jnp.asarray([w6], dtype=float)
    for mode, out in _modes(phi, d2, w6s, "nu_max").items():
        np.testing.assert_allclose(out[0], ref, rtol=1e-10, err_msg=mode)


@pytest.mark.parametrize(
    ("d2", "w6", "ref"),
    [
        (
            0.12681066388734152,
            [0.8574523451203369, 0, 0.0, 0.3022904282417156, 0, 0.00012452681794100295],
            0.67603578688886419,
        ),
        (
            0.006675352224559938,
            [0.853858725942092, 0, 0.0, 0.34946321378808043, 0.006756204593709586, 0],
            0.47443837804743122,
        ),
        (
            0.1998180082782834,
            [
                0.9400163454144771,
                0,
                0.0003718976666733659,
                0.4745755467627043,
                0,
                0.015098925258557364,
            ],
            0.62124966321214053,
        ),
        (
            0.00040843374664793395,
            [
                0.7617990966245303,
                0,
                -0.24377506808601226,
                0.256751875219363,
                0.017068849694398296,
                -0.232363571119376,
            ],
            0.3511782366539008,
        ),
        (
            0.07693502189694096,
            [
                2.78477096285149,
                0,
                -0.26560597962601246,
                -0.800999646830936,
                0,
                -0.0002560988459677947,
            ],
            0.27765297171020402,
        ),
        (
            0.0037968127434207088,
            [
                0.696534532254808,
                0,
                -0.008574644314932038,
                0.19360089109586875,
                0,
                -7.172460457657375e-05,
            ],
            0.38241384983001522,
        ),
        (
            3.5614845128782164,
            [0.8991441631035181, 0, 0.0, 0.4592410915589253, 0, -0.3069959396818531],
            3.9246968969484642,
        ),
    ],
    ids=[
        "lz0",
        "moving",
        "near_plane",
        "deep",
        "split",
        "fill_third",
        "fill_two_thirds",
    ],
)
def test_lam_min_window_below_lam_pos(d2, w6, ref):
    """lam_min is the first root below lam with F'' > 0 and no model root.

    The ring potential has an F < 0 window between lam and D2. Unsplit, the
    lam_min points were only lam_lower + (lam_pos - lam_lower) 10^-k, and v
    was not one of them, so a sampled F(v) < 0 was ignored ("lz0") and the
    windows of the others fell between lam_pos and the first point: lam_min
    was D2 or a root near it (0.1268, 0.0067, 0.1998, 0.0277). "split" has
    a concave F and the window between the first two points below v (0.0769
    against 0.278). The empty slots below v now hold the points lam_lower +
    (top - lam_lower) 10^-k, k in LAM_LADDER_FILL. "fill_third" needs the
    point at k = 1/3 and "fill_two_thirds" the one at k = 2/3 (without it
    lam_min was D2 on both). Against
    mpmath bisection, in a batch, under jit and in eager mode.
    """
    w6s = jnp.asarray([w6], dtype=float)
    for mode, out in _modes(_phi_ring, d2, w6s, "lam_min").items():
        np.testing.assert_allclose(out[0], ref, rtol=1e-10, err_msg=mode)


@pytest.mark.parametrize(
    ("phi", "d2", "w6", "key", "ref"),
    [
        (
            _phi_tori,
            1000.0,
            [0.8160898319496159, 0, 0.8738676913353984, 0, 0, 0],
            "lam_max",
            1001.3917524580567,
        ),
        (
            _phi_tori,
            3000.0,
            [0.3657530619779185, 0, 0.10627868779457035, 0, 0.7092455371800751, 0],
            "lam_max",
            3003.0687853824134,
        ),
        (
            _phi_tori,
            3000.0,
            [
                0.04523456121991825,
                0,
                -0.07286706643667112,
                -0.3496944420998471,
                0.2033319155862683,
                0.3427286064798803,
            ],
            "lam_max",
            3003.9795858526654,
        ),
        (
            _phi_tori,
            3000.0,
            [
                1.8899218819091204,
                0,
                -0.0043680831184249235,
                0,
                0.0014311428846289927,
                0,
            ],
            "lam_min",
            3000.008374236031,
        ),
        (
            _phi_hole,
            3000.0,
            [0.902486524228502, 0, -0.00983747306125338, 0, 0.004983683356720498, 0],
            "lam_max",
            3018.834849143358,
        ),
        (
            _phi_ring,
            3000.0,
            [
                0.1786959810353936,
                0,
                -0.9756942357957565,
                -0.13288123675629004,
                0,
                0.18605078128644648,
            ],
            "lam_max",
            3012.359892706126,
        ),
        (
            _phi_ring,
            1255.259311293454,
            [
                0.40879234827896105,
                0,
                -0.018052910768443963,
                0.2602501995965288,
                0.0018667947002329005,
                8.104138112350324e-05,
            ],
            "lam_max",
            1265.4688344439753,
        ),
        (
            _phi_ring,
            1952.2986764150523,
            [
                0.48358860034166157,
                0,
                -0.11445922654147635,
                -0.10064089412681076,
                0,
                0.0003862673664694209,
            ],
            "lam_max",
            1962.3186909440421,
        ),
        (
            _phi_tori,
            1311.9275995744204,
            [
                0.002081169028604452,
                0,
                0.0,
                -0.16049780158255825,
                0.03952181440959307,
                0.00018125602203778902,
            ],
            "lam_max",
            1315.8716954207885,
        ),
        (
            _phi_ring_cusp,
            287.42773422900444,
            [
                0.0006413579458437425,
                0,
                1.29481924212247,
                0.08088354724574438,
                5.802073047311886e-05,
                0.07444752750424075,
            ],
            "lam_max",
            288.03286335370996,
        ),
    ],
    ids=[
        "tori_1e3",
        "tori",
        "tori_moving",
        "tori_lam_min",
        "hole",
        "ring",
        "ring_noroot_a",
        "ring_noroot_b",
        "tori_plane",
        "ring_cusp_floor",
    ],
)
def test_lam_limits_converged_large_D2(phi, d2, w6, key, ref):
    """lam_max and lam_min converge at a large D2 with a convex F at lam.

    F'' > 0 at lam and both roots of F's quadratic model are at or below
    lam, so the model gave no step above lam: no points above v, no
    near-root point, and the bracket closed only at lam_pos e^8 with the
    root ~1e-3 above lam_pos. 16 iterations left lam_max 0.3 to 99.9 % of
    the extent off (tori: 3000.13 against 3003.07, mpmath). Those lanes
    now take the 100-fold rise ladder above lam. On "tori_lam_min" (an
    edge lane, F concave with its vertex below D2) every lam_min ladder
    point was below D2: the D2 ladder now runs from lam_pos there
    (3000.0083816 against 3000.0083742). On "ring_noroot_a/b" the model
    has no real root (the step is NaN), and lam_max was 1265.46902 and
    1962.32451 against 1265.46883 and 1962.31869. "tori_plane" pins the
    ladder's scale lam - D2 (with (lam - D2) / 100 it gave 1316.776 against
    1315.872), and "ring_cusp_floor" (lam - D2 = 4e-7 lam / 1e3) its floor
    1e-8 lam (288.033266 against 288.032863 without it). In a batch, under
    jit and in eager mode.
    """
    w6s = jnp.asarray([w6], dtype=float)
    for mode, out in _modes(phi, d2, w6s, key).items():
        np.testing.assert_allclose(out[0], ref, rtol=1e-13, err_msg=mode)
