"""Tests for the Staeckel-fudge Taylor coefficients of the Q-form regimes."""

import decimal
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import galax.potential as gp
from ._helpers import (
    _GRAD_POTS,
    D2,
    MN_D2,
    _assert_jac_close,
    _D2_pipeline,
    _kk,
    _limits_fn,
    _live_limits,
    _mn,
    _near_circular_w6s,
    _nfw,
    _param_jacs,
    _phi,
    _w6s,
    kk_delta,
    kk_lambda_roots_decimal,
    kk_turning_points,
)
from galax.actionangle._src import _staeckel_limits as limits
from galax.actionangle._src._staeckel_limits import (
    taylor_coeffs,
    taylor_coeffs_batch,
)
from galax.actionangle._src._staeckel_prolate import (
    integrals_of_motion,
    make_F,
)


def _builtin_pots():
    snlm = jnp.zeros((3, 3, 3)).at[0, 0, 0].set(1.0).at[0, 2, 0].set(0.1)
    return {
        "mn": (_mn(), 0.2),
        "burkert": (
            gp.BurkertPotential(
                m=u.Q(1e11, "Msun"), r_s=u.Q(5.0, "kpc"), units="galactic"
            ),
            0.12,
        ),
        "zhao": (
            gp.ZhaoPotential(
                m=u.Q(1e11, "Msun"),
                r_s=u.Q(5.0, "kpc"),
                alpha=1.0,
                beta=4.0,
                gamma=1.0,
                units="galactic",
            ),
            0.12,
        ),
        "scf": (
            gp.SCFPotential(
                m_tot=u.Q(1e11, "Msun"),
                r_s=u.Q(5.0, "kpc"),
                Snlm=snlm,
                Tnlm=jnp.zeros_like(snlm),
                units="galactic",
            ),
            0.12,
        ),
        "mw2022": (gp.MilkyWayPotential2022(), 0.22),
    }


@pytest.mark.parametrize("name", ["mn", "burkert", "zhao", "scf", "mw2022"])
def test_taylor_coeffs_trace_all_builtins(name):
    pot, vc = _builtin_pots()[name]
    w6s = jnp.asarray(
        [[8.0, 0.5, 0.6, 0.15 * vc, 0.9 * vc, 0.2 * vc], [5.0, -1.0, 1e-6, 0, vc, 1e-7]]
    )
    D2s = _D2_pipeline(pot, w6s)

    phi = _phi(pot)

    def live_F(w6, d2):
        return make_F(phi, integrals_of_motion(phi, w6, d2))

    def refs(w6, d2, m):  # F'' at m and m -+ h, F'(0) on the nu branch
        F = live_F(w6, d2)
        h = 1e-3 * (m - d2)
        return F.d2F_lam(jnp.stack([m - h, m, m + h])), F.dF_nu(jnp.asarray(0.0)), h

    # three programs, not one (compile memory): the limits, the coefficients
    # (values: the limits' tangents are not needed) and the references
    lim = _limits_fn(pot)(w6s, D2s)
    tay = jax.jit(jax.vmap(lambda w, d, l: taylor_coeffs(live_F(w, d), l)))(
        w6s, D2s, lim
    )
    d2F, dFnu0, h = jax.jit(jax.vmap(refs))(w6s, D2s, tay.m)
    for k in ("m", "c2", "c3", "c4", "F2", "F4"):
        assert np.all(np.isfinite(getattr(tay, k))), (k, getattr(tay, k))
    np.testing.assert_array_equal(tay.m, 0.5 * (lim.lam_min + lim.lam_max))
    assert not tay.use_n[0]
    assert tay.use_n[1]  # z ~ 1e-6: near-planar
    # the coefficients are the Taylor coefficients of F
    np.testing.assert_allclose(tay.c2, d2F[:, 1] / 2, rtol=1e-10)
    np.testing.assert_allclose(tay.F2, dFnu0, rtol=1e-10)
    c3_fd = (d2F[:, 2] - d2F[:, 0]) / (2 * h) / 6
    c4_fd = (d2F[:, 2] - 2 * d2F[:, 1] + d2F[:, 0]) / h**2 / 24
    np.testing.assert_allclose(tay.c3, c3_fd, rtol=1e-4)
    np.testing.assert_allclose(tay.c4, c4_fd, rtol=1e-3)


def test_taylor_thresholds(monkeypatch):
    """Pin the Taylor thresholds.

    use_l iff w < 1e-2 hi (m inside the branch); use_n iff nu_max < 1e-4 D2
    and |F4| nu_max < 3e-5 |F2| (F2, F4 finite). At a near-planar MN orbit
    (R = 5, z = 1e-6), with nu_max moved across each threshold in turn. The
    moved limits are not F's roots, so the a-posteriori series check (pinned
    by the kink tests) is switched off.
    """
    monkeypatch.setattr(limits, "SERIES_CHECK_N", 1e30)
    monkeypatch.setattr(limits, "SERIES_CHECK_L", 1e30)
    lim, F = _live_limits(_mn(), jnp.asarray([5.0, 0.0, 1e-6, 0.05, 0.2, 1e-7]), MN_D2)
    hi = lim.lam_max
    for r, use_l in ((0.9e-2, True), (1.1e-2, False)):
        tay = taylor_coeffs(F, lim._replace(lam_min=hi * (1 - r)))
        assert bool(tay.use_l) is use_l, r
    tay = taylor_coeffs(F, lim)
    nu_s = 3e-5 * abs(float(tay.F2)) / abs(float(tay.F4))  # the series bound
    assert nu_s < 0.9e-4 * MN_D2  # so it binds first
    for nu, use_n in (
        (0.9 * nu_s, True),
        (1.1 * nu_s, False),
        (1.1e-4 * MN_D2, False),
    ):
        tay = taylor_coeffs(F, lim._replace(nu_max=jnp.asarray(nu)))
        assert bool(tay.use_n) is use_n, nu
    # and the D2 bound, with the series bound out of the way
    monkeypatch.setattr(limits, "NEAR_PLANAR_SERIES", 1e30)
    for r, use_n in ((0.9e-4, True), (1.1e-4, False)):
        tay = taylor_coeffs(F, lim._replace(nu_max=jnp.asarray(r * MN_D2)))
        assert bool(tay.use_n) is use_n, r


def test_near_shell_switch_continuity_mw2022():
    """At the switch (w ~ 1e-2 hi), MW2022 R ~ 1, pipeline Delta: the worst case.

    The quartic and direct Q agree to <= 2e-6 pointwise (the Q-form switch).
    """
    pot = gp.MilkyWayPotential2022()
    phi = _phi(pot)
    R = 1.0
    vc = float(jnp.sqrt(R * jax.grad(lambda r: phi(r, jnp.asarray(0.0)))(R)))

    def solve(vR):
        w6 = jnp.asarray([[R, 0.0, 0.05, vR, vc, 0.01 * vc]])
        return _live_limits(pot, w6[0], _D2_pipeline(pot, w6)[0])

    lim, _ = solve(1e-2 * vc)  # w/hi is linear in v_R: aim just below 1e-2
    r0 = float((lim.lam_max - lim.lam_min) / lim.lam_max)
    lim, F = solve(1e-2 * vc * 0.99e-2 / r0)
    lo, hi = float(lim.lam_min), float(lim.lam_max)
    assert 0.9e-2 < (hi - lo) / hi < 1e-2
    tay = taylor_coeffs(F, lim)
    assert tay.use_l
    w = hi - lo
    tau = 0.5 * (lo + hi) + 0.5 * w * np.cos(np.pi * (np.arange(48) + 0.5) / 48)
    Q_direct = np.asarray(jax.vmap(F.F_lam)(jnp.asarray(tau))) / (
        (tau - lo) * (hi - tau)
    )
    x = tau - float(tay.m)
    Q_quartic = -(tay.c2 + tay.c3 * x + tay.c4 * (x**2 + w**2 / 4))
    np.testing.assert_allclose(Q_quartic, Q_direct, rtol=2e-6, atol=0)


def test_taylor_use_l_off_where_coeffs_nan():
    """M = D2 (R ~ 1e-9 inside the foci): c_k are NaN, so use_l must be off."""
    pot = _mn()
    w6 = jnp.asarray([1e-9, 0.0, 1.0, 0.0, 0.0, 0.02])
    lim, F = _live_limits(pot, w6, MN_D2)
    tay = taylor_coeffs(F, lim)
    assert lim.lam_max - lim.lam_min < 1e-4 * lim.lam_max  # near-shell by w
    assert np.all(np.isnan([tay.c2, tay.c3, tay.c4]))
    assert not tay.use_l


def test_taylor_coeffs_kk_analytic():
    """KK Taylor coefficients (c2..c4, F2, F4) against F's closed form.

    F'' = 2(E + G') + G'' (tau - D2), etc.

    F = (E tau - I + G(tau) - G(nu))(tau - D2) - Lz^2 tau/2, G = GM sqrt(tau+c^2).
    """
    pot = _kk()
    phi = _phi(pot)
    t = u.Q(0.0, "Myr")
    GM = float(pot.constants["G"].value * pot.m_tot(t, ustrip=pot.units["mass"]))
    c2_ = float(pot.c(t, ustrip=pot.units["length"])) ** 2
    for w6 in _w6s(6):
        lim, F = _live_limits(pot, w6, D2)
        if not np.isfinite(lim.lam_max):
            continue
        tay = taylor_coeffs(F, lim)
        iom = integrals_of_motion(phi, w6, D2)
        E, Lz, lam = float(iom.E), float(iom.Lz), float(iom.lam)
        m = float(tay.m)
        s = m + c2_
        G1, G2 = GM / (2 * s**0.5), -GM / (4 * s**1.5)
        G3, G4 = 3 * GM / (8 * s**2.5), -15 * GM / (16 * s**3.5)
        d = m - D2
        # measured <= 3e-15: rtol 3e-14
        np.testing.assert_allclose(tay.c2, (2 * (E + G1) + G2 * d) / 2, rtol=3e-14)
        np.testing.assert_allclose(tay.c3, (3 * G2 + G3 * d) / 6, rtol=3e-14)
        np.testing.assert_allclose(tay.c4, (4 * G3 + G4 * d) / 24, rtol=3e-14)
        # nu branch at u = 0, tau = u^2: F = (E tau + G(tau) - G(lam) - I_nu)
        # (tau - D2) - Lz^2 tau/2, so F2 = F'(0), F4 = F''(0)/2
        g0, g1, g2 = GM * c2_**0.5, GM / (2 * c2_**0.5), -GM / (4 * c2_**1.5)
        A0 = g0 - GM * np.sqrt(lam + c2_) - float(iom.I3 - iom.flam)
        F2 = A0 - (E + g1) * D2 - Lz**2 / 2
        F4 = (2 * (E + g1) - g2 * D2) / 2
        np.testing.assert_allclose(tay.F2, F2, rtol=3e-14)
        np.testing.assert_allclose(tay.F4, F4, rtol=3e-14)


@pytest.mark.parametrize("name", ["nfw", "mw2022"])
def test_use_n_off_where_nu_line_crosses_cusp(name):
    """Near the axis the nu-line runs through the cusp: no near-planar Taylor.

    R = 0 .. 1e-2 at z = 1e-3, and R = 0.1 at z = 2e-3 (nu_max << D2, but not
    << rho^2): F2, F4 there
    are cusp derivatives (F4 up to ~1e103), so use_n is off and the direct
    quadrature is used; a genuine near-planar orbit (R = 5) keeps use_n.
    """
    pot = _GRAD_POTS[name]()
    w6s = [[R, 0.0, 1e-3, 0.01, 0.0, 1e-4] for R in (0.0, 1e-12, 1e-8, 1e-4, 1e-2)]
    # R = 0.1, z = 2e-3: 1e-4 rho^2 < nu_max < 1e-4 rho (pins rho^2, not rho)
    w6s.append([0.1, 0.0, 2e-3, 0.01, 0.0, 1e-4])
    w6s = jnp.asarray([*w6s, [5.0, 0.0, 1e-6, 0.05, 0.2, 1e-7]])

    def one(w6):
        lim, F = _live_limits(pot, w6, D2)
        return lim, taylor_coeffs(F, lim)

    lim, tay = jax.jit(jax.vmap(one))(w6s)
    assert np.all(np.asarray(lim.nu_max)[:-1] < 1e-4 * D2)  # the old gate's "yes"
    np.testing.assert_array_equal(tay.use_n, [False] * 6 + [True])


@pytest.fixture(scope="module")
def param_jacs():
    """See `_helpers._param_jacs`."""
    return _param_jacs()


def test_taylor_coeffs_autodiff_both_modes(param_jacs):
    """taylor_coeffs' fused nests: plain autodiff in both modes.

    w.r.t. the phase point and MN's parameters (F's closed-over tracers):
    values bitwise those of two separate nested-jvp nests; jacfwd and jacrev
    of taylor_coeffs both equal the separate nests' jacfwd. At the generic
    point of `param_jacs`.
    """
    _, _, Jf, Jr, vals, _ = param_jacs
    fused, plain = slice(3, 8), slice(8, 13)
    np.testing.assert_array_equal(vals[0, fused], vals[0, plain])
    J_ref = Jf[0, plain]
    for J in (Jf[0, fused], Jr[0, fused]):
        assert np.all(np.isfinite(J))
        assert np.all(np.any(J[:, 6:] != 0, axis=1))  # parameter tangents flow
        _assert_jac_close(J, J_ref, rtol=1e-13, atol_rel=1e-14)


@pytest.mark.parametrize(("name", "d2"), [("mn", 2.25), ("kk", 8.0)])
def test_use_n_on_smooth_near_axis(name, d2):
    """Smooth potentials keep the near-planar series near the axis.

    R = 3e-3, z = 3e-5 Delta: nu_max is far below 1e-4 rho^2, which the
    old geometric gate required, but the series converges (|F4| nu_max ~
    1e-8 |F2|), and its Q matches the 6th-order series to ~1e-16, where
    the direct quadrature loses ~3e-5 to cancellation.
    """
    pot = {"mn": _mn, "kk": _kk}[name]()
    phi = _phi(pot)
    R = 3e-3
    vc = float(jnp.sqrt(R * jax.grad(lambda r: phi(r, jnp.asarray(0.0)))(R)))
    w6 = jnp.asarray([R, 0.0, 3e-5 * np.sqrt(d2), 0.1 * vc, vc, 0.0])
    lim, F = _live_limits(pot, w6, d2)
    tay = taylor_coeffs(F, lim)
    assert tay.use_n
    hi = float(lim.nu_max)
    d = limits._derivs(F.F_nu_u, jnp.asarray(0.0), 6)
    Fk = [float(d[2 * k]) / math.factorial(2 * k) for k in range(4)]
    tau = hi * np.sin(np.pi * np.linspace(0.05, 0.95, 10) / 2) ** 2
    q6 = -sum(
        Fk[k] * sum(tau**j * hi ** (k - 1 - j) for j in range(k)) for k in (1, 2, 3)
    )
    q_taylor = -(tay.F2 + tay.F4 * (hi + tau))
    np.testing.assert_allclose(q_taylor, q6, rtol=1e-13)


def _flops_per_point(consume, n=4):
    """XLA flops per point of the vmapped value-and-gradient of the MN pipeline."""
    pot = _mn()
    w6s = jnp.asarray(_w6s(n, seed=2))
    d2s = jnp.full(n, MN_D2)

    def one(w6, d2):
        lim, F = _live_limits(pot, w6, d2)
        return lim.lam_min + lim.lam_max + lim.nu_max + consume(F, lim)

    def f(w6s):
        return jnp.nansum(jax.vmap(one)(w6s, d2s))

    cost = jax.jit(jax.value_and_grad(f)).lower(w6s).compile().cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return cost["flops"] / n


def test_taylor_coeffs_fused_flops():
    """The fused Taylor nests stay fused under vmap (value and gradient).

    Read through the `Taylor` fields, the coefficients' nests share their
    work (measured ~46k flops per point in all); summing `_coeffs`' stacked
    output instead falls off a fusion cliff (~114k). A consumer that loses
    the fusion fails the absolute bound.
    """

    def fields_sum(F, lim):
        t = taylor_coeffs(F, lim)
        return t.c2 + t.c3 + t.c4 + t.F2 + t.F4

    fields = _flops_per_point(fields_sum)
    cliff = _flops_per_point(
        lambda F, lim: jnp.sum(limits._coeffs(F, 0.5 * (lim.lam_min + lim.lam_max)))
    )
    assert fields < 60_000, fields
    assert fields < 0.6 * cliff, (fields, cliff)


def _batch_w6s(pot, n=24):
    """Make generic orbits, with near-planar and near-circular (near-shell) ones."""
    phi = _phi(pot)
    w6s = np.array(_w6s(n, seed=4))
    R = w6s[:8, 0]
    vc = np.sqrt(R * np.asarray(jax.vmap(jax.grad(lambda r: phi(r, 0.0 * r)))(R)))
    w6s[:4, 2], w6s[:4, 5] = 1e-6, 1e-7  # near-planar
    w6s[4:8, 1:4] = 0.0  # near-circular: a near shell
    w6s[4:8, 4], w6s[4:8, 3] = vc[4:], 1e-3 * vc[4:]
    return jnp.asarray(w6s)


def _taylor_lane(phi, w6, d2, lim):
    """`taylor_coeffs` as the batch calls it per lane (with ``iom``: polished)."""
    iom = integrals_of_motion(phi, w6, d2)
    return taylor_coeffs(make_F(phi, iom), lim, iom)


def test_taylor_coeffs_batch_matches_per_lane():
    """`taylor_coeffs_batch`: per-lane values and gradients, both branches.

    With room for every candidate in one chunk and without (two chunks),
    under jit and grad: the flags equal `taylor_coeffs`', the
    coefficients and the gradient of a masked sum agree to 1e-14, and the
    non-candidate lanes are NaN. The compacted gradient's scratch memory
    stays near the per-lane program's (measured 1.1x): without the
    per-lane checkpoint the cond kept its branches' residuals (4x). On MN
    only: KK takes the same branches, and each case is 6 compiles (memory).
    """
    pot = _mn()
    phi = _phi(pot)
    w6s = _batch_w6s(pot)
    d2s = jnp.full(len(w6s), MN_D2)

    def per_lane(w6s):
        lim = jax.vmap(lambda w, d: _live_limits(pot, w, d)[0])(w6s, d2s)
        tay = jax.vmap(lambda w, d, l: _taylor_lane(phi, w, d, l))(w6s, d2s, lim)
        return tay

    def batched(w6s, size):
        lim = jax.vmap(lambda w, d: _live_limits(pot, w, d)[0])(w6s, d2s)
        return taylor_coeffs_batch(phi, w6s, d2s, lim, size)

    def used(t):
        cl = jnp.where(t.use_l, t.c2 + 2 * t.c3 + 3 * t.c4, 0.0)
        return jnp.sum(cl + jnp.where(t.use_n, t.F2 + 5 * t.F4, 0.0))

    ref = jax.jit(per_lane)(w6s)
    n_use = int(np.sum(ref.use_l | ref.use_n))
    assert 0 < n_use < len(w6s)

    def grad_exe(f):
        return jax.jit(jax.grad(lambda w: used(f(w)))).lower(w6s).compile()

    def temp(exe):
        return exe.memory_analysis().temp_size_in_bytes

    ref_exe = grad_exe(per_lane)
    g_ref = ref_exe(w6s)
    for size in (n_use, n_use - 1):  # one chunk; two chunks
        tb = jax.jit(lambda w, s=size: batched(w, s))(w6s)
        for k in ("use_l", "use_n"):
            np.testing.assert_array_equal(getattr(tb, k), getattr(ref, k), err_msg=k)
        for k, use in (
            ("c2", "use_l"),
            ("c4", "use_l"),
            ("F2", "use_n"),
            ("F4", "use_n"),
        ):
            u_ = np.asarray(getattr(ref, use))
            np.testing.assert_allclose(
                np.asarray(getattr(tb, k))[u_],
                np.asarray(getattr(ref, k))[u_],
                rtol=1e-14,
            )
        assert np.any(np.isnan(np.asarray(tb.F2)))  # generic lanes: not computed
        exe = grad_exe(lambda w, s=size: batched(w, s))
        if size == n_use:
            assert temp(exe) < 2 * temp(ref_exe), (temp(exe), temp(ref_exe))
        g = exe(w6s)
        np.testing.assert_array_less(
            np.abs(np.asarray(g - g_ref)), 1e-14 * np.max(np.abs(np.asarray(g_ref)))
        )


def test_use_n_off_where_F_has_a_kink():
    """Kuzmin (a razor-thin disk): F_nu has a |u| term at u = 0.

    The nested jvps see d|z|/dz = 0 there, so F2, F4 miss it and the
    convergence gate passed (F2 = +0.046, F - F0 linear and negative in |u|:
    a negative near-planar Q). The a-posteriori check at u = sqrt(nu_max)/2
    turns use_n off; MN at the same point keeps it.
    """
    w6 = jnp.asarray([5.0, 0.0, 1e-4, 0.0, 0.2, 1e-4])
    kuzmin = gp.KuzminPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(3.0, "kpc"), units="galactic"
    )
    for pot, use in ((kuzmin, False), (_mn(), True)):
        lim, F = _live_limits(pot, w6, MN_D2)
        tay = taylor_coeffs(F, lim)
        assert bool(tay.use_n) is use, pot


def _mn_cut():
    """MN + a hard-cutoff NFW (r_t = 8): F'' jumps at r_t."""
    cut = gp.HardCutoffNFWPotential(
        m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), r_t=u.Q(8.0, "kpc"), units="galactic"
    )
    return gp.CompositePotential(disk=_mn(), halo=cut)


def test_use_l_off_where_F_has_a_kink():
    """MN + a hard-cutoff NFW (r_t = 8), near-circular at R = 8 (w/lam ~ 6e-3).

    F'' jumps at r_t, inside the orbit: the quartic Q was 10-29% off, and
    use_l had no convergence check. The a-posteriori check at m -+ w/4 turns
    it off; MN alone keeps it.
    """
    mn = _mn()
    for pot, use in ((_mn_cut(), False), (mn, True)):
        phi = _phi(pot)
        vc = float(
            jnp.sqrt(8.0 * jax.grad(lambda r, f=phi: f(r, jnp.asarray(0.0)))(8.0))
        )
        w6 = jnp.asarray([8.0, 0.0, 0.05, 0.002 * vc, vc, 0.002 * vc])
        lim, F = _live_limits(pot, w6, MN_D2)
        tay = taylor_coeffs(F, lim)
        assert bool(tay.use_l) is use, pot


def test_taylor_coeffs_batch_is_chunked():
    """Chunked compaction: one scan over size-K chunks, a cond per chunk.

    No whole-batch cond: the old compact-or-every-lane cond compiled the
    Taylor nest twice and fell off a cliff past K candidates (1.75x MN, 2.9x
    MW2022 value and gradient). Values: `test_taylor_coeffs_batch_matches_
    per_lane`, with fewer and more candidates than K.
    """
    pot = _mn()
    phi = _phi(pot)
    w6s = _batch_w6s(pot)
    d2s = jnp.full(len(w6s), MN_D2)
    lim = _limits_fn(pot)(w6s, d2s)
    jaxpr = jax.make_jaxpr(lambda w: taylor_coeffs_batch(phi, w, d2s, lim, 4))(w6s)
    top = [e.primitive.name for e in jaxpr.jaxpr.eqns]
    assert "cond" not in top, top
    (scan,) = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "scan"]
    assert "cond" in [e.primitive.name for e in scan.params["jaxpr"].jaxpr.eqns]


@pytest.mark.parametrize("name", ["mn", "kk"])
def test_shell_tangent_on_the_taylor_lanes(name):
    """attach_tangents(shell=False) + taylor_coeffs_batch(shell=True).

    The shell lanes' double-root tangent moves from every lane's tangent
    pass to the compacted Taylor lanes: the batch's limits, m and the
    coefficients have the default path's values and gradients (1e-14; m on
    the shell lanes lacked the shell tangent before the batch returned the
    limits).
    Circular orbits (shells), near-circular and generic lanes. Reverse mode
    on kk; a fixed-direction jvp on mn (half the compile memory for CI).
    """
    pot = {"mn": _mn, "kk": _kk}[name]()
    phi = _phi(pot)
    w6s = jnp.concat([_near_circular_w6s(pot, 0.0)[::40], _batch_w6s(pot)], axis=0)
    d2s = _D2_pipeline(pot, w6s)

    def run(w6s, shell):
        lim = jax.vmap(lambda w, d: _live_limits(pot, w, d, shell=not shell)[0])(
            w6s, d2s
        )
        t = taylor_coeffs_batch(phi, w6s, d2s, lim, shell=shell)
        c = [jnp.where(t.use_l, x, 0.0) for x in (t.c2, t.c3, t.c4)]
        return jnp.stack([t.lam_min, t.lam_max, t.m, *c]), lim.shell

    v = jax.random.normal(jax.random.key(0), w6s.shape)
    vals = {}
    for shell in (False, True):
        out, sh = jax.jit(lambda w, s=shell: run(w, s))(w6s)
        f = lambda w, s=shell: jnp.sum(run(w, s)[0])
        d = jax.grad(f) if name == "kk" else lambda w, f=f: jax.jvp(f, (w,), (v,))[1]
        g = jax.jit(d)(w6s)
        vals[shell] = np.asarray(out), np.asarray(g)
        jax.clear_caches()
    assert sh.sum() >= 4  # shells present
    np.testing.assert_allclose(vals[True][0], vals[False][0], rtol=1e-14, atol=0)
    g0, g1 = vals[False][1], vals[True][1]
    np.testing.assert_array_less(np.abs(g1 - g0), 1e-14 * np.max(np.abs(g0)) + 1e-300)


_POLISH_ES = (1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)  # w/lam ~ 2.4 e
_POLISH_GATE_ES = (4.0e-4, 4.3e-4)  # w/lam just inside and outside the gate


def _polish_case():
    """KK near-circular lanes, two at the polish gate, then generic lanes.

    The last lane is generic: it is the chunks' fill lane. Returns (pot,
    w6s, lam, run), run(w6s, size) -> (Taylor, the input Limits).
    """
    pot = _kk()
    phi = _phi(pot)
    R0 = 6.0
    vc = float(jnp.sqrt(R0 * jax.grad(lambda r: phi(r, 0.0 * r))(R0)))
    es = _POLISH_ES + _POLISH_GATE_ES
    near = [[R0, 0.0, 0.0, e * vc, vc, 0.3 * e * vc] for e in es]
    w6s = jnp.concat([jnp.asarray(near), _w6s(6, seed=4)])
    d2s = jnp.full(len(w6s), kk_delta(pot) ** 2)
    lam = jax.vmap(lambda w, d: integrals_of_motion(phi, w, d).lam)(w6s, d2s)

    def run(w6s, size=None):
        lim = jax.vmap(lambda w, d: _live_limits(pot, w, d)[0])(w6s, d2s)
        return taylor_coeffs_batch(phi, w6s, d2s, lim, size), lim

    return pot, w6s, np.asarray(lam), run


def test_polish_near_circular_roots():
    """The batch Newton-polishes near-circular lambda roots (w < POLISH lam).

    `find_limits` is roundoff-limited there (F's maximum is ~ (w/lam)^2 of
    its terms): ~1e-2 of the extent at w/lam = 2.4e-7 on KK. The batch
    polishes on the Taylor nest's quartic, anchored on F(lam) from the
    momenta, to ~1e-9 of the extent (one ulp of lam is 1e-9 there) against
    KK's exact roots (decimal F). One chunk and several give the same roots
    to roundoff; generic lanes (not candidates) keep the input limits
    bitwise; at the gate the polish moves a root by < 1e-8 of the extent,
    so the limits are continuous across it.
    """
    pot, w6s, lam, run = _polish_case()
    n_near, n_gate = len(_POLISH_ES), len(_POLISH_GATE_ES)
    t1, lim = jax.jit(run)(w6s)  # K = 16 > N: one chunk
    t4, _ = jax.jit(lambda w: run(w, 4))(w6s)
    jax.clear_caches()  # the file's peak memory (CI)
    lo, hi = np.asarray(t1.lam_min), np.asarray(t1.lam_max)
    lo0, hi0 = np.asarray(lim.lam_min), np.asarray(lim.lam_max)
    # the chunks' vmap widths differ: equal to roundoff (measured 1.3e-15)
    np.testing.assert_allclose(np.asarray(t4.lam_min), lo, rtol=4e-15)
    np.testing.assert_allclose(np.asarray(t4.lam_max), hi, rtol=4e-15)

    ref = np.array([kk_turning_points(pot, w, exact=True)[:2] for w in w6s[:n_near]])
    ext = ref[:, 1] - ref[:, 0]
    w_rel = ext / lam[:n_near]
    assert w_rel.max() < 3e-4, w_rel
    assert w_rel.min() < 3e-7, w_rel
    err = np.abs(np.stack([lo[:n_near], hi[:n_near]], 1) - ref) / ext[:, None]
    assert np.all(err < 1e-8), err

    gen = slice(n_near + n_gate, None)
    assert not np.any(np.asarray(t1.use_l | t1.use_n)[gen])  # not candidates
    np.testing.assert_array_equal(lo[gen], lo0[gen])
    np.testing.assert_array_equal(hi[gen], hi0[gen])
    np.testing.assert_array_equal(np.asarray(t1.m)[gen], 0.5 * (lo0 + hi0)[gen])

    gi, go = n_near, n_near + 1
    w_gate = (hi0 - lo0) / lam
    assert w_gate[gi] < limits.POLISH < w_gate[go], w_gate[gi : go + 1]
    for k, (a, a0) in enumerate(((lo, lo0), (hi, hi0))):
        assert 0 < abs(a[gi] - a0[gi]) < 1e-8 * (hi0 - lo0)[gi], k
        assert a[go] == a0[go], k


def test_polish_tangents():
    """Polished roots: tangents at the polished root, generic lanes bitwise.

    A jvp along one fixed direction (no jacrev: CI compile memory) against
    KK's exact root derivative: central differences, h = 1e-15, of the
    roots in 40-digit decimals (`kk_lambda_roots_decimal`), never rounded
    to float. To 1e-7 (measured <= 5e-9) on every near-circular lane; the
    tangent at the unpolished root is off by up to 2e-2 at w/lam = 2.4e-7
    (1e-3 at 2.4e-6). On the generic lanes the batch's limits have the
    input limits' jvp and vjp, bitwise.
    """
    pot, w6s, _, run = _polish_case()
    n_near = len(_POLISH_ES)
    gen = slice(n_near + len(_POLISH_GATE_ES), None)

    def f(w):
        t, lim = run(w)
        return jnp.stack([t.lam_min, t.lam_max, lim.lam_min, lim.lam_max])

    key0, key1 = jax.random.split(jax.random.key(0))
    scale = jnp.concat([jnp.full(3, 6.0), jnp.full(3, 0.2)])
    v = jax.random.normal(key0, w6s.shape) * scale
    out, d = (np.asarray(x) for x in jax.jit(lambda w: jax.jvp(f, (w,), (v,)))(w6s))
    jax.clear_caches()  # the file's peak memory (CI)
    assert np.all(np.isfinite(d)), d
    h = decimal.Decimal("1e-15")
    for i in range(n_near):
        w0, dv = ([decimal.Decimal(float(a)) for a in x[i]] for x in (w6s, v))
        rp, rm = (
            kk_lambda_roots_decimal(
                pot, [a + sg * h * b for a, b in zip(w0, dv, strict=True)]
            )
            for sg in (1, -1)
        )
        ref = [float((p - q) / (2 * h)) for p, q in zip(rp, rm, strict=True)]
        np.testing.assert_allclose(d[:2, i], ref, rtol=1e-7, err_msg=str(i))
    np.testing.assert_array_equal(d[:2, gen], d[2:, gen])

    ct = jnp.zeros_like(out).at[:2, gen].set(jax.random.normal(key1, (2, 6)))

    @jax.jit
    def vjps(w):
        _, back = jax.vjp(f, w)
        return back(ct)[0], back(jnp.roll(ct, 2, axis=0))[0]

    g_batch, g_lim = vjps(w6s)
    jax.clear_caches()  # the file's peak memory (CI)
    np.testing.assert_array_equal(np.asarray(g_batch), np.asarray(g_lim))


# An NFW core lane (Delta^2 = 4, w/lam ~ 1e-10, lam at 0.21 of the extent): a
# Newton step from lam_max crosses lam and converges onto lam_min's side
_NFW_CROSSING_W6 = [
    0.019946933287217975, 0.0, 0.0, -1.1041964811188601e-08,
    0.014108493496219409, 0.002895135797033265,
]  # fmt: skip


@pytest.mark.parametrize("case", ["kink", "weak_kink", "kink_root", "nfw_core"])
def test_polish_keeps_roots_of_F(case):
    """A polished root is kept only where it is a root of F itself.

    The quartic at m cannot see a kink: on `test_use_l_off_where_F_has_a_kink`'s
    potential (r_t = 8 inside a near-circular orbit at R = 8, w/lam ~ 3e-4
    and 3e-6) its root moved lam_min by 12% of the extent (|F|/noise 6e6);
    with a cut halo of 3e6 Msun F' there is the quartic's to <1e-5, but
    |F|/noise is 280. A kink just inside a root, (kink - root) / extent = t =
    -3e-4 and -5e-2 (w/lam 2.7e-4 to 2.7e-7): F is C1 there, |F| at the
    quartic's root stays within roundoff, but its slope is the inner
    piece's (off by ~2.3 |t|): lam_max's tangent was off by up to 0.5, its
    position by up to 4e-2 of the extent (12x the solver's). Nor does the
    quartic converge where w ~ lam - D2, below POLISH lam: NFW's cusp with a
    fixed Delta^2 = 4 at R = 0.02, 0.05 (roots off by up to 2e-3, |F|/noise
    1e9); and a Newton step past lam converged to lam_min's root (clamped to
    lam: the extent fell to 0.21). At the batch's limits |F| is within
    SNAP_NOISE of F's roundoff, and a rejected root is the solver's, bitwise.
    """
    if case == "kink":
        pot, d2, R0s, es, f_phi = _mn_cut(), MN_D2, (8.0, 8.0), (1e-4, 1e-6), 1.0
    elif case == "weak_kink":
        cut = gp.HardCutoffNFWPotential(
            m=u.Q(3e6, "Msun"),
            r_s=u.Q(15.0, "kpc"),
            r_t=u.Q(8.0, "kpc"),
            units="galactic",
        )
        pot = gp.CompositePotential(disk=_mn(), halo=cut)
        d2, R0s, es, f_phi = MN_D2, (8.0,), (3e-4,), 1.0
    elif case == "kink_root":  # R0 from a 60-digit root of the fudge F
        R0s = (7.999431810277854, 7.999994872766397, 7.9999994872767015)
        pot, d2, es, f_phi = _mn_cut(), MN_D2, (1e-4, 1e-6, 1e-7), 1.0
    else:
        pot, d2, R0s, es, f_phi = _nfw(), 4.0, (0.02, 0.05), (0.3, 0.3), 0.9
    phi = _phi(pot)
    vc = [float(jnp.sqrt(R * jax.grad(lambda r: phi(r, 0.0 * r))(R))) for R in R0s]
    w6s = jnp.asarray(
        [
            [R, 0.0, 0.0, e * v, f_phi * v, 0.3 * e * v]
            for R, e, v in zip(R0s, es, vc, strict=True)
        ]
        + ([_NFW_CROSSING_W6] if case == "nfw_core" else [])
    )
    d2s = jnp.full(len(w6s), d2)

    @jax.jit
    def run(w6s):
        lim = jax.vmap(lambda w, d: _live_limits(pot, w, d)[0])(w6s, d2s)
        t = taylor_coeffs_batch(phi, w6s, d2s, lim)

        def resid(w, d, taus):
            iom = integrals_of_motion(phi, w, d)
            f, noise = make_F(phi, iom).F_noise(taus)
            return jnp.abs(f) / noise, iom.lam

        L = jnp.stack([t.lam_min, t.lam_max], 1)
        return lim, L, *jax.vmap(resid)(w6s, d2s, L)

    lim, *out = run(w6s)
    L, r, lam = (np.asarray(a) for a in out)
    jax.clear_caches()  # the file's peak memory (CI)
    L0 = np.stack([np.asarray(lim.lam_min), np.asarray(lim.lam_max)], 1)
    assert np.all((L0[:, 1] - L0[:, 0]) / lam < limits.POLISH), L0  # polish candidates
    assert np.all(r <= limits.SNAP_NOISE), r
    if case in ("kink", "weak_kink"):  # the kink is below m: lam_min is the solver's
        np.testing.assert_array_equal(L[:, 0], L0[:, 0])
        assert np.all(L[:, 1] != L0[:, 1])  # the other side is polished
    elif case == "kink_root":  # the kink is just inside lam_max: the solver's
        np.testing.assert_array_equal(L[:, 1], L0[:, 1])
        assert np.all(L[:, 0] != L0[:, 0])
    else:  # no root of the quartic is a root of F
        np.testing.assert_array_equal(L, L0)
