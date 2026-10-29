"""Tests for the Staeckel-fudge Taylor coefficients of the Q-form regimes."""

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
    _param_jacs,
    _phi,
    _w6s,
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
        tay = jax.vmap(
            lambda w, d, l: taylor_coeffs(
                make_F(phi, integrals_of_motion(phi, w, d)), l
            )
        )(w6s, d2s, lim)
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


def test_use_l_off_where_F_has_a_kink():
    """MN + a hard-cutoff NFW (r_t = 8), near-circular at R = 8 (w/lam ~ 6e-3).

    F'' jumps at r_t, inside the orbit: the quartic Q was 10-29% off, and
    use_l had no convergence check. The a-posteriori check at m -+ w/4 turns
    it off; MN alone keeps it.
    """
    mn = _mn()
    cut = gp.HardCutoffNFWPotential(
        m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), r_t=u.Q(8.0, "kpc"), units="galactic"
    )
    for pot, use in ((gp.CompositePotential(disk=mn, halo=cut), False), (mn, True)):
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
    pass to the compacted Taylor lanes: the limits (plus ``shell_t``) and
    the coefficients have the default path's values and gradients (1e-14).
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
        lo, hi = lim.lam_min + t.shell_t, lim.lam_max + t.shell_t
        c = [jnp.where(t.use_l, x, 0.0) for x in (t.c2, t.c3, t.c4)]
        return jnp.stack([lo, hi, *c]), lim.shell

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
