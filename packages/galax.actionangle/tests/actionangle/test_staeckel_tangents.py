"""Tests for the Staeckel-fudge implicit-function tangents of the limits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ._helpers import (
    MN_D2,
    MN_P,
    _assert_jac_close,
    _live_limits,
    _mn,
    _mn_p,
    _phi,
    _scales,
    _vc,
)
from galax.actionangle._src._staeckel_limits import (
    _derivs,
    default_phi_inf,
    find_limits,
    taylor_coeffs,
)
from galax.actionangle._src._staeckel_prolate import (
    integrals_of_motion,
    make_F,
)


def _param_out(theta, *, plain):
    """Limits and Taylor coefficients w.r.t. (w6, MN parameters).

    (lam_min, lam_max, nu_max, c2, c3, c4, F2, F4); with ``plain`` also the
    coefficients from two separate nested-jvp nests (not `taylor_coeffs`'
    fused one). Aux: (the outputs, the Limits).
    """
    lim, F = _live_limits(_mn_p(theta[6:]), theta[:6], MN_D2)
    t = taylor_coeffs(F, lim)
    rows = [lim.lam_min, lim.lam_max, lim.nu_max, t.c2, t.c3, t.c4, t.F2, t.F4]
    if plain:
        m = 0.5 * (lim.lam_min + lim.lam_max)
        _, _, d2, d3, d4 = _derivs(F.F_lam, m, 4)
        _, _, u2, _, u4 = _derivs(F.F_nu_u, jnp.zeros_like(m), 4)
        rows += [d2 / 2, d3 / 6, d4 / 24, u2 / 2, u4 / 24]
    out = jnp.stack(rows)
    return out, (out, lim)


@pytest.fixture(scope="module")
def param_jacs():
    """Compute jacfwd (with the plain nests) and jacrev of `_param_out`, two points.

    A generic point and the lambda edge (z = 0, v_R = 0: one limit is the
    phase point's lambda). One compile per mode serves the tangent and the
    Taylor-autodiff tests; the values come from the jacfwd program.
    """
    w6s = jnp.asarray(
        [
            [8.0, 0.5, 0.6, 0.03, 0.2, 0.04],  # generic
            [3.0, 0.0, 0.0, 0.0, 0.18, 0.05],  # z = 0, v_R = 0: lambda edge
        ]
    )
    thetas = jnp.concat([w6s, jnp.broadcast_to(MN_P, (2, 3))], axis=1)
    fwd = jax.jacfwd(lambda t: _param_out(t, plain=True), has_aux=True)
    rev = jax.jacrev(lambda t: _param_out(t, plain=False), has_aux=True)
    Jf, (vals, lim0) = jax.jit(jax.vmap(fwd))(thetas)
    Jr, _ = jax.jit(jax.vmap(rev))(thetas)
    return w6s, thetas, np.asarray(Jf), np.asarray(Jr), np.asarray(vals), lim0


def test_tangent_simple_fd(param_jacs):
    """Simple-root tangents of all three limits against central differences.

    w.r.t. the phase point and MN's parameters, both modes; at a generic
    point and on the lambda edge (z = 0, v_R = 0: one limit is the phase
    point's lambda).
    """
    w6s, thetas, Jf, Jr, _, lim0 = param_jacs

    def f(theta):
        lim = _live_limits(_mn_p(theta[6:]), theta[:6], MN_D2)[0]
        return jnp.stack([lim.lam_min, lim.lam_max, lim.nu_max])

    assert not np.any(lim0.shell | lim0.planar | lim0.lmin_pinned | lim0.nu_pinned)
    # the edge branch: one lambda limit is the phase point's lambda
    phiz = _phi(_mn_p(MN_P))
    lam = jax.vmap(lambda w: integrals_of_motion(phiz, w, MN_D2).lam)(w6s)
    assert (lim0.lam_min[1] == lam[1]) | (lim0.lam_max[1] == lam[1])

    # tangents leave the values unchanged (two programs: to roundoff)
    def raw(w):
        iomz = integrals_of_motion(phiz, w, MN_D2)
        return find_limits(make_F(phiz, iomz), iomz, default_phi_inf(phiz, iomz))

    raw0 = jax.jit(jax.vmap(raw))(w6s)
    for k in ("lam_min", "lam_max", "nu_max"):
        np.testing.assert_allclose(getattr(lim0, k), getattr(raw0, k), rtol=1e-14)

    # central differences, h_i = 1e-5 max(|theta_i|, scale_i), all at once
    th = np.asarray(thetas)
    h = 1e-5 * np.maximum(np.abs(th), np.stack([_scales(t) for t in th]))
    E = np.eye(9)[None] * h[:, None, :]  # (point, input, theta)
    fj = jax.jit(jax.vmap(f))
    fp = np.asarray(fj(jnp.asarray((th[:, None] + E).reshape(-1, 9)))).reshape(2, 9, 3)
    fm = np.asarray(fj(jnp.asarray((th[:, None] - E).reshape(-1, 9)))).reshape(2, 9, 3)
    fd = np.swapaxes((fp - fm) / (2 * h[:, :, None]), 1, 2)  # (point, limit, input)
    for J in (Jf[:, :3], Jr[:, :3]):
        assert np.all(np.isfinite(J))
        for i in range(2):  # rtol 1e-5; atol for vanishing entries (d/dv_R, edge)
            _assert_jac_close(J[i], fd[i], rtol=1e-5, atol_rel=1e-8)


def _shell_lims(theta):
    return _live_limits(_mn_p(theta[6:]), theta[:6], MN_D2)[0]


def _shell_mid(theta):
    lim = _shell_lims(theta)
    return 0.5 * (lim.lam_min + lim.lam_max)


_shell_lims_j = jax.jit(_shell_lims)  # one compile for every FD probe


def test_tangent_double_root_via_midpoint():
    """Planar circular MN orbits (snapped lambda shells): both modes against FD."""
    R = jnp.asarray([2.0, 2.25])
    w6s = jnp.stack([R, 0 * R, 0 * R, 0 * R, _vc(_phi(_mn_p(MN_P)), R), 0 * R], 1)
    thetas = jnp.concat([w6s, jnp.broadcast_to(MN_P, (2, 3))], axis=1)
    Js = [
        jax.jit(jax.vmap(jac(_shell_mid)))(thetas) for jac in (jax.jacfwd, jax.jacrev)
    ]
    for i, (w6, theta0) in enumerate(zip(w6s, thetas, strict=True)):
        _check_double_root(w6, theta0, [np.asarray(J[i]) for J in Js])


def _check_double_root(w6, theta0, Js):
    lim = _shell_lims_j(theta0)
    assert lim.shell
    assert lim.snapped
    assert np.all(np.isfinite(Js))

    # FD of the solved mid-point. A perturbed orbit may itself be a shell
    # (w/lam below the 1e-7 noise snap), and then its mid-point is the phase
    # point's lam: widen h until neither is.
    lims_j = _shell_lims_j
    th = np.asarray(theta0)
    hs = 1e-4 * np.maximum(np.abs(th), _scales(th))
    fd = np.zeros_like(th)
    for i in range(len(th)):
        for h in hs[i] * np.array([1.0, 10.0, 100.0]):
            e = np.zeros_like(th)
            e[i] = h
            lp, lm = lims_j(th + e), lims_j(th - e)
            fd[i] = (lp.lam_min + lp.lam_max - lm.lam_min - lm.lam_max) / (4 * h)
            if not (lp.shell | lm.shell) or fd[i] == 0:  # 0: symmetric input
                break
        else:
            pytest.fail(f"input {i}: perturbed orbits stay shells")
    dlam = jax.grad(lambda w: integrals_of_motion(_phi(_mn_p(MN_P)), w, MN_D2).lam)(w6)
    for J in Js:
        _assert_jac_close(J, fd, rtol=1e-5, atol_rel=1e-8)
        # the mid-point tangent is the shell's: not the phase point's lambda
        assert np.max(np.abs(J[:6] - np.asarray(dlam))) > 1e-3 * np.max(np.abs(J[:6]))


def test_taylor_coeffs_autodiff_both_modes(param_jacs):
    """taylor_coeffs' fused nests: plain autodiff in both modes.

    w.r.t. the phase point and MN's parameters (F's closed-over tracers):
    values bitwise those of two separate nested-jvp nests; jacfwd and jacrev
    of taylor_coeffs both equal the separate nests' jacfwd. At the generic
    point of `param_jacs` (here, so one worker compiles it).
    """
    _, _, Jf, Jr, vals, _ = param_jacs
    fused, plain = slice(3, 8), slice(8, 13)
    np.testing.assert_array_equal(vals[0, fused], vals[0, plain])
    J_ref = Jf[0, plain]
    for J in (Jf[0, fused], Jr[0, fused]):
        assert np.all(np.isfinite(J))
        assert np.all(np.any(J[:, 6:] != 0, axis=1))  # parameter tangents flow
        _assert_jac_close(J, J_ref, rtol=1e-13, atol_rel=1e-14)


@pytest.mark.parametrize("jac", [jax.jacfwd, jax.jacrev])
def test_pinned_limits_follow_traced_delta(jac):
    """L_z = 0 on MN: a pinned lam_min = D2 and a pinned nu_max = D2 move with D2.

    d(limit)/d(D2) is 1 there (it was 0: the pinned lanes had no tangent);
    every limit's D2 derivative matches central differences.
    """
    pot = _mn()
    w6s = jnp.asarray(
        [
            [5.459, 0.0, 2.144, 0.09, 0.0, -0.065],  # lam_min pinned
            [0.337, 0.0, -2.431, -0.173, 0.0, -0.090],  # nu_max pinned at D2
        ]
    )
    d2s = jnp.full(2, MN_D2)

    def lims(w, d):
        lim, _ = _live_limits(pot, w, d)
        return jnp.stack([lim.lam_min, lim.lam_max, lim.nu_max]), lim

    J, lim = jax.jit(jax.vmap(jac(lims, argnums=1, has_aux=True)))(w6s, d2s)
    assert lim.lmin_pinned[0]
    assert lim.nu_pinned[1]
    assert lim.nu_max[1] == MN_D2
    h = 1e-6
    f = jax.jit(jax.vmap(lambda w, d: lims(w, d)[0]))
    fd = (f(w6s, d2s + h) - f(w6s, d2s - h)) / (2 * h)
    np.testing.assert_allclose(J, fd, rtol=1e-6, atol=1e-8)
    assert J[0, 0] == 1.0
    assert J[1, 2] == 1.0
