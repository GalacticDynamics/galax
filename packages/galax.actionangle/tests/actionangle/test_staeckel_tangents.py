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
    _param_jacs,
    _phi,
    _scales,
)
from galax.actionangle._src._staeckel_limits import (
    default_phi_inf,
    find_limits,
)
from galax.actionangle._src._staeckel_prolate import (
    integrals_of_motion,
    make_F,
)


@pytest.fixture(scope="module")
def param_jacs():
    """See `_helpers._param_jacs`."""
    return _param_jacs()


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


@pytest.fixture(scope="module")
def shell_w6s():
    """Planar circular MN orbits: every radius is a lambda shell (snapped)."""
    pot = _mn_p(MN_P)
    phi = _phi(pot)
    R = jnp.arange(2.0, 16.0, 0.25)
    vc = jnp.sqrt(R * jax.vmap(jax.grad(lambda r: phi(r, 0.0 * r)))(R))
    w6s = jnp.stack([R, 0 * R, 0 * R, 0 * R, vc, 0 * R], axis=1)
    lim = jax.jit(jax.vmap(lambda w: _live_limits(pot, w, MN_D2)[0]))(w6s)
    assert np.all(lim.shell & lim.snapped)
    return w6s[:2]


def _shell_lims(theta):
    return _live_limits(_mn_p(theta[6:]), theta[:6], MN_D2)[0]


def _shell_mid(theta):
    lim = _shell_lims(theta)
    return 0.5 * (lim.lam_min + lim.lam_max)


_shell_lims_j = jax.jit(_shell_lims)  # one compile for every FD probe


@pytest.mark.parametrize("jac", [jax.jacfwd, jax.jacrev])
def test_tangent_double_root_via_midpoint(jac, shell_w6s):
    thetas = jnp.concat([shell_w6s, jnp.broadcast_to(MN_P, (2, 3))], axis=1)
    Js = np.asarray(jax.jit(jax.vmap(jac(_shell_mid)))(thetas))
    for w6, theta0, J in zip(shell_w6s, thetas, Js, strict=True):
        _check_double_root(w6, theta0, J)


def _check_double_root(w6, theta0, J):
    assert _shell_lims_j(theta0).shell
    assert np.all(np.isfinite(J))

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
    _assert_jac_close(J, fd, rtol=1e-5, atol_rel=1e-8)
    # the mid-point tangent is the shell's: not the phase point's lambda
    dlam = jax.grad(lambda w: integrals_of_motion(_phi(_mn_p(MN_P)), w, MN_D2).lam)(w6)
    assert np.max(np.abs(J[:6] - np.asarray(dlam))) > 1e-3 * np.max(np.abs(J[:6]))


def test_first_order_only_pinned():
    """The tangents are first-order: the Hessian (w.r.t. v_R) is knowingly off."""
    pot = _mn()
    w6 = jnp.asarray([8.0, 0.0, 0.6, 0.03, 0.2, 0.04])  # y = 0: v_R = v_x

    def lmin(vR):
        lim, _ = _live_limits(pot, w6.at[3].set(vR), MN_D2)
        return lim.lam_min

    vR, h = w6[3], 1e-4 * w6[3]
    g = jax.jit(jax.grad(lmin))
    lmin_j = jax.jit(lmin)
    # the gradient itself is right ...
    fd1 = (lmin_j(vR + h) - lmin_j(vR - h)) / (2 * h)
    np.testing.assert_allclose(g(vR), fd1, rtol=1e-6)
    # ... its derivative is not: the frozen points miss d2F/dL dtheta and dF'
    H = jax.jit(jax.hessian(lmin))(vR)
    H_fd = (g(vR + h) - g(vR - h)) / (2 * h)
    assert np.abs(H - H_fd) > 1e-3 * np.abs(H_fd), (H, H_fd)


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
