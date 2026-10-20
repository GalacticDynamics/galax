"""Tests for the analytic isochrone action-angle core."""

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from ..optional_deps import OptDeps
from galax.dynamics._src.actionangle.isochrone import _isochrone_aaf
from galax.dynamics.actionangle import (
    ActionsAnglesFrequencies,
    isochrone_actions_angles_freqs as iaaf,
)

M, B, G = 2.3, 1.3, 1.0
GM = G * M


def _energy(xyz, v, b=B):
    return 0.5 * jnp.sum(v**2) - GM / (b + jnp.sqrt(b**2 + jnp.sum(xyz**2)))


def _bound_points(n, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3)) * 2.0
    v = rng.normal(size=(n, 3))
    out = []
    for x, vv in zip(xyz, v, strict=True):
        phi = -GM / (B + np.sqrt(B**2 + x @ x))
        vmax = np.sqrt(-2 * phi)
        out.append((x, vv / np.linalg.norm(vv) * rng.uniform(0.2, 0.9) * vmax))
    return out


def test_isochrone_analytic_identities():
    for x_, v_ in _bound_points(200):
        x, v = jnp.asarray(x_), jnp.asarray(v_)
        J, th, Om = _isochrone_aaf(M, B, G, x, v)
        Lvec = jnp.cross(x, v)
        L, Lz = jnp.linalg.norm(Lvec), Lvec[2]
        E = _energy(x, v)
        s = jnp.sqrt(L**2 + 4 * GM * B)
        Om_r = (-2 * E) ** 1.5 / GM
        Om_z = Om_r / 2 * (1 + L / s)
        np.testing.assert_allclose(
            J[0], GM / jnp.sqrt(-2 * E) - 0.5 * (L + s), rtol=1e-12
        )
        np.testing.assert_allclose(J[1], Lz, rtol=1e-12)
        np.testing.assert_allclose(J[2], L - jnp.abs(Lz), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(Om[0], Om_r, rtol=1e-12)
        np.testing.assert_allclose(Om[2], Om_z, rtol=1e-12)
        np.testing.assert_allclose(Om[1], jnp.sign(Lz) * Om_z, rtol=1e-12)
        assert jnp.all((th >= 0) & (th < 2 * jnp.pi))


def test_isochrone_unbound_nan():
    x = jnp.array([1.0, 0.5, 0.2])
    v = jnp.array([3.0, 1.0, 2.0])
    assert _energy(x, v) >= 0
    J, th, Om = _isochrone_aaf(M, B, G, x, v)
    assert jnp.isfinite(J[1])
    assert jnp.isnan(J[0])
    assert jnp.isnan(J[2])
    assert jnp.all(jnp.isnan(th))
    assert jnp.all(jnp.isnan(Om))


def _flat(xyz, v, b, m=M):
    J, th, Om = _isochrone_aaf(m, b, G, xyz, v)
    return jnp.concat([J, jnp.sin(th), jnp.cos(th), Om])


def _jacs_finite(xyz, v, b=B, m=M):
    """Check jacfwd and jacrev of `_flat` w.r.t. (xyz, v, b, m) are finite."""
    args = (jnp.asarray(xyz), jnp.asarray(v), jnp.asarray(b), jnp.asarray(m))
    for jac in (jax.jacfwd, jax.jacrev):
        for j in jac(_flat, argnums=(0, 1, 2, 3))(*args):
            assert jnp.all(jnp.isfinite(j))


def _assert_angles(th, ref, atol):
    """Angles equal modulo 2 pi."""
    th, ref = np.asarray(th), np.asarray(ref)
    np.testing.assert_allclose(np.sin(th), np.sin(ref), atol=atol)
    np.testing.assert_allclose(np.cos(th), np.cos(ref), atol=atol)


@pytest.mark.parametrize("argnum", [0, 1, 2, 3])
@pytest.mark.parametrize("jac", [jax.jacfwd, jax.jacrev])
def test_isochrone_grad_matches_fd(argnum, jac):
    for x, v in _bound_points(10, seed=1):
        args = [jnp.asarray(x), jnp.asarray(v), jnp.asarray(B), jnp.asarray(M)]
        J = jac(_flat, argnums=argnum)(*args)
        assert jnp.all(jnp.isfinite(J))
        a = args[argnum]

        def f(q, a=a, args=args):
            return _flat(
                *[q.reshape(a.shape) if k == argnum else args[k] for k in range(4)]
            )

        cols = []
        for i in range(a.size):
            h = 1e-5 * jnp.abs(a.ravel()[i])
            ap, am = a.ravel().at[i].add(h), a.ravel().at[i].add(-h)
            cols.append((f(ap) - f(am)) / (2 * h))
        fd = jnp.stack(cols, axis=-1).reshape(J.shape)
        np.testing.assert_allclose(J, fd, rtol=1e-5, atol=1e-7)


def test_isochrone_unbound_lanes_nan_free_grads():
    """Unbound lanes are NaN, but their gradients are finite (zero)."""
    x = jnp.array([[1.0, 0.5, 0.2], [1.0, 0.5, 0.2], [0.3, -1.0, 0.4]])
    v = jnp.array([[0.3, 0.6, 0.1], [3.0, 1.0, 2.0], [0.2, 0.1, -9.0]])
    bound = jnp.array([True, False, False])
    assert jnp.array_equal(jax.vmap(_energy)(x, v) < 0, bound)

    def loss(x, v, m, b):
        J, th, Om = jax.vmap(lambda x, v: _isochrone_aaf(m, b, G, x, v))(x, v)
        return jnp.nansum(J[:, ::2]) + jnp.nansum(jnp.sin(th)) + jnp.nansum(Om)

    gx, gv, gm, gb = jax.grad(loss, argnums=(0, 1, 2, 3))(x, v, M, B)
    for g in (gx, gv, gm, gb):
        assert jnp.all(jnp.isfinite(g))
    assert jnp.all(gx[~bound] == 0)
    assert jnp.all(gv[~bound] == 0)
    assert jnp.any(gx[bound] != 0)


def test_isochrone_jit_vmap_no_retrace():
    n = [0]

    def f(xyz, v):
        n[0] += 1
        return _isochrone_aaf(M, B, G, xyz, v)

    g = jax.jit(jax.vmap(f))
    for seed in (2, 3):
        pts = _bound_points(5, seed=seed)
        out = g(jnp.stack([p[0] for p in pts]), jnp.stack([p[1] for p in pts]))
        assert out[0].shape == (5, 3)
    assert n[0] == 1


def _vcirc(r):
    # v^2 = r Phi'(r) = GM r^2 / (s (b + s)^2), s = sqrt(b^2 + r^2)
    s = np.sqrt(B**2 + r**2)
    return np.sqrt(GM * r**2 / (s * (B + s) ** 2))


@pytest.mark.parametrize("r", [0.7, 1.5, 3.0])
@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-9, 1e-6])
def test_isochrone_near_circular(r, eps):
    """Planar (near-)circular orbit at pericentre: theta = (0, phi, pi/2)."""
    vc = _vcirc(r) * (1 + eps)
    x = jnp.array([r * 0.6, r * 0.8, 0.0])
    v = jnp.array([-0.8 * vc, 0.6 * vc, 0.0])
    out = _isochrone_aaf(M, B, G, x, v)
    assert all(jnp.all(jnp.isfinite(o)) for o in out)
    _assert_angles(out[1], [0.0, np.arctan2(0.8, 0.6), np.pi / 2], atol=1e-6)
    _jacs_finite(x, v)


@pytest.mark.parametrize("r", [0.7, 1.5, 3.0])
@pytest.mark.parametrize("inc", [0.0, 0.4, -2.0])
def test_isochrone_exact_circular(r, inc):
    """With x . v = 0, eta = atan2(~0, ~0) falls back to 0 (theta_r = 0).

    Inclined: the point is the ascending (descending for sin(inc) < 0) node,
    so psi = 0 (pi) and theta_phi = node + sgn(Lz) theta_z with A = 0.
    """
    vc = _vcirc(r)
    x = jnp.array([r, 0.0, 0.0])
    v = jnp.array([0.0, vc * np.cos(inc), vc * np.sin(inc)])
    th = _isochrone_aaf(M, B, G, x, v)[1]
    if inc == 0:
        ref = [0.0, 0.0, np.pi / 2]
    else:
        up = np.sin(inc) > 0
        node, psi = (0.0, 0.0) if up else (np.pi, np.pi)
        ref = [0.0, node + np.sign(np.cos(inc)) * psi, psi]
    _assert_angles(th, ref, atol=1e-12)
    _jacs_finite(x, v)


def test_isochrone_angle_below_two_pi():
    """mod(-1e-17, 2 pi) rounds to 2 pi; angles stay in [0, 2 pi)."""
    th = _isochrone_aaf(M, B, G, jnp.array([1.0, 0, 0]), jnp.array([-1e-17, 1.2, 0]))[1]
    assert jnp.all((th >= 0) & (th < 2 * jnp.pi))


def _node_reference(x, v, thz):
    """theta_phi = Omega + sgn(Lz) theta_z, Omega = atan2(Lx, -Ly)."""
    L = np.cross(x, v)
    return np.arctan2(L[0], -L[1]) + np.sign(L[2]) * thz


def test_isochrone_theta_phi_near_z_axis():
    """theta_phi is continuous as R -> 0 (and defined at R = 0)."""
    v = np.array([0.3, 0.2, 0.1])
    by_r = {}
    for R in (1e-4, 1e-8, 1e-11, 1e-12, 0.0):
        x = np.array([0.6 * R, 0.8 * R, 1.5])
        th = np.asarray(_isochrone_aaf(M, B, G, jnp.asarray(x), jnp.asarray(v))[1])
        _assert_angles(th[1], _node_reference(x, v, th[2]), atol=1e-12)
        _jacs_finite(x, v)
        by_r[R] = th
    for R in (1e-11, 1e-12, 0.0):  # theta_r, theta_z continuous
        _assert_angles(by_r[R][::2], by_r[1e-8][::2], atol=1e-7)
    for R in (1e-11, 1e-12):  # Lz = -0.12 R < 0: continuous
        _assert_angles(by_r[R][1], by_r[1e-8][1], atol=1e-7)
    # R = 0: Lz = 0, so Omega_phi = 0 and theta_phi is the node longitude alone
    _assert_angles(by_r[0.0][1], np.arctan2(-0.3, -0.45), atol=1e-12)


@pytest.mark.parametrize("sgn", [1, -1])
def test_isochrone_theta_phi_continuous_through_planar(sgn):
    """Tilting a planar orbit by i: theta_r, theta_phi are continuous in i.

    Below i = 1e-10 the orbit counts as planar (psi = pi/2, gala's convention);
    only theta_z, conjugate to J_z ~ L i^2 / 2, changes there.
    """
    x0, v0 = np.array([1.0, 0.7, 0.0]), np.array([-0.3, 0.6 * sgn, 0.0])
    tilts = []
    for i in (1e-6, 1e-9, 1.01e-10, 0.99e-10, 1e-13, 0.0):
        rot = np.array(
            [[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]]
        )
        x, v = jnp.asarray(rot @ x0), jnp.asarray(rot @ v0)
        tilts.append(_isochrone_aaf(M, B, G, x, v)[1])
        _jacs_finite(x, v)
    for th in tilts[1:]:
        _assert_angles(th[:2], tilts[0][:2], atol=1e-9)


def test_isochrone_origin():
    """At the origin with v != 0 the orbit is radial through the centre."""
    J, th, Om = _isochrone_aaf(M, B, G, jnp.zeros(3), jnp.array([0.3, 0.2, 0.1]))
    assert all(jnp.all(jnp.isfinite(o)) for o in (J, th, Om))
    _assert_angles(th, [0.0, 0.0, 0.0], atol=1e-12)  # pericentre r = 0
    _jacs_finite(jnp.zeros(3), jnp.array([0.3, 0.2, 0.1]))
    # origin at rest (c = 0): the angles are undefined (0), grads still finite
    J, th, Om = _isochrone_aaf(M, B, G, jnp.zeros(3), jnp.zeros(3))
    assert jnp.all(th == 0)
    assert all(jnp.all(jnp.isfinite(o)) for o in (J, Om))
    _jacs_finite(jnp.zeros(3), jnp.zeros(3))


@pytest.mark.parametrize("s", [1e-6, 1e-8, 1e-10])
def test_isochrone_deep_core(s):
    """E ~ Phi(0) without cancellation.

    The core is harmonic, so the angles of s (x, v) are independent of s as
    s -> 0.
    """
    x, v = jnp.array([1.0, 0.3, 0.1]), jnp.array([0.2, 1.0, 0.0])
    ref = _isochrone_aaf(M, B, G, 1e-5 * x, 1e-5 * v)[1]
    out = _isochrone_aaf(M, B, G, s * x, s * v)
    assert all(jnp.all(jnp.isfinite(o)) for o in out)
    _assert_angles(out[1], ref, atol=1e-8)
    _jacs_finite(s * x, s * v)
    # at rest off-centre: apocentre of a radial orbit, theta_r = pi
    th = _isochrone_aaf(M, B, G, s * x, jnp.zeros(3))[1]
    _assert_angles(th, [np.pi, 0.0, 0.0], atol=1e-12)
    _jacs_finite(s * x, jnp.zeros(3))


def _thetar_closed_form(x, v):
    """B&T theta_r (numpy, as in gala); finite for e = 1."""
    r = np.linalg.norm(x)
    E = 0.5 * v @ v - GM / (B + np.sqrt(B**2 + r**2))
    c = GM / (-2 * E) - B
    L2 = np.sum(np.cross(x, v) ** 2)
    e = np.sqrt(1 - L2 * (1 + B / c) / GM / c)
    eta = np.arctan2(x @ v / np.sqrt(-2 * E), B + c - np.sqrt(B**2 + r**2))
    return eta - e * c * np.sin(eta) / (c + B)


@pytest.mark.parametrize("vt", [1e-5, 1e-6, 1e-7, 1e-9, 1e-12, 0.0])
def test_isochrone_near_radial(vt):
    """theta_r is defined for (near-)radial orbits; values and grads finite."""
    x = np.array([1.0, 0.0, 0.3])
    v = 0.4 * x + np.array([0.0, vt, 0.0])
    out = _isochrone_aaf(M, B, G, jnp.asarray(x), jnp.asarray(v))
    assert all(jnp.all(jnp.isfinite(o)) for o in out)
    _assert_angles(out[1][0], _thetar_closed_form(x, v), atol=1e-12)
    _assert_angles(out[1][0], 1.336596, atol=1e-6)  # continuous in vt
    if vt == 0:  # exactly radial: the plane is undefined
        assert jnp.all(out[1][1:] == 0)
    _jacs_finite(x, v)


@pytest.mark.parametrize(
    ("x", "v"),
    [
        ([1.0, 0.5, 0.4], [0.1, 0.5, -0.2]),  # generic
        ([1.0, -0.3, 0.6], [0.2, -0.4, 0.3]),  # retrograde, inclined
        ([1.0, 0.0, 0.5], [0.2, 1e-3, 0.5]),  # near-polar, small Lz > 0
    ],
)
def test_isochrone_angles_advance_linearly(x, v):
    """Integrate the orbit; theta(t) - theta(0) must equal Omega t (mod 2 pi)."""
    import diffrax as dfx

    x0, v0 = jnp.array(x), jnp.array(v)

    def rhs(t, y, args):
        pos = y[:3]
        s = jnp.sqrt(B**2 + pos @ pos)
        return jnp.concat([y[3:], -GM * pos / (s * (B + s) ** 2)])

    _, th0, Om = _isochrone_aaf(M, B, G, x0, v0)
    ts = jnp.linspace(0.0, 2 * 2 * jnp.pi / Om[0], 25)
    sol = dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        dfx.Dopri8(),
        0.0,
        ts[-1],
        None,
        jnp.concat([x0, v0]),
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=dfx.PIDController(rtol=1e-13, atol=1e-13),
        max_steps=100_000,
    )
    th = jax.vmap(lambda y: _isochrone_aaf(M, B, G, y[:3], y[3:])[1])(sol.ys)
    dth = th - th0 - ts[:, None] * Om
    np.testing.assert_allclose(jnp.sin(dth), 0, atol=1e-8)
    np.testing.assert_allclose(jnp.cos(dth), 1, atol=1e-8)


# =====================================================================
# Dispatch overloads

POT = gp.IsochronePotential(
    m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
)
Q_ = np.array([[8.0, 0.5, 1.0], [5.0, 2.0, -0.5], [3.0, -4.0, 0.3]])  # kpc
P_ = np.array([[10.0, 150.0, 20.0], [-60.0, 100.0, 30.0], [80.0, 90.0, -20.0]])
P_KPC_MYR = u.ustrip("kpc/Myr", u.Q(P_, "km/s"))


def test_overload_arrays_vs_quantity():
    arr = iaaf(POT, jnp.asarray(Q_), jnp.asarray(P_KPC_MYR), jnp.asarray(0.0))
    assert isinstance(arr, ActionsAnglesFrequencies)
    assert arr.actions.shape == (3, 3)
    assert not isinstance(arr.actions, u.AbstractQuantity)

    res = iaaf(POT, u.Q(Q_, "kpc"), u.Q(P_KPC_MYR, "kpc/Myr"), u.Q(0.0, "Myr"))
    assert res.actions.unit == u.unit("kpc2/Myr")
    assert res.angles.unit == u.unit("rad")
    assert res.freqs.unit == u.unit("rad/Myr")
    np.testing.assert_allclose(res.actions.value, arr.actions, rtol=1e-12)

    # any Quantity input is enough
    mixed = iaaf(POT, u.Q(Q_, "kpc"), jnp.asarray(P_KPC_MYR), jnp.asarray(0.0))
    assert isinstance(mixed.freqs, u.AbstractQuantity)


def test_overload_broadcasts_q_p_t():
    ref = iaaf(POT, jnp.asarray(Q_[:2]), jnp.asarray(P_KPC_MYR[:2]), 0.0)
    # q (2, 3) with p (3,): p broadcasts
    res = iaaf(POT, jnp.asarray(Q_[:2]), jnp.asarray(P_KPC_MYR[0]), 0.0)
    assert res.actions.shape == (2, 3)
    np.testing.assert_allclose(res.actions[0], ref.actions[0], rtol=1e-12)
    # single point with t (2,): q and p broadcast
    res = iaaf(POT, jnp.asarray(Q_[0]), jnp.asarray(P_KPC_MYR[0]), jnp.zeros(2))
    assert res.actions.shape == (2, 3)
    np.testing.assert_allclose(res.actions[1], ref.actions[0], rtol=1e-12)


def test_quantity_input_units_independent():
    a = iaaf(POT, u.Q(Q_ * 1e3, "pc"), u.Q(P_, "km/s"), u.Q(0.0, "Myr"))
    b = iaaf(POT, u.Q(Q_, "kpc"), u.Q(P_KPC_MYR, "kpc/Myr"), u.Q(0.0, "Myr"))
    for x, y in zip(
        (a.actions, a.angles, a.freqs), (b.actions, b.angles, b.freqs), strict=True
    ):
        np.testing.assert_allclose(x.value, y.value, rtol=1e-12)


def test_overload_psc_and_orbit():
    w = gc.PhaseSpaceCoordinate(q=u.Q(Q_, "kpc"), p=u.Q(P_, "km/s"), t=u.Q(0.0, "Myr"))
    res = iaaf(POT, w)
    assert res.actions.shape == (3, 3)
    ref = iaaf(POT, u.Q(Q_, "kpc"), u.Q(P_, "km/s"), u.Q(0.0, "Myr"))
    np.testing.assert_allclose(res.freqs.value, ref.freqs.value, rtol=1e-12)

    orbit = gd.evaluate_orbit(POT, w, u.Q(jnp.linspace(0.0, 100.0, 5), "Myr"))
    ores = iaaf(POT, orbit)
    assert ores.actions.shape == (3, 5, 3)
    # actions are conserved along the orbit
    np.testing.assert_allclose(
        ores.actions.value[:, 0], ores.actions.value[:, -1], rtol=1e-4
    )


def test_api_grads_match_fd():
    """Gradients through the public API and the potential pytree."""
    q, p = jnp.asarray(Q_[0]), jnp.asarray(P_KPC_MYR[0])

    def f(m, q):
        pot = gp.IsochronePotential(
            m_tot=u.Q(m, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
        )
        res = iaaf(pot, q, p, 0.0)
        return jnp.concat([res.actions, jnp.sin(res.angles), res.freqs])

    m = jnp.asarray(1e11)
    for jac in (jax.jacfwd, jax.jacrev):
        jm, jq = jac(f, argnums=(0, 1))(m, q)
        h = 1e-5 * m
        np.testing.assert_allclose(
            jm, (f(m + h, q) - f(m - h, q)) / (2 * h), rtol=1e-5, atol=1e-16
        )
        fd = jnp.stack(
            [
                (f(m, q.at[i].add(1e-5)) - f(m, q.at[i].add(-1e-5))) / 2e-5
                for i in range(3)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(jq, fd, rtol=1e-5, atol=1e-9)


def test_api_jitted_no_retrace():
    from galax.dynamics._src.actionangle.isochrone import _aaf

    iaaf(POT, jnp.asarray(Q_), jnp.asarray(P_KPC_MYR), 0.0)
    n = _aaf._cache_size()
    iaaf(POT, jnp.asarray(Q_) + 0.1, jnp.asarray(P_KPC_MYR) * 0.9, 1.0)
    assert _aaf._cache_size() == n


@pytest.mark.parametrize(
    ("q", "p"), [(np.ones(2), np.ones(3)), (np.ones(3), np.ones((4, 2)))]
)
def test_parse_validates_shape(q, p):
    with pytest.raises(ValueError, match=r"must have shape \(\.\.\., 3\)"):
        iaaf(POT, jnp.asarray(q), jnp.asarray(p), 0.0)


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
def test_matches_gala_isochrone_xv_to_aa():
    import astropy.units as au
    import gala.dynamics as gala_dyn
    import gala.potential as galap
    from gala.dynamics.actionangle import isochrone_xv_to_aa
    from gala.units import galactic

    gpot = galap.IsochronePotential(m=1e11 * au.Msun, b=2 * au.kpc, units=galactic)
    rng = np.random.default_rng(5)
    xyz = rng.normal(size=(60, 3)) * 4.0
    v = rng.normal(size=(60, 3)) * 0.1  # kpc/Myr
    # planar retrograde, inclined retrograde, planar prograde (at phi = 0 and
    # phi != 0), near-radial (theta_r only, below)
    extra_x = [[6.0, 0, 0], [5.0, 1, 0.5], [6.0, 0, 0], [6.0, 1, 0], [6.0, 1, 0]]
    extra_v = [[0, -0.15, 0], [0.01, -0.12, 0.02], [0, 0.15, 0]]
    extra_v += [[0.01, 0.15, 0], [0.01, -0.15, 0]]
    # np.concatenate, not np.concat: the numpy floor (1.26) predates np.concat
    xyz = np.concatenate([xyz, extra_x])
    v = np.concatenate([v, extra_v])
    phi = -float(gpot.G * 1e11) / (2.0 + np.sqrt(4.0 + (xyz**2).sum(-1)))
    bound = 0.5 * (v**2).sum(-1) + phi < 0
    xyz, v = xyz[bound], v[bound]
    # gala forces psi = +pi/2 for |v_theta| <= 1e-10; psi here is continuous,
    # which differs only at a z-turning point below the plane (z < 0).
    # Planar orbits (z = v_z = 0) follow gala's convention and are compared.
    s = (xyz**2).sum(-1) * v[:, 2] - xyz[:, 2] * (xyz * v).sum(-1)
    keep = ~((np.abs(s) < 1e-8) & (xyz[:, 2] < 0))
    xyz, v = xyz[keep], v[keep]
    assert np.sum((xyz[:, 2] == 0) & (v[:, 2] == 0)) == 4  # planar kept

    res = iaaf(POT, jnp.asarray(xyz), jnp.asarray(v), jnp.asarray(0.0))
    pos = gala_dyn.PhaseSpacePosition(xyz.T * au.kpc, v.T * au.kpc / au.Myr)
    with np.errstate(all="ignore"):  # gala divides by 0 for planar orbits
        out = isochrone_xv_to_aa(pos, gpot)
    act = out[0].to_value("kpc2/Myr").T
    ang = out[1].to_value("rad").T
    frq = out[2].to_value("rad/Myr").T
    np.testing.assert_allclose(res.actions, act, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res.freqs, frq, rtol=1e-10)
    np.testing.assert_allclose(np.sin(res.angles), np.sin(ang), atol=1e-9)
    np.testing.assert_allclose(np.cos(res.angles), np.cos(ang), atol=1e-9)


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
@pytest.mark.parametrize("vt", [1e-5, 1e-6, 1e-7, 1e-9, 1e-12, 0.0])
def test_near_radial_matches_gala(vt):
    """Near-radial orbits: all angles match gala (theta_r only when L = 0)."""
    import astropy.units as au
    import gala.dynamics as gala_dyn
    import gala.potential as galap
    from gala.dynamics.actionangle import isochrone_xv_to_aa
    from gala.units import galactic

    gpot = galap.IsochronePotential(m=1e11 * au.Msun, b=2 * au.kpc, units=galactic)
    x = np.array([5.0, 0.0, 1.5])
    v = -0.02 * x + np.array([0.0, vt, 0.0])
    res = iaaf(POT, jnp.asarray(x), jnp.asarray(v), 0.0)
    pos = gala_dyn.PhaseSpacePosition(x[:, None] * au.kpc, v[:, None] * au.kpc / au.Myr)
    with np.errstate(all="ignore"):
        ang = isochrone_xv_to_aa(pos, gpot)[1].to_value("rad")[:, 0]
    _assert_angles(res.angles[0], ang[0], atol=1e-9)
    if vt > 0:
        # the plane (L = x cross v) is conditioned by eps |x| |v| / L
        cond = np.linalg.norm(x) * np.linalg.norm(v) / np.linalg.norm(np.cross(x, v))
        _assert_angles(res.angles[1:], ang[1:], atol=1e-9 + 1e-15 * cond)
