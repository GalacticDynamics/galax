"""Tests for the analytic isochrone action-angle core."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _flat(xyz, v, b):
    J, th, Om = _isochrone_aaf(M, b, G, xyz, v)
    return jnp.concat([J, jnp.sin(th), jnp.cos(th), Om])


@pytest.mark.parametrize("argnum", [0, 1, 2])
def test_isochrone_grad_matches_fd(argnum):
    for x, v in _bound_points(10, seed=1):
        args = [jnp.asarray(x), jnp.asarray(v), jnp.asarray(B)]
        jac = jax.jacfwd(_flat, argnums=argnum)(*args)
        assert jnp.all(jnp.isfinite(jac))
        a = args[argnum]

        def f(q, a=a, args=args):
            return _flat(
                *[q.reshape(a.shape) if k == argnum else args[k] for k in range(3)]
            )

        cols = []
        for i in range(a.size):
            h = 1e-5 * jnp.abs(a.ravel()[i])
            ap, am = a.ravel().at[i].add(h), a.ravel().at[i].add(-h)
            cols.append((f(ap) - f(am)) / (2 * h))
        fd = jnp.stack(cols, axis=-1).reshape(jac.shape)
        np.testing.assert_allclose(jac, fd, rtol=1e-5, atol=1e-7)


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


@pytest.mark.parametrize("r", [0.7, 1.5, 3.0])
@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-9, 1e-6])
def test_isochrone_near_circular_finite(r, eps):
    # circular speed: v^2 = r Phi'(r) = GM r^2 / (s (b + s)^2), s = sqrt(b^2+r^2)
    s = np.sqrt(B**2 + r**2)
    vc = np.sqrt(GM * r**2 / (s * (B + s) ** 2)) * (1 + eps)
    x = jnp.array([r * 0.6, r * 0.8, 0.0])
    v = jnp.array([-0.8 * vc, 0.6 * vc, 0.0])
    out = _isochrone_aaf(M, B, G, x, v)
    assert all(jnp.all(jnp.isfinite(o)) for o in out)
    jac = jax.jacfwd(_flat, argnums=(0, 1, 2))(x, v, jnp.asarray(B))
    assert all(jnp.all(jnp.isfinite(j)) for j in jac)


@pytest.mark.parametrize(
    ("x", "v"),
    [
        ([0.0, 0.0, 1.5], [0.3, 0.2, 0.1]),  # on the z axis
        ([0.0, 0.0, 0.0], [0.3, 0.2, 0.1]),  # origin
    ],
)
def test_isochrone_axis_and_origin_finite(x, v):
    x, v = jnp.array(x), jnp.array(v)
    assert all(jnp.all(jnp.isfinite(o)) for o in _isochrone_aaf(M, B, G, x, v))
    jac = jax.jacfwd(_flat, argnums=(0, 1, 2))(x, v, jnp.asarray(B))
    assert all(jnp.all(jnp.isfinite(j)) for j in jac)


@pytest.mark.parametrize(
    ("x", "v"),
    [
        ([1.0, 0.5, 0.4], [0.1, 0.5, -0.2]),  # generic
        ([1.0, -0.3, 0.6], [0.2, -0.4, 0.3]),  # retrograde, inclined
        ([1.0, 0.0, 0.5], [0.2, 0.0, 0.5]),  # polar, Lz = 0
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
    # planar retrograde, inclined retrograde, planar prograde
    xyz = np.concatenate([xyz, [[6.0, 0, 0], [5.0, 1, 0.5], [6.0, 0, 0]]])
    v = np.concatenate([v, [[0, -0.15, 0], [0.01, -0.12, 0.02], [0, 0.15, 0]]])
    # keep bound points away from exact z-turning points (gala forces psi=+pi/2)
    phi = -float(gpot.G * 1e11) / (2.0 + np.sqrt(4.0 + (xyz**2).sum(-1)))
    bound = 0.5 * (v**2).sum(-1) + phi < 0
    xyz, v = xyz[bound], v[bound]
    # exclude |v_theta| < 1e-8 (continuous vs forced psi convention)
    s = (xyz**2).sum(-1) * v[:, 2] - xyz[:, 2] * (xyz * v).sum(-1)
    keep = np.abs(s) > 1e-8
    xyz, v = xyz[keep], v[keep]

    res = iaaf(POT, jnp.asarray(xyz), jnp.asarray(v), jnp.asarray(0.0))
    pos = gala_dyn.PhaseSpacePosition(xyz.T * au.kpc, v.T * au.kpc / au.Myr)
    out = isochrone_xv_to_aa(pos, gpot)
    act = out[0].to_value("kpc2/Myr").T
    ang = out[1].to_value("rad").T
    frq = out[2].to_value("rad/Myr").T
    np.testing.assert_allclose(res.actions, act, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res.freqs, frq, rtol=1e-10)
    np.testing.assert_allclose(np.sin(res.angles), np.sin(ang), atol=1e-9)
    np.testing.assert_allclose(np.cos(res.angles), np.cos(ang), atol=1e-9)
