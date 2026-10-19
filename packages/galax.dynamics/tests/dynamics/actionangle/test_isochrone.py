"""Tests for the analytic isochrone action-angle core."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from galax.dynamics._src.actionangle.isochrone import _isochrone_aaf

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
