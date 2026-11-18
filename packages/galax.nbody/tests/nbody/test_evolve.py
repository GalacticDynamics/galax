"""Tests for `galax.nbody.evolve`."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import galax.coordinates as gc
import galax.fields as gf
import galax.nbody as gn
import galax.orbits as go
import galax.potential as gp

_NULL = gp.NullPotential(units="galactic")


def _kepler(m: float = 1e10, a: float = 1.0):
    """Return an equal-mass circular binary, its initial state and period."""
    field = gn.NBodyField(masses=u.Q([m, m], "Msun"), eps=0.0, external_potential=_NULL)
    v = jnp.sqrt(field._G * m / (4 * a))
    q0 = jnp.asarray([[-a, 0, 0], [a, 0, 0]])
    p0 = jnp.asarray([[0, -v, 0], [0, v, 0]])
    return field, q0, p0, 2 * jnp.pi * a / v


def _kepler_error(n: int) -> float:
    field, q0, p0, period = _kepler()
    res = gn.evolve(field, q0, p0, jnp.asarray([0.0, period]), dt=period / n)
    return float(jnp.max(jnp.linalg.norm(res.q[-1] - q0, axis=-1)))


def _cluster(n: int = 16, seed: int = 0, m: float = 1e10):
    """Return a small bound cluster in galactic units."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 3))
    p = rng.normal(size=(n, 3)) * 0.05
    field = gn.NBodyField(
        masses=u.Q(np.full(n, m), "Msun"), eps=u.Q(0.1, "kpc"), external_potential=_NULL
    )
    return field, jnp.asarray(q), jnp.asarray(p)


def test_kepler_two_body_period() -> None:
    """After one period the positions return to 1e-6 relative."""
    assert _kepler_error(20000) < 1e-6


def test_kepler_dt2_convergence() -> None:
    """Halving dt divides the error by about 4."""
    ratio = _kepler_error(200) / _kepler_error(400)
    assert 3.6 < ratio < 4.4


def test_tracers_match_compute_orbit() -> None:
    """Tracers in an NFW potential follow `go.compute_orbit` with Leapfrog."""
    pot = gp.NFWPotential(m=1e11, r_s=10, units="galactic")
    q0 = jnp.asarray([[8.0, 0, 0], [0, 12, 1], [5, 5, 5]])
    p0 = jnp.asarray([[0.0, 0.2, 0], [-0.15, 0, 0.02], [0.1, -0.1, 0.05]])
    dt = 0.5
    ts = dt * jnp.arange(0, 41, 10)
    field = gn.NBodyField(
        masses=jnp.zeros(3), eps=0.0, external_potential=pot, n_massive=0
    )
    res = gn.evolve(field, q0, p0, ts, dt=dt)

    solver = go.OrbitSolver(gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize())
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(q0, "kpc"), p=u.Q(p0, "kpc/Myr"), t=u.Q(0.0, "Myr")
    )
    orbit = go.compute_orbit(pot, w0, u.Q(ts, "Myr"), solver=solver, dt0=u.Q(dt, "Myr"))
    expect = jnp.stack(
        [orbit.q.x.ustrip("kpc"), orbit.q.y.ustrip("kpc"), orbit.q.z.ustrip("kpc")], -1
    )
    np.testing.assert_allclose(res.q, jnp.swapaxes(expect, 0, 1), rtol=1e-12)


def test_forward_back_round_trip() -> None:
    """200 steps out, flip the velocities, 200 steps back: q returns to 1e-10."""
    field, q0, p0 = _cluster()
    dt = 0.01
    ts = jnp.asarray([0.0, 200 * dt])
    out = gn.evolve(field, q0, p0, ts, dt=dt, check_dt=False)
    back = gn.evolve(field, out.q[-1], -out.p[-1], ts, dt=dt, check_dt=False)
    assert float(jnp.max(jnp.abs(back.q[-1] - q0))) < 1e-10
    assert float(jnp.max(jnp.abs(back.p[-1] + p0))) < 1e-10


def test_save_times_exact() -> None:
    """``result.ts`` is ``ts`` bitwise, and uneven intervals work."""
    field, q0, p0 = _cluster(n=4)
    dt = 0.01
    ts = jnp.asarray([0.0, 3 * dt, 4 * dt, 10 * dt])
    res = gn.evolve(field, q0, p0, ts, dt=dt, check_dt=False)
    assert res.ts.shape == ts.shape
    assert bool(jnp.all(res.ts == ts))
    assert res.q.shape == (4, 4, 3)
    assert bool(jnp.all(res.q[0] == q0))
    full = gn.evolve(field, q0, p0, dt * jnp.arange(11), dt=dt, check_dt=False)
    np.testing.assert_allclose(res.q, full.q[jnp.asarray([0, 3, 4, 10])], rtol=1e-12)


def test_quantity_inputs() -> None:
    """Quantities are stripped to the field's units."""
    field, q0, p0 = _cluster(n=4)
    ts = jnp.asarray([0.0, 0.1])
    plain = gn.evolve(field, q0, p0, ts, dt=0.01, check_dt=False)
    qty = gn.evolve(
        field,
        u.Q(q0 * 1e3, "pc"),
        u.Q(p0, "kpc/Myr"),
        u.Q(ts * 1e3, "kyr"),
        dt=u.Q(10.0, "kyr"),
        check_dt=False,
    )
    np.testing.assert_allclose(qty.q, plain.q, rtol=1e-12)


def test_bad_dt_raises_before_tracing() -> None:
    """An interval that is not a multiple of dt raises, naming dt."""
    field, q0, p0 = _cluster(n=4)
    with pytest.raises(ValueError, match="dt"):
        gn.evolve(field, q0, p0, jnp.asarray([0.0, 0.025]), dt=0.01)
    with pytest.raises(ValueError, match="dt"):
        gn.evolve(field, q0, p0, jnp.asarray([0.0, 0.1]), dt=-0.01)
    with pytest.raises(ValueError, match="traced"):
        jax.jit(lambda dt: gn.evolve(field, q0, p0, jnp.asarray([0.0, 0.1]), dt=dt))(
            0.01
        )


def test_nonincreasing_ts_raises() -> None:
    field, q0, p0 = _cluster(n=4)
    for ts in ([0.0, 0.1, 0.1], [0.1, 0.0]):
        with pytest.raises(ValueError, match="increasing"):
            gn.evolve(field, q0, p0, jnp.asarray(ts), dt=0.01)


def test_timestep_guard_warns() -> None:
    """A dt 10x above dt_max warns, or raises when strict."""
    pot = gp.NFWPotential(m=1e11, r_s=10, units="galactic")
    field, q0, _ = _cluster(n=4)
    field = gn.NBodyField(
        masses=field.masses, softening=field.softening, external_potential=pot
    )
    dt_max = gn.check_timestep(field, q0, 0.0, 0.0)
    assert 0 < dt_max < np.inf
    with pytest.warns(UserWarning, match="dt"):
        gn.check_timestep(field, q0, 0.0, 10 * dt_max)
    with pytest.raises(ValueError, match="dt"):
        gn.check_timestep(field, q0, 0.0, 10 * dt_max, strict=True)
    ts = jnp.asarray([0.0, 10 * 10 * dt_max])
    with pytest.warns(UserWarning, match="dt"):
        gn.evolve(field, q0, jnp.zeros_like(q0), ts, dt=10 * dt_max)


def test_component_split_on_result() -> None:
    stars = gn.Component(name="stars", q=jnp.eye(3), p=jnp.zeros((3, 3)), m=1e10, h=0.1)
    tracers = gn.Component(
        name="tracers",
        q=2 * jnp.ones((2, 3)),
        p=jnp.zeros((2, 3)),
        m=0.0,
        h=0.0,
        massive=False,
    )
    system = gn.System.from_components(stars, tracers, units="galactic")
    field = gn.NBodyField.from_system(system, _NULL)
    ps = system.particles
    res = gn.evolve(
        field,
        ps.q,
        ps.p,
        jnp.asarray([0.0, 0.1, 0.2]),
        dt=0.01,
        system=system,
        energy=True,
    )
    q, p = res.component("tracers")
    assert q.shape == p.shape == (3, 2, 3)
    assert bool(jnp.all(q == res.q[:, 3:]))
    assert res.energy is not None
    assert res.energy.shape == (3,)
    no_sys = gn.evolve(field, ps.q, ps.p, jnp.asarray([0.0, 0.1]), dt=0.01)
    assert no_sys.energy is None
    with pytest.raises(ValueError, match="system"):
        no_sys.component("stars")


def test_trajectory_gradient_matches_fd() -> None:
    """d(sum |q(T)|^2)/d(mass scale) by jax.grad matches central FD, under jit."""
    _, q0, p0 = _cluster(n=8, seed=1, m=1e11)
    dt = 0.02
    ts = jnp.asarray([0.0, 75 * dt])  # about one crossing time

    def loss(s):
        field = gn.NBodyField(
            masses=s * jnp.full(8, 1e11), eps=0.1, external_potential=_NULL
        )
        res = gn.evolve(field, q0, p0, ts, dt=dt, check_dt=False)
        return jnp.sum(res.q[-1] ** 2)

    grad = jax.jit(jax.grad(loss))(1.0)
    eps = 1e-6  # the loss is steep enough that 1e-4 leaves 3e-5 FD error
    fd = (jax.jit(loss)(1.0 + eps) - jax.jit(loss)(1.0 - eps)) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


def test_vmap_over_initial_conditions() -> None:
    field, q0, p0 = _cluster(n=4)
    q1 = q0 + 0.1
    ts = jnp.asarray([0.0, 0.05, 0.1])
    batched = jax.vmap(lambda q: gn.evolve(field, q, p0, ts, dt=0.01).q)(
        jnp.stack([q0, q1])
    )
    for i, q in enumerate([q0, q1]):
        one = gn.evolve(field, q, p0, ts, dt=0.01, check_dt=False)
        np.testing.assert_allclose(batched[i], one.q, rtol=1e-12)


def test_nograd_forward_under_checkpoint_adjoint() -> None:
    """A NoGrad backend runs forward with the default checkpoint adjoint."""
    field, q0, p0 = _cluster(n=4)
    nograd = gn.NBodyField(
        masses=field.masses,
        softening=field.softening,
        external_potential=_NULL,
        backend=gn.DirectSum(grad="none"),
    )
    ts = jnp.asarray([0.0, 0.1])
    res = gn.evolve(
        nograd, q0, p0, ts, dt=0.01, adjoint=dfx.RecursiveCheckpointAdjoint()
    )
    ref = gn.evolve(field, q0, p0, ts, dt=0.01)
    np.testing.assert_allclose(res.q, ref.q, rtol=1e-12)
