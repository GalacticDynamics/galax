"""Tests for `galax.nbody.evolve`."""

import logging
import os
import subprocess
import sys
import textwrap

import diffrax as dfx
import equinox as eqx
import jax
import jax.extend.core
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u
from unxt.quantity import AllowValue

import galax.coordinates as gc
import galax.fields as gf
import galax.nbody as gn
import galax.nbody._src.evolve as evolve_module
import galax.orbits as go
import galax.potential as gp
from galax.nbody._src.evolve import advance

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
    ts = np.asarray([0.0, 0.1])  # concrete, so the dt check is the one that fires
    with pytest.raises(ValueError, match="dt must be concrete"):
        jax.jit(lambda dt: gn.evolve(field, q0, p0, ts, dt=dt))(0.01)


@pytest.mark.parametrize(
    ("ts", "dt"),
    [
        (1e4 + 1e-3 * np.arange(200), 1e-3),
        (-13800 + 1e-3 * np.arange(50), 1e-3),
        (u.Q(-13.8 + 1e-6 * np.arange(50), "Gyr"), u.Q(1, "kyr")),
        (u.Q(10 + 1e-6 * np.arange(4), "Gyr"), u.Q(1, "kyr")),
        (np.asarray([0.0, 0.3]), 0.1),
    ],
)
def test_on_grid_ts_at_large_abs_time_accepted(ts, dt) -> None:
    """The step check allows the round-off of ``diff(ts)`` at large ``|ts|``."""
    field = gn.NBodyField(masses=jnp.zeros(2), eps=0.0, external_potential=_NULL)
    q0, p0 = jnp.ones((2, 3)), jnp.zeros((2, 3))
    res = gn.evolve(field, q0, p0, ts, dt=dt, check_dt=False)
    ts_np = np.asarray(u.ustrip(AllowValue, "Myr", ts))
    assert bool(jnp.all(res.ts == ts_np))


def test_off_grid_ts_rejected() -> None:
    """A gap of 1e-7 over a 0.3 interval with dt=0.1 is off the grid."""
    field = gn.NBodyField(masses=jnp.zeros(2), eps=0.0, external_potential=_NULL)
    with pytest.raises(ValueError, match="whole number of steps"):
        gn.evolve(
            field,
            jnp.ones((2, 3)),
            jnp.zeros((2, 3)),
            np.asarray([0.0, 0.3 + 1e-7]),
            dt=0.1,
            check_dt=False,
        )


@pytest.mark.parametrize(
    ("ts", "dt"),
    [
        # 0.4 dt off the grid, which the old |t|-scaled tolerance accepted.
        (np.asarray([13000.0, 13000.0 + 1000.4e-5]), 1e-5),
        (np.asarray([1e4, 1e4 + 1.5e-6]), 1e-6),
        # Rounds to zero steps.
        (np.asarray([1e4, 1e4 + 3e-6]), 1e-5),
    ],
)
def test_off_grid_ts_at_large_abs_time_rejected(ts, dt) -> None:
    """At large ``|ts|`` the step check still rejects intervals off the grid."""
    field = gn.NBodyField(masses=jnp.zeros(2), eps=0.0, external_potential=_NULL)
    with pytest.raises(ValueError, match="whole number of steps"):
        gn.evolve(field, jnp.ones((2, 3)), jnp.zeros((2, 3)), ts, dt=dt, check_dt=False)


def _check_ts(monkeypatch, ts, dt) -> np.ndarray:
    """Run only the save-time check of `evolve` and return its step counts.

    The integration is replaced by a stub, so huge step counts cost nothing.
    """
    seen = []

    def fake_integrate(field, y, t0, dt, n_steps, adjoint):
        seen.append(np.asarray(n_steps))
        s = len(n_steps) + 1
        return jnp.zeros((s, *y[0].shape)), jnp.zeros((s, *y[1].shape))

    monkeypatch.setattr(evolve_module, "_integrate", fake_integrate)
    field = gn.NBodyField(masses=jnp.zeros(1), eps=0.0, external_potential=_NULL)
    res = gn.evolve(
        field, jnp.zeros((1, 3)), jnp.zeros((1, 3)), ts, dt=dt, check_dt=False
    )
    assert bool(jnp.all(res.ts == np.asarray(ts)))
    return seen[0]


def test_ts_far_off_grid_over_many_steps_rejected(monkeypatch) -> None:
    """0.0019 of a step off after 2e6 steps is off the grid."""
    dt = 1e-3
    with pytest.raises(ValueError, match="whole number of steps"):
        _check_ts(monkeypatch, np.asarray([0.0, 2e6 * dt + 0.0019 * dt]), dt)


@pytest.mark.parametrize(("t0", "dt"), [(-1e4, 1e-4), (-1e6, 1e-3), (-13800.0, 1e-3)])
def test_on_grid_ts_from_negative_start_to_zero_accepted(monkeypatch, t0, dt) -> None:
    """``ts`` built as ``t0 + k * dt`` from a negative start up to 0 is on the grid.

    The times near 0 carry the rounding of ``t0``, not of their own size.
    """
    kz = round(-t0 / dt)
    k = np.r_[0, kz - 6 : kz + 1]
    np.testing.assert_array_equal(_check_ts(monkeypatch, t0 + k * dt, dt), np.diff(k))
    if kz < 2e7:  # every step, up to 0
        n = _check_ts(monkeypatch, t0 + np.arange(kz + 1) * dt, dt)
        assert n.shape == (kz,)
        assert np.all(n == 1)


def _random_grid(rng):
    """Return ``(t0, dt, k)``, an on-grid ``ts = t0 + k * dt`` drawn at random.

    ``t0`` spans 0 and +-[1e-3, 1e12], ``dt`` is 100 to 1e6 float spacings of
    ``|t0|``, and intervals are 1 to 1e9 steps. A third of the cases start
    negative and run through or up to 0.
    """
    mode = rng.integers(3)
    s = int(rng.integers(2, 7))
    if mode == 0:
        t0, dt = 0.0, 10 ** rng.uniform(-6, 3)
    else:
        t0 = rng.choice([-1.0, 1.0]) * 10 ** rng.uniform(-3, 12)
        if mode == 2:
            t0 = -abs(t0)
        dt = 10 ** rng.uniform(2, 6) * np.spacing(abs(t0))
    big = int(10 ** rng.uniform(0, 9))
    if mode == 2:
        kz = round(-t0 / dt)
        small = int(10 ** rng.uniform(0, 3))
        end = kz + small * int(rng.integers(0, 3))
        k = np.r_[0, np.arange(end - (s - 2) * small, end + 1, small)]
        k = np.unique(k[k >= 0])
    else:
        k = big * np.arange(s)
    return t0, dt, k


def test_random_on_grid_ts_accepted(monkeypatch) -> None:
    """On-grid ``ts`` built as ``t0 + k * dt`` or with `np.arange` is accepted."""
    rng = np.random.default_rng(20261119)
    for _ in range(300):
        t0, dt, k = _random_grid(rng)
        ts = t0 + k * dt
        if dt < 100 * np.spacing(np.max(np.abs(ts))):
            continue
        np.testing.assert_array_equal(_check_ts(monkeypatch, ts, dt), np.diff(k))
        if len(np.unique(np.diff(k))) == 1:
            # np.arange adds one rounding of t0 per element, so keep it short.
            step = (k[1] - k[0]) * dt
            ts = np.arange(t0, t0 + (len(k) - 0.5) * step, step)
            n = _check_ts(monkeypatch, ts, dt)
            assert np.all(n == k[1] - k[0])


def test_random_off_grid_ts_rejected(monkeypatch) -> None:
    """A time moved off the grid by ``max(1e-3 step, 100 float spacings)`` raises."""
    rng = np.random.default_rng(20261120)
    tested = 0
    for _ in range(300):
        t0, dt, k = _random_grid(rng)
        ts = t0 + k * dt
        fmin = max(1e-3, 100 * np.spacing(np.max(np.abs(ts))) / dt)
        if fmin > 0.4:
            continue
        j = int(rng.integers(1, len(ts)))
        ts[j] += rng.choice([-1.0, 1.0]) * rng.uniform(fmin, 0.5) * dt
        with pytest.raises(ValueError, match="whole number of steps"):
            _check_ts(monkeypatch, ts, dt)
        tested += 1
    assert tested > 200


_ULP_1E9 = float(np.spacing(1e9))


@pytest.mark.parametrize(
    ("ts", "dt"),
    [
        # dt of 64 and 100 float spacings at 1e9, 2.3 steps apart.
        (np.asarray([1e9, 1e9 + 2.3 * 64 * _ULP_1E9]), 64 * _ULP_1E9),
        (np.asarray([1e9, 1e9 + 2.3 * 100 * _ULP_1E9]), 100 * _ULP_1E9),
        (np.asarray([1e10, 1e10 + 3.43e-5]), 1e-5),
        # dt of 1000 float spacings is fine, and 2.3 steps is still off the grid.
        (np.asarray([1e9, 1e9 + 2.3 * 1000 * _ULP_1E9]), 1000 * _ULP_1E9),
    ],
)
def test_off_grid_ts_with_dt_near_float_spacing_rejected(ts, dt) -> None:
    """A dt near the float spacing of ``ts`` cannot hide an off-grid interval."""
    field = gn.NBodyField(masses=jnp.zeros(2), eps=0.0, external_potential=_NULL)
    with pytest.raises(ValueError, match=r"whole number of steps|too small"):
        gn.evolve(field, jnp.ones((2, 3)), jnp.zeros((2, 3)), ts, dt=dt, check_dt=False)


def test_on_grid_ts_with_dt_of_1000_float_spacings_accepted() -> None:
    """At 1e9 a dt of 1000 float spacings still accepts on-grid intervals."""
    field = gn.NBodyField(masses=jnp.zeros(2), eps=0.0, external_potential=_NULL)
    dt = 1000 * _ULP_1E9
    ts = 1e9 + dt * np.arange(4)
    res = gn.evolve(
        field, jnp.ones((2, 3)), jnp.zeros((2, 3)), ts, dt=dt, check_dt=False
    )
    assert bool(jnp.all(res.ts == ts))


@pytest.mark.parametrize("check_dt", [True, False])
@pytest.mark.parametrize("energy", [False, True])
def test_eager_calls_reuse_the_compilation(
    caplog, energy: bool, check_dt: bool
) -> None:
    """A second eager call with a new field of the same shapes compiles nothing."""
    # Shapes used by no other test, so the first call must compile.
    n = 7 + energy + 2 * check_dt
    field, q0, p0 = _cluster(n=n)
    ts = 0.01 * np.asarray([0, 1, 2, 5, 8, 9, 13])

    def run(field, q):
        r = gn.evolve(field, q, p0, ts, dt=0.01, check_dt=check_dt, energy=energy)
        return r.q, r.energy

    def compiles():
        return [r for r in caplog.records if "Compiling" in r.getMessage()]

    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        jax.block_until_ready(run(field, q0))
    # Positive control: the log capture does see compiles.
    assert compiles()
    caplog.clear()
    # Different masses and softening, same shapes. Lighter and softer, so
    # check_timestep does not warn.
    field2 = gn.NBodyField(
        masses=u.Q(np.full(n, 5e9), "Msun"),
        eps=u.Q(0.2, "kpc"),
        external_potential=_NULL,
    )
    q1 = (q0 + 1).block_until_ready()
    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        jax.block_until_ready(run(field2, q1))
    assert not compiles()


def test_requires_x64() -> None:
    """With x64 off, evolve raises instead of running in float32."""
    field, q0, p0 = _cluster(n=4)
    jax.config.update("jax_enable_x64", False)  # noqa: FBT003
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            gn.evolve(field, q0, p0, np.asarray([0.0, 0.1]), dt=0.01)
    finally:
        jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def test_dt_below_float_spacing_of_ts_raises() -> None:
    """A dt that rounds grid points together at large |ts| raises."""
    field, q0, p0 = _cluster(n=4)
    with pytest.raises(ValueError, match="too small for the magnitude of ts"):
        gn.evolve(
            field, q0, p0, np.asarray([1e10, 1e10 + 1e-5]), dt=1e-7, check_dt=False
        )


@pytest.mark.slow
def test_evolve_under_runtime_type_checking() -> None:
    """A tiny run works with beartype checking the jaxtyping annotations."""
    pytest.importorskip("beartype")
    code = textwrap.dedent("""
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import numpy as np
        import galax.nbody as gn
        import galax.potential as gp
        field = gn.NBodyField(
            masses=jnp.full(2, 1e9),
            eps=0.1,
            external_potential=gp.NullPotential(units="galactic"),
        )
        q0 = jnp.asarray([[-1.0, 0, 0], [1.0, 0, 0]])
        r = gn.evolve(
            field, q0, jnp.zeros((2, 3)), np.asarray([0.0, 0.1, 0.3]),
            dt=0.05, check_dt=False, energy=True,
        )
        assert r.q.shape == (3, 2, 3)
    """)
    env = os.environ | {"GALAX_ENABLE_RUNTIME_TYPECHECKING": "beartype.beartype"}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


def test_nonincreasing_ts_raises() -> None:
    field, q0, p0 = _cluster(n=4)
    for ts in ([0.0, 0.1, 0.1], [0.1, 0.0]):
        with pytest.raises(ValueError, match="increasing"):
            gn.evolve(field, q0, p0, jnp.asarray(ts), dt=0.01)


def test_nonfinite_ts_or_dt_raises() -> None:
    field, q0, p0 = _cluster(n=4)
    for ts in ([0.0, jnp.nan], [0.0, jnp.inf], [jnp.nan, 0.1]):
        with pytest.raises(ValueError, match="finite"):
            gn.evolve(field, q0, p0, jnp.asarray(ts), dt=0.01)
    for dt in (jnp.nan, jnp.inf):
        with pytest.raises(ValueError, match="finite"):
            gn.evolve(field, q0, p0, jnp.asarray([0.0, 0.1]), dt=dt)


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
    with pytest.warns(UserWarning, match="dt") as rec:
        gn.evolve(field, q0, jnp.zeros_like(q0), ts, dt=10 * dt_max)
    assert rec[0].filename == __file__
    with pytest.warns(UserWarning, match="dt") as rec:
        gn.check_timestep(field, q0, 0.0, 10 * dt_max)
    assert rec[0].filename == __file__


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


def test_ts_span_overflow_rejected() -> None:
    """Times whose span overflows float64 raise instead of passing the grid check."""
    field = gn.NBodyField(masses=jnp.zeros(2), eps=0.0, external_potential=_NULL)
    with pytest.raises(ValueError, match="overflows float64"):
        gn.evolve(
            field,
            jnp.ones((2, 3)),
            jnp.zeros((2, 3)),
            np.asarray([-1e308, 1e308]),
            dt=1e295,
            check_dt=False,
        )


@pytest.mark.parametrize(
    ("ts", "dt"),
    [
        # 1.7 steps, where the rounded grid point n*dt overflows to inf.
        (np.asarray([0.0, 1.7e308]), 1e308),
        # A subnormal dt, which XLA flushes to zero.
        (np.asarray([0.0, 1e-310, 2e-310]), 1e-310),
    ],
)
def test_extreme_ts_rejected(ts, dt) -> None:
    """Grids at the ends of the float64 range raise instead of running."""
    field = gn.NBodyField(masses=jnp.zeros(1), eps=0.0, external_potential=_NULL)
    with pytest.raises(ValueError, match=r"whole number of steps|subnormal"):
        gn.evolve(field, jnp.ones((1, 3)), jnp.zeros((1, 3)), ts, dt=dt, check_dt=False)


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), -1.0, np.asarray([0.01])])
def test_check_timestep_rejects_bad_dt(dt) -> None:
    """``check_timestep`` raises for a dt that is negative or not finite."""
    field = gn.NBodyField(masses=jnp.ones(2), eps=0.1, external_potential=_NULL)
    with pytest.raises(ValueError, match="finite, non-negative scalar"):
        gn.check_timestep(field, jnp.eye(2, 3), 0.0, dt, strict=True)


def test_single_save_time_far_from_zero_accepted() -> None:
    """With one save time there are no steps, so the dt floor does not apply."""
    field = gn.NBodyField(masses=jnp.zeros(1), eps=0.0, external_potential=_NULL)
    res = gn.evolve(
        field,
        jnp.ones((1, 3)),
        jnp.zeros((1, 3)),
        np.asarray([1e20]),
        dt=1.0,
        check_dt=False,
    )
    assert res.q.shape == (1, 1, 3)


def test_non_monotonic_ts_near_float_max_raises_cleanly() -> None:
    """Non-increasing ``ts`` near the float64 maximum raises without a warning."""
    field = gn.NBodyField(masses=jnp.zeros(1), eps=0.0, external_potential=_NULL)
    with pytest.raises(ValueError, match="strictly increasing"):
        gn.evolve(
            field,
            jnp.ones((1, 3)),
            jnp.zeros((1, 3)),
            np.asarray([-1e308, 1e308, 0.0]),
            dt=1e300,
            check_dt=False,
        )


def test_empty_field_evolves() -> None:
    """With no particles there is no time-step limit and nothing to integrate."""
    field = gn.NBodyField(masses=jnp.zeros(0), eps=0.1, external_potential=_NULL)
    res = gn.evolve(field, jnp.zeros((0, 3)), jnp.zeros((0, 3)), [0.0, 1.0], dt=0.1)
    assert res.q.shape == (2, 0, 3)


def test_system_size_checked_before_dt() -> None:
    """A wrong-size ``system`` raises before the time-step check can warn."""
    stars = gn.Component(name="stars", q=jnp.eye(3), p=jnp.zeros((3, 3)), m=1e9, h=0.1)
    system = gn.System.from_components(stars, units="galactic")
    field = gn.NBodyField(masses=jnp.full(2, 1e9), eps=0.1, external_potential=_NULL)
    with pytest.raises(ValueError, match="system has 3 particles but q0 has 2"):
        gn.evolve(
            field,
            jnp.eye(2, 3),
            jnp.zeros((2, 3)),
            [0.0, 100.0],
            dt=100.0,
            system=system,
        )


def test_system_of_wrong_size_rejected() -> None:
    """A ``system`` whose particle count differs from ``q0`` raises."""
    stars = gn.Component(name="stars", q=jnp.eye(3), p=jnp.zeros((3, 3)), m=1e9, h=0.1)
    system = gn.System.from_components(stars, units="galactic")
    field = gn.NBodyField(masses=jnp.full(4, 1e9), eps=0.1, external_potential=_NULL)
    with pytest.raises(ValueError, match="system has 3 particles but q0 has 4"):
        gn.evolve(
            field,
            jnp.eye(4, 3),
            jnp.zeros((4, 3)),
            np.asarray([0.0, 0.1]),
            dt=0.01,
            system=system,
            check_dt=False,
        )


def test_component_on_vmapped_result() -> None:
    """``component`` splits on the particle axis, also with a leading batch axis."""
    stars = gn.Component(name="stars", q=jnp.eye(3), p=jnp.zeros((3, 3)), m=1e10, h=0.1)
    tracers = gn.Component(
        name="tracers",
        q=2 * jnp.ones((2, 3)),
        p=jnp.ones((2, 3)),
        m=0.0,
        h=0.0,
        massive=False,
    )
    system = gn.System.from_components(stars, tracers, units="galactic")
    field = gn.NBodyField.from_system(system, _NULL)
    ps = system.particles
    ts = jnp.asarray([0.0, 0.1, 0.2, 0.3])

    def run(q):
        return gn.evolve(field, q, ps.p, ts, dt=0.01, system=system, check_dt=False)

    res = run(ps.q)
    q, p = res.component("stars")
    assert q.shape == p.shape == (4, 3, 3)
    assert bool(jnp.all(q == res.q[:, :3]))
    assert bool(jnp.all(p == res.p[:, :3]))
    resv = jax.vmap(run)(jnp.stack([ps.q, ps.q + 1]))
    q, p = resv.component("tracers")
    assert q.shape == p.shape == (2, 4, 2, 3)
    assert bool(jnp.all(q == resv.q[:, :, 3:]))
    assert bool(jnp.all(p == resv.p[:, :, 3:]))


def _mass_loss(n_steps: int, dt: float = 0.02, **kw):
    """Return ``s -> sum |q(T)|^2`` for an 8-body cluster with masses scaled by s."""
    _, q0, p0 = _cluster(n=8, seed=1, m=1e11)
    ts = jnp.asarray([0.0, n_steps * dt])

    def loss(s):
        field = gn.NBodyField(
            masses=s * jnp.full(8, 1e11), eps=0.1, external_potential=_NULL
        )
        res = gn.evolve(field, q0, p0, ts, dt=dt, check_dt=False, **kw)
        return jnp.sum(res.q[-1] ** 2)

    return loss


# 75 steps of 0.02 are about one crossing time.
@pytest.mark.parametrize("n_steps", [38, 75, 150], ids=["0.5tc", "1tc", "2tc"])
def test_trajectory_gradient_matches_fd(n_steps: int) -> None:
    """d(sum |q(T)|^2)/d(mass scale) by jax.grad matches central FD, under jit."""
    loss = _mass_loss(n_steps)
    grad = jax.jit(jax.grad(loss))(1.0)
    eps = 1e-6  # the loss is steep enough that 1e-4 leaves 3e-5 FD error
    fd = (jax.jit(loss)(1.0 + eps) - jax.jit(loss)(1.0 - eps)) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


def test_vmap_grad_matches_per_sample_grad() -> None:
    loss = _mass_loss(38)
    s = jnp.asarray([0.8, 1.0, 1.2])
    batched = jax.jit(jax.vmap(jax.grad(loss)))(s)
    one = jnp.stack([jax.jit(jax.grad(loss))(x) for x in s])
    np.testing.assert_allclose(batched, one, rtol=1e-12)


def test_jacfwd_with_forward_mode_adjoint() -> None:
    """jax.jacfwd works with ForwardMode and matches jax.grad."""
    loss = _mass_loss(38, adjoint=dfx.ForwardMode())
    fwd = jax.jit(jax.jacfwd(loss))(1.0)
    rev = jax.jit(jax.grad(_mass_loss(38)))(1.0)
    np.testing.assert_allclose(fwd, rev, rtol=1e-10)


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


def test_forward_memory_is_chunk_bounded_with_field_argument() -> None:
    # 1000 = 3 * 256 + 232, so the last chunk sits outside the chunk loop. XLA
    # used to hoist its pair softening and self mask out of the time loop and
    # carry them through every step.
    n, chunk = 1000, 256
    rng = np.random.default_rng(0)
    field = gn.NBodyField(
        masses=np.full(n, 1.0 / n),
        softening=rng.uniform(0.01, 0.02, n),
        external_potential=_NULL,
        backend=gn.DirectSum(chunk_size=chunk),
    )
    q = jnp.asarray(rng.normal(size=(n, 3)))
    dyn, static = eqx.partition(field, eqx.is_array)

    def fwd(d, q, p):
        f, adjoint = eqx.combine(d, static), dfx.RecursiveCheckpointAdjoint()
        return advance(f, (q, p), 1e-3 * jnp.arange(4.0), adjoint)

    def acc(d, q):
        return eqx.combine(d, static).acceleration(0.0, q)

    def temp(f, *args):
        return jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes

    t_fwd, t_acc = temp(fwd, dyn, q, q), temp(acc, dyn, q)
    assert t_fwd < 16 * chunk * n * 8
    assert t_acc < 16 * chunk * n * 8
    # The time loop must not add a (chunk, N) array, not even a boolean one.
    assert t_fwd - t_acc < chunk * n


def test_energy_memory_is_one_evaluation() -> None:
    """The energy at S saves needs about the temp memory of one evaluation."""
    n, s = 1024, 20
    rng = np.random.default_rng(0)
    field = gn.NBodyField(masses=np.full(n, 1.0 / n), eps=0.1, external_potential=_NULL)
    q = jnp.asarray(rng.normal(size=(n, 3)))
    ts = jnp.arange(s, dtype=float)

    def temp(f, *args):
        return jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes

    def run(q, *, energy):
        res = gn.evolve(field, q, q, ts, dt=1.0, check_dt=False, energy=energy)
        return res.q, res.energy

    # The integration alone takes about 4x one evaluation, so compare the
    # memory the energy adds, not the total.
    added = temp(lambda q: run(q, energy=True), q) - temp(
        lambda q: run(q, energy=False), q
    )
    one = temp(lambda q: gn.total_energy(field, 0.0, q, q), q)
    assert added < 4 * one


def test_empty_ts_raises() -> None:
    field, q0, p0 = _cluster(n=4)
    with pytest.raises(ValueError, match="at least one time"):
        gn.evolve(field, q0, p0, np.asarray([]), dt=0.01)


def test_jit_with_closed_over_field_and_state_checks_dt() -> None:
    """Under jit the dt check runs on a concrete, closed-over field and state."""
    field, q0, p0 = _cluster(n=4)
    ts = np.asarray([0.0, 0.05, 0.1])
    ref = gn.evolve(field, q0, p0, ts, dt=0.01).q
    np.testing.assert_allclose(
        jax.jit(lambda p: gn.evolve(field, q0, p, ts, dt=0.01).q)(p0), ref, rtol=1e-14
    )
    np.testing.assert_allclose(
        jax.jit(lambda: gn.evolve(field, q0, p0, ts, dt=0.01).q)(), ref, rtol=1e-14
    )

    def loss(p):
        return jnp.sum(gn.evolve(field, q0, p, ts, dt=0.01).q[-1] ** 2)

    np.testing.assert_allclose(jax.jit(jax.grad(loss))(p0), jax.grad(loss)(p0))

    pot = gp.NFWPotential(m=1e11, r_s=10, units="galactic")
    nfw = gn.NBodyField(
        masses=field.masses, softening=field.softening, external_potential=pot
    )
    big = 10 * gn.check_timestep(nfw, q0, 0.0, 0.0)
    with pytest.warns(UserWarning, match="dt"):
        jax.jit(lambda: gn.evolve(nfw, q0, p0, np.asarray([0.0, 10 * big]), dt=big).q)()


def test_jit_with_concrete_quantity_ts_in_other_units() -> None:
    """Concrete Quantity ts and dt in kyr are converted, not treated as traced."""
    field, q0, p0 = _cluster(n=4)
    ts = u.Q(np.asarray([0.0, 100.0]), "kyr")
    dt = u.Q(10.0, "kyr")
    out = jax.jit(lambda q: gn.evolve(field, q, p0, ts, dt=dt).q)(q0)
    ref = gn.evolve(field, q0, p0, np.asarray([0.0, 0.1]), dt=0.01).q
    np.testing.assert_allclose(out, ref, rtol=1e-12)


def _count_while(jaxpr) -> int:
    """Count ``while`` primitives in ``jaxpr`` and all its sub-jaxprs."""
    n = 0
    for eqn in jaxpr.eqns:
        n += eqn.primitive.name == "while"
        for v in eqn.params.values():
            for sub in v if isinstance(v, tuple | list) else (v,):
                if isinstance(sub, jax.extend.core.ClosedJaxpr):
                    n += _count_while(sub.jaxpr)
                elif isinstance(sub, jax.extend.core.Jaxpr):
                    n += _count_while(sub)
    return n


def test_graph_has_one_loop_per_run_of_equal_segments() -> None:
    """Under an outer jit the graph grows with the runs of equal segments, not saves."""
    field, q0, p0 = _cluster(n=4)

    def count(ts):
        def f(q):
            return gn.evolve(field, q, p0, ts, dt=0.01, check_dt=False).q

        return _count_while(jax.make_jaxpr(f)(q0).jaxpr)

    one = count(0.01 * np.arange(2))
    assert one > 0
    assert count(0.01 * np.arange(5)) == count(0.01 * np.arange(50)) == one
    # Runs of 1, 3 and 1 steps: one loop per run of equal segments.
    assert count(0.01 * np.asarray([0, 1, 2, 5, 8, 9])) == 3 * one


def _scan_and_segment_loop():
    """Return evolve's ``q`` and a segment loop over `advance` for one run of 3."""
    field, q0, p0 = _cluster(n=4)
    dt = 0.01
    steps = np.asarray([0, 3, 6, 9])
    ts = dt * steps
    adjoint = dfx.RecursiveCheckpointAdjoint()

    def ref(q):
        # The grid points are ts[0] + (k + j) * dt, with ts[0] = 0.
        p, qs = p0, [q]
        for k, n in zip(steps[:-1], np.diff(steps), strict=True):
            grid = (int(k) + jnp.arange(int(n) + 1)) * jnp.asarray(dt)
            q, p = advance(field, (q, p), grid, adjoint)
            qs.append(q)
        return jnp.stack(qs)

    def new(q):
        return gn.evolve(field, q, p0, ts, dt=dt, check_dt=False).q

    return new, ref, q0


def test_scan_matches_one_segment_at_a_time() -> None:
    """A run of equal segments gives the states of a segment loop.

    The second and third segments of the run start at traced times inside the
    scan, which once made diffrax reject t1 != ts[-1] under jit.
    """
    new, ref, q0 = _scan_and_segment_loop()
    np.testing.assert_allclose(new(q0), ref(q0), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(jax.jit(new)(q0), ref(q0), rtol=1e-15, atol=1e-15)


@pytest.mark.slow
def test_scan_gradient_matches_one_segment_at_a_time() -> None:
    """A run of equal segments gives the gradients of a segment loop under jit."""
    new, ref, q0 = _scan_and_segment_loop()
    g_ref = jax.grad(lambda q: jnp.sum(ref(q) ** 2))(q0)
    g_jit = jax.jit(jax.grad(lambda q: jnp.sum(new(q) ** 2)))(q0)
    np.testing.assert_allclose(g_jit, g_ref, rtol=1e-14)
