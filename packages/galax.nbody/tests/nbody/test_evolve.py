"""Tests for `galax.nbody.evolve` and running an `NBodyField` through diffrax."""

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

import diffraxtra as dfxtra
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
    sol = gn.evolve(field, q0, p0, 0.0, period / n, n)
    return float(jnp.max(jnp.linalg.norm(sol.ys[0][-1] - q0, axis=-1)))


def _cluster(n: int = 16, seed: int = 0, m: float = 1e10):
    """Return a small bound cluster in galactic units."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 3))
    p = rng.normal(size=(n, 3)) * 0.05
    field = gn.NBodyField(
        masses=u.Q(np.full(n, m), "Msun"), eps=u.Q(0.1, "kpc"), external_potential=_NULL
    )
    return field, jnp.asarray(q), jnp.asarray(p)


def _temp(f, *args) -> int:
    return jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes


def test_kepler_two_body_period() -> None:
    """After one period the positions return to 1e-6 relative."""
    assert _kepler_error(20000) < 1e-6


def test_kepler_dt2_convergence() -> None:
    """Halving dt divides the error by about 4."""
    ratio = _kepler_error(200) / _kepler_error(400)
    assert 3.6 < ratio < 4.4


def _growing_kepler():
    """Return a Kepler potential whose mass grows linearly in time."""
    m = gp.params.LinearParameter(
        slope=u.Q(1e9, "Msun/Myr"),
        point_time=u.Q(0.0, "Myr"),
        point_value=u.Q(1e11, "Msun"),
    )
    return gp.KeplerPotential(m_tot=m, units="galactic")


@pytest.mark.parametrize(
    ("pot", "t0"),
    [
        (gp.NFWPotential(m=1e11, r_s=10, units="galactic"), 0.0),
        (_growing_kepler(), 3.0),
    ],
    ids=["nfw", "growing-kepler"],
)
def test_tracers_match_compute_orbit(pot, t0) -> None:
    """Tracers follow `go.compute_orbit` with Leapfrog, also with forces that change."""
    q0 = jnp.asarray([[8.0, 0, 0], [0, 12, 1], [5, 5, 5]])
    p0 = jnp.asarray([[0.0, 0.2, 0], [-0.15, 0, 0.02], [0.1, -0.1, 0.05]])
    dt = 0.5
    ts = t0 + dt * jnp.arange(0, 41, 10)
    field = gn.NBodyField(
        masses=jnp.zeros(3), eps=0.0, external_potential=pot, n_massive=0
    )
    sol = gn.evolve(field, q0, p0, t0, dt, 40, save_every=10)
    np.testing.assert_allclose(sol.ts, ts, rtol=1e-15)

    solver = go.OrbitSolver(gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize())
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(q0, "kpc"), p=u.Q(p0, "kpc/Myr"), t=u.Q(t0, "Myr")
    )
    orbit = go.compute_orbit(pot, w0, u.Q(ts, "Myr"), solver=solver, dt0=u.Q(dt, "Myr"))
    expect = jnp.stack(
        [orbit.q.x.ustrip("kpc"), orbit.q.y.ustrip("kpc"), orbit.q.z.ustrip("kpc")], -1
    )
    np.testing.assert_allclose(sol.ys[0], jnp.swapaxes(expect, 0, 1), rtol=1e-12)


def test_forward_back_round_trip() -> None:
    """200 steps out, flip the velocities, 200 steps back: q returns to 1e-10."""
    field, q0, p0 = _cluster()
    out = gn.evolve(field, q0, p0, 0.0, 0.01, 200)
    q1, p1 = out.ys[0][-1], out.ys[1][-1]
    back = gn.evolve(field, q1, -p1, 0.0, 0.01, 200)
    assert float(jnp.max(jnp.abs(back.ys[0][-1] - q0))) < 1e-10
    assert float(jnp.max(jnp.abs(back.ys[1][-1] + p0))) < 1e-10


def test_negative_dt_runs_backward() -> None:
    """A negative ``dt`` from the end time returns to the start."""
    field, q0, p0 = _cluster()
    pot = gp.NFWPotential(m=1e11, r_s=10, units="galactic")
    field = gn.NBodyField(
        masses=field.masses, softening=field.softening, external_potential=pot
    )
    out = gn.evolve(field, q0, p0, 5.0, 0.01, 200)
    back = gn.evolve(field, out.ys[0][-1], out.ys[1][-1], out.ts[-1], -0.01, 200)
    np.testing.assert_allclose(back.ts[-1], 5.0, rtol=1e-14)
    assert float(jnp.max(jnp.abs(back.ys[0][-1] - q0))) < 1e-10


def test_save_times_and_first_save() -> None:
    """Saves are ``t0`` and every ``save_every`` steps; the first is ``(q0, p0)``."""
    field, q0, p0 = _cluster(n=4)
    t0, dt = 1234.5678, 0.01
    sol = gn.evolve(field, q0, p0, t0, dt, 12, save_every=3)
    np.testing.assert_allclose(sol.ts, t0 + dt * np.arange(0, 13, 3), rtol=1e-15)
    assert float(sol.ts[0]) == t0
    assert sol.ys[0].shape == sol.ys[1].shape == (5, 4, 3)
    assert bool(jnp.all(sol.ys[0][0] == q0))
    assert bool(jnp.all(sol.ys[1][0] == p0))
    every = gn.evolve(field, q0, p0, t0, dt, 12, save_every=1)
    np.testing.assert_allclose(sol.ys[0], every.ys[0][::3], rtol=1e-14)
    ends = gn.evolve(field, q0, p0, t0, dt, 12)
    assert ends.ts.shape == (2,)
    np.testing.assert_allclose(ends.ys[0], every.ys[0][::12], rtol=1e-14)


def test_quantity_inputs() -> None:
    """Quantities are stripped to the field's units."""
    field, q0, p0 = _cluster(n=4)
    plain = gn.evolve(field, q0, p0, 0.0, 0.01, 10)
    qty = gn.evolve(
        field,
        u.Q(q0 * 1e3, "pc"),
        u.Q(p0, "kpc/Myr").uconvert("km/s"),
        u.Q(0.0, "kyr"),
        u.Q(10.0, "kyr"),
        10,
    )
    np.testing.assert_allclose(qty.ys[0], plain.ys[0], rtol=1e-12)
    np.testing.assert_allclose(qty.ys[1], plain.ys[1], rtol=1e-12)


@pytest.mark.parametrize(
    ("n_steps", "save_every", "match"),
    [
        (0, None, "n_steps must be a positive int"),
        (2.0, None, "n_steps must be a positive int"),
        (True, None, "n_steps must be a positive int"),
        (10, 0, "save_every must be a positive int"),
        (10, 3, "save_every=3 must divide n_steps=10"),
    ],
)
def test_bad_step_counts_raise(n_steps, save_every, match) -> None:
    field, q0, p0 = _cluster(n=4)
    with pytest.raises(ValueError, match=match):
        gn.evolve(field, q0, p0, 0.0, 0.01, n_steps, save_every=save_every)


def test_requires_x64() -> None:
    """With x64 off, evolve raises instead of running in float32."""
    field, q0, p0 = _cluster(n=4)
    jax.config.update("jax_enable_x64", False)  # noqa: FBT003
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            gn.evolve(field, q0, p0, 0.0, 0.01, 10)
    finally:
        jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@pytest.mark.slow
def test_evolve_under_runtime_type_checking() -> None:
    """A tiny run works with beartype checking the jaxtyping annotations."""
    pytest.importorskip("beartype")
    code = textwrap.dedent("""
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import galax.nbody as gn
        import galax.potential as gp
        field = gn.NBodyField(
            masses=jnp.full(2, 1e9),
            eps=0.1,
            external_potential=gp.NullPotential(units="galactic"),
        )
        q0 = jnp.asarray([[-1.0, 0, 0], [1.0, 0, 0]])
        sol = gn.evolve(field, q0, jnp.zeros((2, 3)), 0.0, 0.05, 6, save_every=2)
        assert sol.ys[0].shape == (4, 2, 3)
        assert gn.total_energy(field, sol).shape == (4,)
        import numpy as np
        e = gn.total_energy(field, 0.0, np.asarray(q0), np.zeros((2, 3)))
        assert e.shape == ()
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


def test_empty_field_evolves() -> None:
    """With no particles there is nothing to integrate."""
    field = gn.NBodyField(masses=jnp.zeros(0), eps=0.1, external_potential=_NULL)
    sol = gn.evolve(field, jnp.zeros((0, 3)), jnp.zeros((0, 3)), 0.0, 0.1, 10)
    assert sol.ys[0].shape == (2, 0, 3)


# ----------------------------------------------------------------------------
# Traced times and recompilation


def test_traced_t0_and_dt() -> None:
    """``t0`` and ``dt`` work under jit and vmap, and grad works over ``dt``."""
    field, q0, p0 = _cluster(n=4)

    def end(t0, dt):
        return gn.evolve(field, q0, p0, t0, dt, 50).ys[0][-1]

    ref = end(1234.5678, 0.01)
    np.testing.assert_allclose(jax.jit(end)(1234.5678, 0.01), ref, rtol=1e-14)
    batched = jax.jit(jax.vmap(end, in_axes=(0, None)))(jnp.asarray([0.0, 1e4]), 0.01)
    # NullPotential: the start time does not change the trajectory.
    np.testing.assert_allclose(batched, jnp.stack([ref, ref]), rtol=1e-12)

    def loss(dt):
        return jnp.sum(end(0.0, dt) ** 2)

    g = jax.jit(jax.grad(loss))(0.01)
    eps = 1e-7
    fd = (loss(0.01 + eps) - loss(0.01 - eps)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-6)


def test_grad_over_t0_with_time_dependent_forces() -> None:
    """With a growing external mass, d(loss)/d(t0) is nonzero and matches FD."""
    field, q0, p0 = _cluster(n=4)
    field = gn.NBodyField(
        masses=field.masses,
        softening=field.softening,
        external_potential=_growing_kepler(),
    )
    q0 = q0 + jnp.asarray([8.0, 0.0, 0.0])

    def loss(t0):
        return jnp.sum(gn.evolve(field, q0, p0, t0, 0.01, 50).ys[0][-1] ** 2)

    g = jax.jit(jax.grad(loss))(3.0)
    eps = 1e-5
    fd = (loss(3.0 + eps) - loss(3.0 - eps)) / (2 * eps)
    assert abs(float(g)) > 1e-6
    np.testing.assert_allclose(g, fd, rtol=1e-5)


@pytest.mark.parametrize("dt", [0.0, float("nan"), float("inf")])
def test_bad_dt_raises(dt) -> None:
    field, q0, p0 = _cluster(n=4)
    with pytest.raises(eqx.EquinoxRuntimeError, match="dt must be finite, not 0"):
        gn.evolve(field, q0, p0, 0.0, dt, 10)
    run = jax.jit(lambda dt: gn.evolve(field, q0, p0, 0.0, dt, 10).ys[0])
    with pytest.raises(jax.errors.JaxRuntimeError, match="dt must be finite, not 0"):
        run(dt)


def _compiles(caplog):
    return [r for r in caplog.records if "Compiling" in r.getMessage()]


def test_eager_calls_reuse_the_compilation(caplog) -> None:
    """A new field of the same shapes, or new t0 and dt, compiles nothing."""
    n = 7  # shapes no other test uses, so the first call must compile
    field, q0, p0 = _cluster(n=n)

    def run(field, q, t0, dt):
        sol = gn.evolve(field, q, p0, t0, dt, 13, save_every=1)
        return sol.ys[0], gn.total_energy(field, sol)

    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        jax.block_until_ready(run(field, q0, 0.0, 0.01))
    # Positive control: the log capture does see compiles.
    assert _compiles(caplog)
    caplog.clear()
    field2 = gn.NBodyField(
        masses=u.Q(np.full(n, 5e9), "Msun"),
        eps=u.Q(0.2, "kpc"),
        external_potential=_NULL,
    )
    q1 = (q0 + 1).block_until_ready()
    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        jax.block_until_ready(run(field2, q1, 3.0, 0.02))
    assert not _compiles(caplog)


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


def test_graph_has_one_loop_however_often_it_saves() -> None:
    """The graph holds one solve, whatever ``save_every``."""
    field, q0, p0 = _cluster(n=4)

    def count(save_every):
        def f(q):
            return gn.evolve(field, q, p0, 0.0, 0.01, 60, save_every=save_every).ys

        return _count_while(jax.make_jaxpr(f)(q0).jaxpr)

    one = count(60)
    assert one > 0
    assert count(1) == count(5) == one


# ----------------------------------------------------------------------------
# The same field through diffrax directly


def test_native_diffrax_spellings_match() -> None:
    """`diffeqsolve`, `DiffEqSolver` and `OrbitSolver` give `evolve`'s states."""
    field, q0, p0 = _cluster(n=8)
    t0, dt, n, k = 0.0, 0.01, 40, 10
    ref = gn.evolve(field, q0, p0, t0, dt, n, save_every=k)
    grid = t0 + dt * jnp.arange(n + 1)
    lf = gf.Leapfrog()
    kw = {"saveat": dfx.SaveAt(t0=True, ts=grid[k::k])}
    ctrl = dfx.StepTo(ts=grid)

    s1 = dfx.diffeqsolve(
        field.terms(lf), lf, t0, grid[-1], None, (q0, p0), stepsize_controller=ctrl,
        max_steps=n, **kw,
    )  # fmt: skip
    s2 = dfxtra.DiffEqSolver(lf, stepsize_controller=ctrl, max_steps=n)(
        field, t0, grid[-1], None, (q0, p0), **kw
    )
    s3 = go.OrbitSolver(lf, stepsize_controller=ctrl, max_steps=n).solve(
        field, (q0, p0), t0, grid[-1], **kw
    )
    for s in (s1, s2, s3):
        np.testing.assert_allclose(s.ys[0], ref.ys[0], rtol=1e-14, atol=1e-14)


def test_other_solvers_on_the_grid_and_adaptive() -> None:
    """SemiImplicitEuler uses the two-term dispatch, and Dopri8 runs adaptively."""
    field, q0, p0 = _cluster(n=8)
    assert isinstance(field.terms(dfx.SemiImplicitEuler()), tuple)
    ref = gn.evolve(field, q0, p0, 0.0, 0.001, 1000)
    semi = gn.evolve(field, q0, p0, 0.0, 0.001, 1000, solver=dfx.SemiImplicitEuler())
    # First order, so close but not equal: about 1e-4 apart here.
    gap = float(jnp.max(jnp.abs(semi.ys[0][-1] - ref.ys[0][-1])))
    assert 1e-5 < gap < 1e-3
    # A solver on the default `__call__` path runs on the same grid as diffrax.
    tsit = gn.evolve(field, q0, p0, 0.0, 0.001, 1000, solver=dfx.Tsit5())
    grid = 0.001 * jnp.arange(1001)
    direct = dfx.diffeqsolve(
        field.terms(dfx.Tsit5()), dfx.Tsit5(), 0.0, grid[-1], None, (q0, p0),
        stepsize_controller=dfx.StepTo(ts=grid), max_steps=1000,
    )  # fmt: skip
    np.testing.assert_allclose(tsit.ys[0][-1], direct.ys[0][-1], rtol=1e-13)
    assert float(jnp.max(jnp.abs(tsit.ys[0][-1] - ref.ys[0][-1]))) > 1e-9
    solver = dfxtra.DiffEqSolver(
        dfx.Dopri8(),
        stepsize_controller=dfx.PIDController(rtol=1e-11, atol=1e-11),
        max_steps=10_000,
    )
    ad = solver(field, 0.0, 1.0, None, (q0, p0), saveat=dfx.SaveAt(t1=True))
    np.testing.assert_allclose(ad.ys[0][-1], ref.ys[0][-1], atol=1e-6)


# ----------------------------------------------------------------------------
# Results: energy and components


def _two_components():
    stars = gn.Component(name="stars", q=jnp.eye(3), p=jnp.zeros((3, 3)), m=1e10, h=0.1)
    tracers = gn.Component(
        name="tracers",
        q=2 * jnp.ones((2, 3)),
        p=jnp.ones((2, 3)),
        m=0.0,
        h=0.0,
        kind="tracer",
    )
    system = gn.System.from_components(stars, tracers, units="galactic")
    return system, gn.NBodyField.from_system(system, _NULL)


def test_component_split_and_energy_on_a_solution() -> None:
    system, field = _two_components()
    ps = system.particles
    sol = gn.evolve(field, ps.q, ps.p, 0.0, 0.01, 20, save_every=10)
    parts = system.split(sol.ys[0], axis=-2)
    assert parts["tracers"].shape == (3, 2, 3)
    assert bool(jnp.all(parts["tracers"] == sol.ys[0][:, 3:]))
    e = gn.total_energy(field, sol)
    assert e.shape == (3,)
    for i in range(3):
        one = gn.total_energy(field, sol.ts[i], sol.ys[0][i], sol.ys[1][i])
        np.testing.assert_allclose(e[i], one, rtol=1e-14)


def test_energy_on_a_solution_reads_each_save_time() -> None:
    """With a time-dependent potential each save's energy uses its own time."""
    field, q0, p0 = _cluster(n=4)
    field = gn.NBodyField(
        masses=field.masses,
        softening=field.softening,
        external_potential=_growing_kepler(),
    )
    q0 = q0 + jnp.asarray([8.0, 0.0, 0.0])
    sol = gn.evolve(field, q0, p0, 3.0, 0.5, 20, save_every=5)
    e = gn.total_energy(field, sol)
    at_zero = [
        gn.total_energy(field, 0.0, sol.ys[0][i], sol.ys[1][i]) for i in range(5)
    ]
    for i in range(5):
        one = gn.total_energy(field, sol.ts[i], sol.ys[0][i], sol.ys[1][i])
        np.testing.assert_allclose(e[i], one, rtol=1e-14)
        assert abs(float(e[i] - at_zero[i])) > 1e-6 * abs(float(e[i]))


def test_total_energy_is_its_own_function() -> None:
    """``gn.total_energy`` is not merged into ``gp.total_energy``, and takes numpy."""
    field, q0, p0 = _cluster(n=4)
    assert gn.total_energy is not gp.total_energy
    ref = gn.total_energy(field, 0.0, q0, p0)
    np.testing.assert_allclose(
        gn.total_energy(field, 0, np.asarray(q0), np.asarray(p0)), ref, rtol=1e-15
    )


def test_component_split_on_a_vmapped_solution() -> None:
    system, field = _two_components()
    ps = system.particles
    qs = jax.vmap(
        lambda q: gn.evolve(field, q, ps.p, 0.0, 0.01, 30, save_every=10).ys[0]
    )(jnp.stack([ps.q, ps.q + 1]))
    stars = system.split(qs, axis=-2)["stars"]
    assert stars.shape == (2, 4, 3, 3)
    assert bool(jnp.all(stars == qs[:, :, :3]))


def test_first_save_is_the_initial_state_even_if_a_row_breaks() -> None:
    """The first save is ``(q0, p0)`` even when a tracer is NaN after a step."""
    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    field = gn.NBodyField(
        masses=jnp.asarray([1e9, 0.0]), eps=0.0, external_potential=pot, n_massive=1
    )
    # The tracer sits at the Kepler centre, so its acceleration is NaN.
    q0 = jnp.asarray([[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    p0 = jnp.asarray([[0.0, 0.2, 0.0], [0.0, 0.1, 0.0]])
    sol = gn.evolve(field, q0, p0, 0.0, 0.01, 10)
    assert bool(jnp.all(sol.ys[0][0] == q0))
    assert bool(jnp.all(sol.ys[1][0] == p0))
    assert bool(jnp.all(jnp.isfinite(sol.ys[0][-1][0])))


# ----------------------------------------------------------------------------
# Gradients


def _mass_loss(n_steps: int, dt: float = 0.02, **kw):
    """Return ``s -> sum |q(T)|^2`` for an 8-body cluster with masses scaled by s."""
    _, q0, p0 = _cluster(n=8, seed=1, m=1e11)

    def loss(s):
        field = gn.NBodyField(
            masses=s * jnp.full(8, 1e11), eps=0.1, external_potential=_NULL
        )
        sol = gn.evolve(field, q0, p0, 0.0, dt, n_steps, **kw)
        return jnp.sum(sol.ys[0][-1] ** 2)

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

    def run(q):
        return gn.evolve(field, q, p0, 0.0, 0.01, 10, save_every=5).ys[0]

    batched = jax.vmap(run)(jnp.stack([q0, q1]))
    for i, q in enumerate([q0, q1]):
        np.testing.assert_allclose(batched[i], run(q), rtol=1e-12)


def test_nograd_forward_under_checkpoint_adjoint() -> None:
    """A NoGrad backend runs forward with the default checkpoint adjoint."""
    field, q0, p0 = _cluster(n=4)
    nograd = gn.NBodyField(
        masses=field.masses,
        softening=field.softening,
        external_potential=_NULL,
        backend=gn.DirectSum(grad="none"),
    )
    res = gn.evolve(nograd, q0, p0, 0.0, 0.01, 10)
    ref = gn.evolve(field, q0, p0, 0.0, 0.01, 10)
    np.testing.assert_allclose(res.ys[0], ref.ys[0], rtol=1e-12)


# ----------------------------------------------------------------------------
# Memory


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
        return gn.evolve(eqx.combine(d, static), q, p, 0.0, 1e-3, 3).ys

    def acc(d, q):
        return eqx.combine(d, static).acceleration(0.0, q)

    t_fwd, t_acc = _temp(fwd, dyn, q, q), _temp(acc, dyn, q)
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

    def run(q):
        return gn.evolve(field, q, q, 0.0, 1.0, s - 1, save_every=1)

    def run_energy(q):
        sol = run(q)
        return sol.ys, gn.total_energy(field, sol)

    # The integration alone takes about 4x one evaluation, so compare the
    # memory the energy adds, not the total.
    added = _temp(run_energy, q) - _temp(lambda q: run(q).ys, q)
    one = _temp(lambda q: gn.total_energy(field, 0.0, q, q), q)
    assert added < 4 * one


# ----------------------------------------------------------------------------
# check_timestep


def test_timestep_guard_warns() -> None:
    """A dt 10x above dt_max warns, or raises when strict."""
    pot = gp.NFWPotential(m=1e11, r_s=10, units="galactic")
    field, q0, _ = _cluster(n=4)
    field = gn.NBodyField(
        masses=field.masses, softening=field.softening, external_potential=pot
    )
    dt_max = gn.check_timestep(field, q0, 0.0, 0.0)
    assert 0 < dt_max < np.inf
    with pytest.raises(ValueError, match="dt"):
        gn.check_timestep(field, q0, 0.0, 10 * dt_max, strict=True)
    with pytest.warns(UserWarning, match="dt") as rec:
        gn.check_timestep(field, q0, 0.0, 10 * dt_max)
    assert rec[0].filename == __file__


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), -1.0, np.asarray([0.01])])
def test_check_timestep_rejects_bad_dt(dt) -> None:
    """``check_timestep`` raises for a dt that is negative or not finite."""
    field = gn.NBodyField(masses=jnp.ones(2), eps=0.1, external_potential=_NULL)
    with pytest.raises(ValueError, match="finite, non-negative scalar"):
        gn.check_timestep(field, jnp.eye(2, 3), 0.0, dt, strict=True)


@pytest.mark.parametrize("eps", [0.0, 0.1])
def test_check_timestep_nan_position_is_reported(eps) -> None:
    """A NaN position of an active particle gives a NaN ``dt_max``, softened or not."""
    pot = gp.NFWPotential(m=1e11, r_s=10, units="galactic")
    field = gn.NBodyField(masses=jnp.zeros(2), eps=eps, external_potential=pot)
    q0 = jnp.asarray([[8.0, 0.0, 0.0], [jnp.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="dt_max is NaN"):
        gn.check_timestep(field, q0, 0.0, 1.0, strict=True)


def test_total_energy_of_a_solution_under_filter_jit() -> None:
    field, q0, p0 = _cluster(n=4)
    sol = gn.evolve(field, q0, p0, 0.0, 0.01, 10, save_every=5)
    np.testing.assert_allclose(
        eqx.filter_jit(gn.total_energy)(field, sol),
        gn.total_energy(field, sol),
        rtol=1e-14,
    )


@pytest.mark.parametrize(
    ("t0", "dt"), [(0.0, jnp.asarray([0.01, 0.02])), (jnp.zeros(2), 0.01)]
)
def test_nonscalar_t0_or_dt_raises(t0, dt) -> None:
    field, q0, p0 = _cluster(n=4)
    with pytest.raises(ValueError, match="t0 and dt must be scalars"):
        gn.evolve(field, q0, p0, t0, dt, 2)


def test_numpy_integer_step_counts() -> None:
    field, q0, p0 = _cluster(n=4)
    sol = gn.evolve(field, q0, p0, 0.0, 0.01, np.int64(10), save_every=np.int32(5))
    assert sol.ts.shape == (3,)


def test_quantity_t0_in_other_units_with_changing_forces() -> None:
    """A Quantity ``t0`` and ``dt`` in Gyr are converted before the forces read them."""
    field, q0, p0 = _cluster(n=4)
    field = gn.NBodyField(
        masses=field.masses,
        softening=field.softening,
        external_potential=_growing_kepler(),
    )
    q0 = q0 + jnp.asarray([8.0, 0.0, 0.0])
    plain = gn.evolve(field, q0, p0, 3.0, 0.5, 20, save_every=10)
    qty = gn.evolve(
        field, q0, p0, u.Q(3e-3, "Gyr"), u.Q(5e-4, "Gyr"), 20, save_every=10
    )
    np.testing.assert_allclose(qty.ts, plain.ts, rtol=1e-14)
    np.testing.assert_allclose(qty.ys[0], plain.ys[0], rtol=1e-12)


def test_max_steps_is_the_step_count() -> None:
    """``max_steps`` follows ``n_steps``, so the checkpoint memory follows the run."""
    field, q0, p0 = _cluster(n=4)
    assert gn.evolve(field, q0, p0, 0.0, 0.01, 37).stats["max_steps"] == 37
