"""Tests for `SpraySimulator` in per-particle mode with the shared track."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxmore.loop_strategies as lstrat
import numpy as np
import pytest

import diffraxtra as dfxtra

import galax.dynamics as gd
import galax.fields as gf
import galax.potential as gp
from .test_spray_whole_stream import (
    DFS,
    HOST,
    KEY,
    LEAP,
    MSAT,
    USYS,
    W_TODAY,
    adaptive,
    plummer,
)
from galax.dynamics._src.experimental import simulator
from galax.dynamics._src.experimental.solve import progenitor_states
from galax.dynamics._src.experimental.track import _hermite

gde = gd.experimental
LOOPS = [lstrat.VMap, lstrat.Scan]
FWD = dfxtra.DiffEqSolver(
    gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize(), adjoint=dfx.ForwardMode()
)


class _PlainTrack(eqx.Module):
    """The knot table read with plain autodiff: no `custom_jvp`, no `S`."""

    t0: jax.Array
    dt: jax.Array
    w: jax.Array

    def __call__(self, t: jax.Array) -> jax.Array:
        return _hermite(self.t0, self.dt, self.w[:, :3], self.w[:, 3:], t)


def _plain_make_track(pot, w_today, schedule, solver):
    h = jnp.asarray(schedule.t_age, dtype=float) / schedule.n_steps()
    w = progenitor_states(pot, w_today, schedule, solver)
    return _PlainTrack(schedule.lattice()[0], h, w)


def _loss(loop, df_cls=gde.Chen2024DF, solver=LEAP):
    sim = gde.SpraySimulator(
        df_cls(),
        gde.PerParticle(gde.SharedTrack(), loop),
        progenitor_potential=plummer,
        solver=solver,
    )

    # `simulate`, not the jitted `sim`, so a patched `make_track` is retraced.
    def f(s, w, t_age, t1):
        pot = gp.NFWPotential(m=1e12 * s, r_s=15.0, units=USYS)
        r = simulator.simulate(sim, KEY, pot, w, MSAT, gde.Uniform(t_age, 4, 20, t1))
        return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

    return f


X = (jnp.asarray(1.0), W_TODAY, jnp.asarray(300.0), jnp.asarray(0.0))


def _flat(gs):
    return jnp.concat([jnp.ravel(a) for a in gs])


@pytest.mark.parametrize("prog", [None, plummer])
@pytest.mark.parametrize("loop", LOOPS)
def test_track_matches_whole_stream(loop, prog) -> None:
    """Forces are evaluated on the lattice, where the track is exact."""
    sched = gde.Uniform(jnp.asarray(300.0), 5, 8)
    df = gde.Chen2024DF()
    whole = gde.SpraySimulator(df, gde.WholeStream(), progenitor_potential=prog)
    per = gde.SpraySimulator(
        df, gde.PerParticle(gde.SharedTrack(), loop), progenitor_potential=prog
    )
    want = whole(KEY, HOST, W_TODAY, MSAT, sched)
    got = per(KEY, HOST, W_TODAY, MSAT, sched)
    for g, w in ((got.lead, want.lead), (got.trail, want.trail)):
        np.testing.assert_allclose(g[:, :3], w[:, :3], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(got.release_times, want.release_times)


@pytest.mark.parametrize("loop", LOOPS)
@pytest.mark.parametrize("df_cls", DFS)
def test_track_grad_equals_table_grad(df_cls, loop, monkeypatch) -> None:
    """The track's gradient is the plain knot table's exact gradient."""
    f = _loss(loop, df_cls)
    g = _flat(jax.jit(jax.grad(f, argnums=(0, 1, 2, 3)))(*X))
    monkeypatch.setattr(simulator, "make_track", _plain_make_track)
    want = _flat(jax.jit(jax.grad(f, argnums=(0, 1, 2, 3)))(*X))
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-12 * jnp.abs(want).max())


@pytest.mark.parametrize("loop", LOOPS)
def test_track_grad_fd_jvp(loop) -> None:
    """Reverse mode vs ``jvp`` with ForwardMode and vs central differences."""
    f = jax.jit(_loss(loop))
    g = _flat(jax.grad(f, argnums=(0, 1, 2, 3))(*X))
    jf = _flat(jax.jit(jax.jacfwd(_loss(loop, solver=FWD), argnums=(0, 1, 2, 3)))(*X))
    np.testing.assert_allclose(g, jf, rtol=1e-10, atol=1e-10 * jnp.abs(g).max())

    flat = _flat(X)
    eps = jnp.concat(
        [jnp.asarray([1e-6]), jnp.full(6, 1e-6), jnp.asarray([1e-4, 1e-4])]
    )

    def f_flat(z):
        return f(z[0], z[1:7], z[7], z[8])

    fd = jnp.asarray(
        [
            (f_flat(flat.at[i].add(eps[i])) - f_flat(flat.at[i].add(-eps[i])))
            / (2 * eps[i])
            for i in range(flat.size)
        ]
    )
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7 * jnp.abs(g).max())


@pytest.mark.parametrize("loop", LOOPS)
def test_track_forward_drops_S(loop, monkeypatch) -> None:
    """A forward-only run costs what the plain knot table does: `S` is dropped."""

    def cost(f):
        c = jax.jit(f).lower(*X).compile()
        return c.memory_analysis().temp_size_in_bytes, c.cost_analysis()["flops"]

    (temp, flops) = cost(_loss(loop))
    monkeypatch.setattr(simulator, "make_track", _plain_make_track)
    (temp0, flops0) = cost(_loss(loop))
    # S here is 61 knots x 6 x 10 leaves = 29 KiB; computing it in the forward
    # adds ~60 KiB temp and 2.5x the flops. What is left is theta (< 1 KiB).
    assert temp - temp0 < 4096, (temp, temp0)
    assert flops < 1.01 * flops0, (flops, flops0)


def test_track_adaptive() -> None:
    sched = gde.Uniform(jnp.asarray(500.0), 8)
    df = gde.Fardal2015DF()
    solver = adaptive(1e-10, 10_000)

    def run(progenitor):
        sim = gde.SpraySimulator(
            df,
            gde.PerParticle(progenitor),
            progenitor_potential=plummer,
            solver=solver,
        )
        return sim(KEY, HOST, W_TODAY, MSAT, sched)

    a, b = run(gde.Copies()), run(gde.SharedTrack())
    d = jnp.linalg.norm(jnp.concat([a.lead - b.lead, a.trail - b.trail])[:, :3], axis=1)
    assert jnp.median(d) < 1e-6, jnp.median(d)
