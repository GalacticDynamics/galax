"""Tests for `SpraySimulator` in per-particle mode with the shared track."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxmore.loop_strategies as lstrat
import numpy as np
import pytest

import galax.dynamics as gd
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
from galax.dynamics._src.experimental.solve import is_lattice, progenitor_states
from galax.dynamics._src.experimental.track import (
    KNOTS_PER_RELEASE,
    MIN_KNOTS,
    _hermite,
    make_track,
)

gde = gd.experimental
LOOPS = [lstrat.VMap, lstrat.Scan]


class _PlainTrack(eqx.Module):
    """The knot table read with plain autodiff: no `custom_jvp`, no `S`."""

    t0: jax.Array
    dt: jax.Array
    w: jax.Array

    def __call__(self, t: jax.Array) -> jax.Array:
        return _hermite(self.t0, self.dt, self.w[:, :3], self.w[:, 3:], t)


def _plain_make_track(pot, w_today, schedule, solver, ws, wrt):
    """`make_track`'s knots, with the plain table's autodiff."""
    if is_lattice(solver):
        w = ws
    else:
        n = max(KNOTS_PER_RELEASE * schedule.n_release, MIN_KNOTS)
        schedule = gde.Uniform(schedule.t_age, n, 1, schedule.t1)
        w = progenitor_states(pot, w_today, schedule, solver)
    h = jnp.asarray(schedule.t_age, dtype=float) / schedule.n_steps()
    return _PlainTrack(schedule.lattice()[0], h, w)


def _loss(loop, df_cls=gde.Chen2024DF, solver=LEAP, psolver=None, **track_kw):
    sim = gde.SpraySimulator(
        df_cls(),
        gde.PerParticle(gde.SharedTrack(**track_kw), loop),
        progenitor_potential=plummer,
        solver=solver,
        progenitor_solver=psolver,
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


@pytest.mark.filterwarnings("ignore:Chen2024DF without a progenitor:RuntimeWarning")
@pytest.mark.parametrize("other", ["whole", "copies"])
@pytest.mark.parametrize("prog", [None, plummer])
@pytest.mark.parametrize("loop", LOOPS)
def test_track_matches_whole_stream(loop, prog, other) -> None:
    """Forces are evaluated on the lattice, where the track is exact."""
    sched = gde.Uniform(jnp.asarray(300.0), 5, 8)
    df = gde.Chen2024DF()
    mode = gde.WholeStream() if other == "whole" else gde.PerParticle(gde.Copies())
    ref = gde.SpraySimulator(df, mode, progenitor_potential=prog)
    per = gde.SpraySimulator(
        df, gde.PerParticle(gde.SharedTrack(), loop), progenitor_potential=prog
    )
    want = ref(KEY, HOST, W_TODAY, MSAT, sched)
    got = per(KEY, HOST, W_TODAY, MSAT, sched)
    for g, w in ((got.lead, want.lead), (got.trail, want.trail)):
        np.testing.assert_allclose(g[:, :3], w[:, :3], rtol=0, atol=1e-11)
        np.testing.assert_allclose(g[:, 3:], w[:, 3:], rtol=0, atol=1e-13)
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
def test_track_forward_drops_S(loop, monkeypatch) -> None:
    """A forward-only run costs what the plain knot table does: `S` is dropped."""

    def cost(f):
        c = jax.jit(f).lower(*X).compile()
        loops = c.as_text().count(" while(")
        return c.memory_analysis().temp_size_in_bytes, c.cost_analysis()["flops"], loops

    (temp, flops, loops) = cost(_loss(loop))
    # The plain table on the lattice reuses the release solve's states.
    monkeypatch.setattr(simulator, "make_track", _plain_make_track)
    (temp0, flops0, loops0) = cost(_loss(loop))
    # S here is 61 knots x 6 x 10 leaves = 29 KiB; computing it in the forward
    # adds ~60 KiB temp and 2.5x the flops. What is left is theta and a few
    # elementwise ops (< 3 KiB, ~1.6% of these tiny solves' flops). Solving the
    # knots again instead of reusing ``ws`` adds two while loops.
    assert loops == loops0, (loops, loops0)
    assert temp - temp0 < 4096, (temp, temp0)
    assert flops < 1.03 * flops0, (flops, flops0)


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


@pytest.mark.parametrize("loop", LOOPS)
def test_track_grad_adaptive_knots(loop, monkeypatch) -> None:
    """With an adaptive progenitor solve the knots are uniform, not the lattice."""
    f = _loss(loop, psolver=adaptive(1e-10, 10_000))
    g = _flat(jax.jit(jax.grad(f, argnums=(0, 1, 2, 3)))(*X))
    monkeypatch.setattr(simulator, "make_track", _plain_make_track)
    want = _flat(jax.jit(jax.grad(f, argnums=(0, 1, 2, 3)))(*X))
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-12 * jnp.abs(want).max())


@pytest.mark.parametrize("loop", LOOPS)
def test_track_failed_progenitor_is_nan(loop) -> None:
    """A failed progenitor solve gives an all-NaN result under jit, no raise."""
    sim = gde.SpraySimulator(
        gde.Fardal2015DF(),
        gde.PerParticle(gde.SharedTrack(), loop),
        progenitor_potential=plummer,
        progenitor_solver=adaptive(1e-10, 5),
    )
    res = sim(KEY, HOST, W_TODAY, MSAT, gde.Uniform(jnp.asarray(300.0), 4, 5))
    for a in (res.lead, res.trail, res.progenitor):
        assert jnp.all(jnp.isnan(a))


MW = gp.MilkyWayPotential(units=USYS)


def _mw(s):
    halo = gp.NFWPotential(m=5.4e11 * s, r_s=15.62, units=USYS)
    return gp.CompositePotential(
        **{k: v for k, v in MW.items() if k != "halo"}, halo=halo
    )


def _wrt(w, pot, s):
    return w, pot["halo"].m, s.t_age


@pytest.mark.parametrize(
    ("psolver", "shapes"),
    [(None, [(4, 20), (5, 8)]), (adaptive(1e-10, 10_000), [(4, 20)])],
    ids=["lattice", "adaptive"],
)
def test_track_wrt(psolver, shapes) -> None:
    """Restricting theta keeps its leaves' gradients and shrinks S.

    One `wrt` serves every schedule shape: it isn't tied to a treedef.
    """

    def grad(**track_kw):
        sim = gde.SpraySimulator(
            gde.Chen2024DF(),
            gde.PerParticle(gde.SharedTrack(**track_kw)),
            progenitor_potential=plummer,
            progenitor_solver=psolver,
        )

        def f(s, w, t_age, n, spr):
            r = simulator.simulate(
                sim, KEY, _mw(s), w, MSAT, gde.Uniform(t_age, n, spr)
            )
            return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

        return jax.jit(jax.grad(f, argnums=(0, 1, 2)), static_argnums=(3, 4))

    x = (jnp.asarray(1.0), W_TODAY, jnp.asarray(300.0))
    full, part = grad(), grad(wrt=_wrt)
    for shape in shapes:
        want = _flat(full(*x, *shape))
        got = _flat(part(*x, *shape))
        np.testing.assert_allclose(
            got, want, rtol=1e-12, atol=1e-12 * jnp.abs(want).max(), err_msg=str(shape)
        )

    sched = gde.Uniform(jnp.asarray(300.0), *shapes[0])
    solver = LEAP if psolver is None else psolver
    ws = progenitor_states(_mw(1.0), W_TODAY, sched, solver)

    def n_theta(wrt):
        track = eqx.filter_jit(make_track)(_mw(1.0), W_TODAY, sched, solver, ws, wrt)
        return track.Sx.shape[-1]

    assert n_theta(_wrt) == 9  # w_today, the halo mass, t_age, t1 (always)
    assert n_theta(None) == 22  # + 13 other host leaves


@pytest.mark.filterwarnings("ignore:Chen2024DF without a progenitor:RuntimeWarning")
def test_track_wrt_keeps_schedule() -> None:
    """A ``wrt`` without ``t_age``/``t1`` still gives their gradient (R19).

    The schedule's leaves move the knot times, so they're always in theta;
    holding their knot values shifted the track in time (about -1.18 vs 0.0051).
    """

    def grad(progenitor):
        sim = gde.SpraySimulator(
            gde.Fardal2015DF(),
            gde.PerParticle(progenitor),
            progenitor_potential=plummer,
        )

        def f(t_age, t1):
            r = sim(KEY, HOST, W_TODAY, MSAT, gde.Uniform(t_age, 5, 20, t1))
            return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

        return jax.jit(jax.grad(f, argnums=(0, 1)))(
            jnp.asarray(500.0), jnp.asarray(0.0)
        )

    want = jnp.stack(grad(gde.Copies()))
    got = jnp.stack(grad(gde.SharedTrack(wrt=lambda w, *_: (w,))))
    # d/dt1 is round-off: a static host is invariant under time translation.
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10 * jnp.abs(want).max())


def test_track_wrt_must_select_inputs() -> None:
    sched = gde.Uniform(jnp.asarray(300.0), 4, 20)
    ws = progenitor_states(HOST, W_TODAY, sched, LEAP)
    with pytest.raises(ValueError, match="wrt must return"):
        make_track(HOST, W_TODAY, sched, LEAP, ws, lambda w, *_: w[:3])


@pytest.mark.slow
def test_track_memory() -> None:
    """Adaptive MW host, 1000 particles, 3 Gyr, vmap, gradient: S's memory."""
    solver = adaptive(1e-7, 2**14)
    x = jnp.concat([jnp.asarray([1.0]), W_TODAY, jnp.asarray([3000.0])])

    def temp(progenitor):
        sim = gde.SpraySimulator(
            gde.Chen2024DF(),
            gde.PerParticle(progenitor),
            progenitor_potential=plummer,
            solver=solver,
        )

        def f(x):
            sched = gde.Uniform(x[7], 500)
            r = simulator.simulate(sim, KEY, _mw(x[0]), x[1:7], MSAT, sched)
            return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

        c = jax.jit(jax.grad(f)).lower(x).compile()
        return c.memory_analysis().temp_size_in_bytes

    track, copies = temp(gde.SharedTrack()), temp(gde.Copies())
    assert track <= copies + 64 * 2**20, (track / 2**20, copies / 2**20)
