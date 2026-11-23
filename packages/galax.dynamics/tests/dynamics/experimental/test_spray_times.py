"""Tests for the `Times` schedule (arbitrary release times)."""

import functools as ft

import diffrax as dfx
import jax
import jax.numpy as jnp
import jaxmore.loop_strategies as lstrat
import numpy as np
import pytest

import diffraxtra as dfxtra
import unxt as u

import galax.dynamics as gd
import galax.fields as gf
import galax.potential as gp
from .test_spray_whole_stream import (
    HOST,
    KEY,
    LEAP,
    MSAT,
    USYS,
    W_TODAY,
    adaptive,
    plummer,
)
from galax.dynamics._src.experimental.df import sample_releases
from galax.dynamics._src.experimental.fields import _CopiesField
from galax.dynamics._src.experimental.simulator import _evolve
from galax.dynamics._src.experimental.solve import progenitor_states, solve_on

gde = gd.experimental
T1 = jnp.asarray(0.0)
# Off the 30-step lattice over [-290.3, 0] (h ~ 9.68 Myr), except the first.
TS = jnp.asarray([-290.3, -201.7, -150.2, -42.9, -12.5])
SOLVERS = {"leapfrog": LEAP, "adaptive": adaptive(1e-10, 10_000)}


def whole(solver=LEAP) -> gde.SpraySimulator:
    return gde.SpraySimulator(
        gde.Fardal2015DF(),
        gde.WholeStream(),
        progenitor_potential=plummer,
        solver=solver,
    )


def release_ics(sched, solver=LEAP):
    ws = progenitor_states(HOST, W_TODAY, sched, solver)
    idx = sched.release_index()
    ts = sched.release_times()
    Ms = jnp.broadcast_to(MSAT, ts.shape)
    xl, vl, xt, vt = sample_releases(
        gde.Fardal2015DF(), KEY, HOST, ts, ws[idx, :3], ws[idx, 3:], Ms
    )
    return ws, jnp.concat([xl, vl], axis=-1), jnp.concat([xt, vt], axis=-1)


def test_times_schedule() -> None:
    sched = gde.Times(TS[::-1], 30)  # any order; t1 defaults to max(ts)
    assert float(sched.t1) == -12.5
    grid = sched.grid()
    assert grid.shape == (30 + 1 + 5,)
    assert jnp.all(jnp.diff(grid) >= 0)
    np.testing.assert_array_equal(sched.release_times(), TS[::-1])
    np.testing.assert_array_equal(grid[sched.release_index()], TS[::-1])
    np.testing.assert_array_equal(sched.lattice(), sched.uniform().lattice())
    u = gde.Uniform(jnp.asarray(300.0), 4, 5)
    np.testing.assert_array_equal(u.grid(), u.lattice())


@pytest.mark.parametrize(
    "kw",
    [
        {"ts": jnp.zeros((2, 3)), "steps": 4},
        {"ts": jnp.zeros(0), "steps": 4},
        {"ts": TS, "steps": 0},
        {"ts": TS, "steps": 4, "t1": jnp.zeros(2)},
    ],
)
def test_times_checks(kw) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        gde.Times(**kw)


def _loss(sim):
    def loss(ts: jax.Array) -> jax.Array:
        r = sim(KEY, HOST, W_TODAY, MSAT, gde.Times(ts, 30, T1))
        return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

    return loss


def test_times_off_lattice_grad() -> None:
    loss = _loss(whole())
    g = jax.grad(loss)(TS)
    eps = 1e-4
    fd = jnp.stack(
        [
            (loss(TS.at[i].add(eps)) - loss(TS.at[i].add(-eps))) / (2 * eps)
            for i in range(TS.shape[0])
        ]
    )
    assert jnp.all(g != 0), g
    np.testing.assert_allclose(g, fd, rtol=1e-5)

    fwd = dfxtra.DiffEqSolver(
        gf.Leapfrog(),
        stepsize_controller=dfx.ConstantStepSize(),
        adjoint=dfx.ForwardMode(),
    )
    jf = jax.jacfwd(_loss(whole(fwd)))(TS)
    np.testing.assert_allclose(jf, g, rtol=1e-10)


def test_times_coincident_point() -> None:
    """Releases on a lattice point (and at t1) run and are differentiable."""
    lattice = gde.Uniform(jnp.asarray(300.0), 31, 1).lattice()  # h = 10 Myr
    ts = jnp.asarray([-300.0, float(lattice[7]), -123.4, 0.0])
    sched = gde.Times(ts, 30, T1)
    assert jnp.sum(sched.grid() == lattice[7]) == 2
    res = whole()(KEY, HOST, W_TODAY, MSAT, sched)
    for a in (res.lead, res.trail, res.progenitor):
        assert jnp.all(jnp.isfinite(a))
    np.testing.assert_allclose(res.progenitor, W_TODAY, rtol=0, atol=1e-12)
    # The release at t1 is its release state.
    _, lead0, trail0 = release_ics(sched)
    np.testing.assert_allclose(res.lead[-1], lead0[-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(res.trail[-1], trail0[-1], rtol=0, atol=1e-13)

    def loss(ts: jax.Array) -> jax.Array:
        r = whole()(KEY, HOST, W_TODAY, MSAT, gde.Times(ts, 30, T1))
        return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

    assert jnp.all(jnp.isfinite(jax.grad(loss)(ts)))


def test_times_no_leak() -> None:
    """Each tracer moves as a lone two-row solve from its release on the grid.

    The row pair (tracer, progenitor copy) starts from the release states and
    steps on the merged grid from the release to t1: no half-kick leaks across
    the release.
    """
    sched = gde.Times(TS, 30, T1)
    res = whole()(KEY, HOST, W_TODAY, MSAT, sched)
    ws, lead0, trail0 = release_ics(sched)
    grid, idx = sched.grid(), sched.release_index()
    copies = _CopiesField(HOST, plummer(MSAT))
    for i in range(TS.shape[0]):
        j = int(idx[i])
        sub = grid[j:]
        terms = copies.terms(LEAP)
        for got, w0 in ((res.lead[i], lead0[i]), (res.trail[i], trail0[i])):
            y0 = (jnp.stack([w0[:3], ws[j, :3]]), jnp.stack([w0[3:], ws[j, 3:]]))
            want = jnp.concat(solve_on(LEAP, terms, y0, sub), axis=-1)
            np.testing.assert_allclose(got, want[0], rtol=0, atol=1e-11)
    np.testing.assert_allclose(res.progenitor, W_TODAY, rtol=0, atol=1e-12)


@pytest.mark.parametrize("solver", list(SOLVERS))
def test_times_unsorted(solver) -> None:
    """Shuffled release times give the same per-release results, permuted.

    The DF's draws follow the position in ``ts``, so this feeds the same
    release states in, permuted.
    """
    s = SOLVERS[solver]
    perm = jnp.asarray([3, 0, 4, 2, 1])
    sched, schedp = gde.Times(TS, 30, T1), gde.Times(TS[perm], 30, T1)
    ws, lead0, trail0 = release_ics(sched, s)
    wsp = progenitor_states(HOST, W_TODAY, schedp, s)
    np.testing.assert_allclose(
        wsp[schedp.release_index()], ws[sched.release_index()][perm], atol=1e-13
    )
    sim = whole(s)
    args = (sim, HOST, plummer(MSAT))
    out = _evolve(sim.mode, *args, sched, W_TODAY, ws, lead0, trail0)
    outp = _evolve(sim.mode, *args, schedp, W_TODAY, wsp, lead0[perm], trail0[perm])
    np.testing.assert_allclose(outp[0], out[0][perm], rtol=0, atol=1e-12)
    np.testing.assert_allclose(outp[1], out[1][perm], rtol=0, atol=1e-12)
    np.testing.assert_allclose(outp[2], out[2], rtol=0, atol=1e-12)


PER = {
    "copies-vmap": gde.PerParticle(gde.Copies()),
    "copies-scan": gde.PerParticle(gde.Copies(), lstrat.Scan),
    "track-vmap": gde.PerParticle(gde.SharedTrack()),
    "track-scan": gde.PerParticle(gde.SharedTrack(), lstrat.Scan),
}


def per(mode, solver=LEAP, prog=plummer) -> gde.SpraySimulator:
    return gde.SpraySimulator(
        gde.Fardal2015DF(), PER[mode], progenitor_potential=prog, solver=solver
    )


ALL = {"whole": gde.WholeStream(), **PER}


def any_mode(mode, solver=LEAP) -> gde.SpraySimulator:
    return gde.SpraySimulator(
        gde.Fardal2015DF(), ALL[mode], progenitor_potential=plummer, solver=solver
    )


def flat(r) -> jax.Array:
    return jnp.concat([r.lead, r.trail, r.progenitor[None]])


@pytest.mark.parametrize("solver", list(SOLVERS))
@pytest.mark.parametrize("mode", list(ALL))
def test_zero_span(mode, solver) -> None:
    """Schedules over zero time run, eagerly and under jit, and never raise.

    One release with the default ``t1 = max(ts)``, or a `Uniform` with
    ``t_age = 0``: every release is at ``t1``, so the stream is its release
    states.
    """
    s = SOLVERS[solver]
    sim = any_mode(mode, s)
    cases = (
        (lambda ts: gde.Times(ts, 10), jnp.asarray([-100.0])),
        (lambda a: gde.Uniform(a, 4, 5), jnp.asarray(0.0)),
    )
    for make, x in cases:
        _, lead0, trail0 = release_ics(make(x), s)

        def run(x, make=make):
            return flat(sim(KEY, HOST, W_TODAY, MSAT, make(x)))

        for got in (run(x), jax.jit(run)(x)):
            n = lead0.shape[0]
            np.testing.assert_allclose(got[:n], lead0, rtol=0, atol=1e-13)
            np.testing.assert_allclose(got[n : 2 * n], trail0, rtol=0, atol=1e-13)
            np.testing.assert_allclose(got[-1], W_TODAY, rtol=0, atol=1e-13)


@pytest.mark.parametrize("mode", list(ALL))
def test_release_after_t1_is_nan(mode) -> None:
    """A release time after ``t1`` gives NaN under jit, not a wrong stream."""
    sim = any_mode(mode)

    def run(ts):
        return flat(sim(KEY, HOST, W_TODAY, MSAT, gde.Times(ts, 20, T1)))

    got = jax.jit(run)(jnp.asarray([-200.0, -50.0, 40.0]))
    assert jnp.all(jnp.isnan(got))


@pytest.mark.parametrize("prog", ["plummer", "none"])
@pytest.mark.parametrize("mode", list(PER))
def test_times_per_particle_matches_whole_stream(mode, prog) -> None:
    """Per-particle `Times` with Leapfrog equals the whole stream.

    Copies and the host alone step on the merged grid like the whole stream,
    so they match it to round-off. The track's knots come from a separate
    Leapfrog solve on the uniform lattice, 2.9e-3 kpc from the merged-grid
    orbit the releases come from, so the track matches only to about that
    (measured 6.5e-3 kpc, 1.2e-4 kpc/Myr); both are well inside Leapfrog's own
    error at this step (about 2e-2 kpc against an adaptive reference).
    """
    pp = plummer if prog == "plummer" else None
    sched = gde.Times(TS, 30, T1)
    sim_w = gde.SpraySimulator(
        gde.Fardal2015DF(), gde.WholeStream(), progenitor_potential=pp
    )
    want = sim_w(KEY, HOST, W_TODAY, MSAT, sched)
    got = per(mode, prog=pp)(KEY, HOST, W_TODAY, MSAT, sched)
    exact = mode.startswith("copies") or pp is None
    xtol, vtol = (1e-11, 1e-11) if exact else (1e-2, 1e-3)
    for g, w in ((got.lead, want.lead), (got.trail, want.trail)):
        assert jnp.all(jnp.isfinite(g))
        np.testing.assert_allclose(g[:, :3], w[:, :3], rtol=0, atol=xtol)
        np.testing.assert_allclose(g[:, 3:], w[:, 3:], rtol=0, atol=vtol)
    np.testing.assert_array_equal(got.release_times, want.release_times)
    np.testing.assert_allclose(got.progenitor, W_TODAY, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", ["copies-vmap", "track-vmap", "track-scan"])
def test_times_per_particle_grad(mode) -> None:
    """Release-time gradients match finite differences and `jax.jacfwd`."""
    loss = _loss(per(mode))
    g = jax.grad(loss)(TS)
    eps = 1e-4
    fd = jnp.stack(
        [
            (loss(TS.at[i].add(eps)) - loss(TS.at[i].add(-eps))) / (2 * eps)
            for i in range(TS.shape[0])
        ]
    )
    assert jnp.all(g != 0), g
    np.testing.assert_allclose(g, fd, rtol=1e-5)
    fwd = dfxtra.DiffEqSolver(
        gf.Leapfrog(),
        stepsize_controller=dfx.ConstantStepSize(),
        adjoint=dfx.ForwardMode(),
    )
    jf = jax.jacfwd(_loss(per(mode, fwd)))(TS)
    np.testing.assert_allclose(jf, g, rtol=1e-10)


@pytest.mark.parametrize("mode", ["copies-vmap", "track-vmap"])
def test_times_per_particle_adaptive(mode) -> None:
    """Adaptive per-particle solves start at each release time.

    They match a tight whole-stream reference.
    """
    sched = gde.Times(TS, 30, T1)
    ref = whole(adaptive(1e-13, 100_000))(KEY, HOST, W_TODAY, MSAT, sched)
    got = per(mode, adaptive(1e-10, 10_000))(KEY, HOST, W_TODAY, MSAT, sched)
    d = jnp.linalg.norm(
        jnp.concat([got.lead - ref.lead, got.trail - ref.trail])[:, :3], axis=1
    )
    assert jnp.max(d) < 1e-6, d


@pytest.mark.parametrize("solver", list(SOLVERS))
@pytest.mark.parametrize("mode", ["copies-vmap", "track-scan"])
def test_per_particle_nan_time(mode, solver) -> None:
    """A NaN release time or ``t_age`` gives NaN under jit(vmap).

    It never raises, for `Times` and for `Uniform`.
    """
    sim = per(mode, SOLVERS[solver])

    def run_times(ts):
        r = sim(KEY, HOST, W_TODAY, MSAT, gde.Times(ts, 30, T1))
        return jnp.concat([r.lead, r.trail])

    def run_uniform(t_age):
        r = sim(KEY, HOST, W_TODAY, MSAT, gde.Uniform(t_age, 4, 5))
        return jnp.concat([r.lead, r.trail])

    got = jax.jit(jax.vmap(run_times))(jnp.stack([TS, TS.at[2].set(jnp.nan)]))
    assert jnp.all(jnp.isfinite(got[0]))
    assert jnp.all(jnp.isnan(got[1]))
    got = jax.jit(jax.vmap(run_uniform))(jnp.asarray([300.0, jnp.nan]))
    assert jnp.all(jnp.isfinite(got[0]))
    assert jnp.all(jnp.isnan(got[1]))


def test_batched_t1_raises() -> None:
    sim = gde.SpraySimulator(gde.Fardal2015DF(), gde.WholeStream())
    with pytest.raises(ValueError, match="t1"):
        sim(
            KEY,
            HOST,
            W_TODAY,
            MSAT,
            gde.Uniform(jnp.asarray(300.0), 2, 5, jnp.zeros(2)),
        )
    with pytest.raises(ValueError, match="t1"):
        gde.Times(TS, 30, jnp.zeros(2))


@pytest.mark.parametrize(
    "sched", [gde.Uniform(jnp.asarray(300.0), 4, 5), gde.Times(TS, 30)]
)
def test_list_w_today(sched) -> None:
    sim = whole()
    want = sim(KEY, HOST, W_TODAY, MSAT, sched)
    got = sim(KEY, HOST, [float(x) for x in W_TODAY], MSAT, sched)
    np.testing.assert_array_equal(got.lead, want.lead)


@pytest.mark.parametrize("solver", list(SOLVERS))
def test_times_vmap_and_nan(solver) -> None:
    """Under jit(vmap) over ``ts``: equal to separate calls; NaN gives NaN.

    Batched and unbatched round-off differ in the last bits, which adaptive
    step control amplifies to about its tolerance (as with `Uniform`), so the
    adaptive comparison is at 1e-7. A NaN lane leaves the others bitwise as
    they are.
    """
    sim = whole(SOLVERS[solver])
    atol = 1e-12 if solver == "leapfrog" else 1e-7

    def run(ts):
        r = sim(KEY, HOST, W_TODAY, MSAT, gde.Times(ts, 30, T1))
        return jnp.concat([r.lead, r.trail, r.progenitor[None]])

    tss = jnp.stack([TS, TS + 3.1, TS.at[2].set(jnp.nan)])
    got = jax.jit(jax.vmap(run))(tss)
    for k in range(2):
        np.testing.assert_allclose(got[k], run(tss[k]), rtol=0, atol=atol)
    np.testing.assert_array_equal(got[:2], jax.jit(jax.vmap(run))(tss[:2]))
    assert jnp.all(jnp.isnan(got[2]))
    assert jnp.all(jnp.isnan(run(tss[2])))


# A host whose mass grows in time, with t1 != 0: a stream from the wrong force
# times (an offset in the time the host is evaluated at) shows up here.
TD_HOST = gp.NFWPotential(
    m=gp.params.LinearParameter(
        slope=u.Q(5e8, "Msun/Myr"),
        point_time=u.Q(0.0, "Myr"),
        point_value=u.Q(1e12, "Msun"),
    ),
    r_s=15.0,
    units=USYS,
)
TD_T1 = jnp.asarray(250.0)
TD_MODES = {
    "whole": gde.WholeStream(),
    "copies": gde.PerParticle(gde.Copies()),
    "track": gde.PerParticle(gde.SharedTrack()),
}


def td_schedule(kind: str, t1=TD_T1):
    if kind == "uniform":
        return gde.Uniform(jnp.asarray(1000.0), 16, 1, t1)
    ts = t1 + jnp.asarray([-1000.0, -873.2, -640.9, -401.7, -222.2, -35.6])
    return gde.Times(ts, 30, t1)


@ft.cache
def td_legacy(kind: str):
    """Return the legacy generator's solves from our release states to ``t1``."""
    solver = adaptive(1e-7, 2**16)
    sched = td_schedule(kind)
    ws = progenitor_states(TD_HOST, W_TODAY, sched, solver)
    idx, ts = sched.release_index(), sched.release_times()
    xl, vl, xt, vt = sample_releases(
        gde.Chen2024DF(),
        KEY,
        TD_HOST,
        ts,
        ws[idx, :3],
        ws[idx, 3:],
        jnp.broadcast_to(MSAT, ts.shape),
    )
    w0s = jnp.stack(
        [jnp.concat([xl, vl], axis=-1), jnp.concat([xt, vt], axis=-1)], axis=1
    )
    gen = gd.MockStreamGenerator(
        gd.ChenStreamDF(), TD_HOST, progenitor_potential=plummer(MSAT)
    )

    def pair(w0, t0, prog_w0):
        return gen._integrate_to(w0, u.Q(t0, "Myr"), u.Q(sched.t1, "Myr"), prog_w0)

    keep = ts < sched.t1  # a release at t1 is its release state
    return keep, jax.jit(jax.vmap(pair))(w0s[keep], ts[keep], ws[idx][keep])


@pytest.mark.parametrize("kind", ["uniform", "times"])
@pytest.mark.parametrize("mode", list(TD_MODES))
def test_time_dependent_host_matches_legacy(mode, kind) -> None:
    keep, (lead, trail) = td_legacy(kind)
    sim = gde.SpraySimulator(
        gde.Chen2024DF(),
        TD_MODES[mode],
        progenitor_potential=plummer,
        solver=adaptive(1e-7, 2**16),
    )
    res = sim(KEY, TD_HOST, W_TODAY, MSAT, td_schedule(kind))
    d = jnp.linalg.norm(
        jnp.concat([lead - res.lead[keep], trail - res.trail[keep]])[:, :3], axis=1
    )
    assert jnp.median(d) < 1e-5, d
    assert jnp.max(d) < 1e-4, d


@pytest.mark.parametrize("kind", ["uniform", "times"])
@pytest.mark.parametrize("mode", list(TD_MODES))
def test_time_dependent_host_time_grads(mode, kind) -> None:
    """Gradients in ``t1`` and the release times (``t_age`` or ``ts``)."""
    sim = gde.SpraySimulator(
        gde.Fardal2015DF(), TD_MODES[mode], progenitor_potential=plummer
    )
    base = td_schedule(kind)

    def loss(x: jax.Array) -> jax.Array:
        if kind == "uniform":
            sched = gde.Uniform(x[1], 4, 5, x[0])
        else:
            sched = gde.Times(x[1:], 30, x[0])
        r = sim(KEY, TD_HOST, W_TODAY, MSAT, sched)
        return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

    if kind == "uniform":
        x = jnp.stack([TD_T1, base.t_age])
    else:
        x = jnp.concat([TD_T1[None], base.ts])
    g = jax.grad(loss)(x)
    eps = 1e-4
    fd = jnp.stack(
        [
            (loss(x.at[i].add(eps)) - loss(x.at[i].add(-eps))) / (2 * eps)
            for i in range(x.shape[0])
        ]
    )
    assert jnp.all(g != 0), g
    np.testing.assert_allclose(g, fd, rtol=1e-5)


ON_LATTICE = [
    *[("leapfrog", m, h) for m in ALL for h in ("static", "td")],
    ("adaptive", "whole", "static"),
]


@pytest.mark.parametrize(("solver", "mode", "host"), ON_LATTICE)
def test_times_on_lattice_equals_uniform(solver, mode, host) -> None:
    """`Times` on `Uniform`'s release times gives `Uniform`'s stream.

    The time-dependent host, with ``t1 != 0``, catches a term that is given
    the grid index instead of the physical time.
    """
    pot, t1 = (HOST, T1) if host == "static" else (TD_HOST, TD_T1)
    uni = gde.Uniform(jnp.asarray(300.0), 5, 8, t1)
    times = gde.Times(uni.release_times(), uni.n_steps(), uni.t1)
    sim = any_mode(mode, SOLVERS[solver])
    a = sim(KEY, pot, W_TODAY, MSAT, uni)
    b = sim(KEY, pot, W_TODAY, MSAT, times)
    for x, y in ((a.lead, b.lead), (a.trail, b.trail), (a.progenitor, b.progenitor)):
        assert jnp.all(jnp.isfinite(x))
        np.testing.assert_allclose(y, x, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(b.release_times, a.release_times)


def test_times_fixed_step_runge_kutta() -> None:
    """With a fixed-step Runge-Kutta solver (Tsit5), copies equal the stream.

    Both step on the merged grid, so a copy takes the progenitor row's steps.
    """
    tsit = dfxtra.DiffEqSolver(
        dfx.Tsit5(),
        stepsize_controller=dfx.ConstantStepSize(),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    sched = gde.Times(TS, 30, T1)
    want = flat(any_mode("whole", tsit)(KEY, HOST, W_TODAY, MSAT, sched))
    got = flat(any_mode("copies-vmap", tsit)(KEY, HOST, W_TODAY, MSAT, sched))
    assert jnp.all(jnp.isfinite(want))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-11)
