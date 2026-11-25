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
    release_ics,
)
from galax.dynamics._src.experimental.fields import _CopiesField
from galax.dynamics._src.experimental.simulator import _evolve
from galax.dynamics._src.experimental.solve import progenitor_states, solve_on


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Drop compiled programs after each test: they add up to gigabytes here."""
    yield
    jax.clear_caches()


gde = gd.experimental
T1 = jnp.asarray(0.0)
# Off the 30-step lattice over [-290.3, 0] (h ~ 9.68 Myr), except the first.
TS = jnp.asarray([-290.3, -201.7, -150.2, -42.9, -12.5])
SOLVERS = {"leapfrog": LEAP, "adaptive": adaptive(1e-10, 10_000)}
LEAP_FWD = dfxtra.DiffEqSolver(
    gf.Leapfrog(),
    stepsize_controller=dfx.ConstantStepSize(),
    adjoint=dfx.ForwardMode(),
)
FARDAL = gde.Fardal2015DF()
ALL = {
    "whole": gde.WholeStream(),
    "copies-vmap": gde.PerParticle(gde.Copies()),
    "copies-scan": gde.PerParticle(gde.Copies(), lstrat.Scan),
    "track-vmap": gde.PerParticle(gde.SharedTrack()),
    "track-scan": gde.PerParticle(gde.SharedTrack(), lstrat.Scan),
}
PER = [m for m in ALL if m != "whole"]


def make_sim(mode="whole", solver=LEAP, *, prog=plummer, df=FARDAL):
    return gde.SpraySimulator(df, ALL[mode], progenitor_potential=prog, solver=solver)


def flat(r) -> jax.Array:
    return jnp.concat([r.lead, r.trail, r.progenitor[None]])


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


# Loss weights, one row per output state (up to 8 releases).
LOSS_W = jax.random.normal(jax.random.key(7), (2 * 8 + 1, 6))


def wloss(sim, make=lambda ts: gde.Times(ts, 30, T1), pot=HOST):
    """Return a weighted sum of every output state, as a function of ``x``."""

    def loss(x: jax.Array) -> jax.Array:
        f = flat(sim(KEY, pot, W_TODAY, MSAT, make(x)))
        return jnp.sum(f * LOSS_W[: f.shape[0]])

    return loss


def check_grad(loss, x, eps=1e-3, rtol=1e-5):
    """Check ``jax.grad(loss)`` at ``x`` against central differences."""
    x = jnp.asarray(x)
    g = jax.grad(loss)(x)
    fd = [
        (loss(x.at[i].add(eps)) - loss(x.at[i].add(-eps))) / (2 * eps)
        for i in range(x.shape[0])
    ]
    np.testing.assert_allclose(g, jnp.stack(fd), rtol=rtol)
    return g


@pytest.mark.parametrize("mode", ["whole", "copies-vmap", "track-vmap", "track-scan"])
def test_times_grad(mode) -> None:
    """Release-time gradients match `jax.jacfwd` with `ForwardMode`.

    Finite differences check these gradients in a time-dependent host, in
    test_time_dependent_host_time_grads.
    """
    g = jax.grad(wloss(make_sim(mode)))(TS)
    jf = jax.jacfwd(wloss(make_sim(mode, LEAP_FWD)))(TS)
    np.testing.assert_allclose(jf, g, rtol=1e-10)


# The lattice of `Times(ts, 30, T1)` with min(ts) = -300: h = 10 Myr.
TIE_LATTICE = gde.Uniform(jnp.asarray(300.0), 31, 1).lattice()
L7 = float(TIE_LATTICE[7])
# Every kind of tie, as (ts, t1); t1 None is the default, max(ts).
TIES = {
    # Pairs at min(ts), on a lattice point, off the lattice and at t1.
    "explicit-t1": ([-300.0, -300.0, L7, L7, -123.4, -123.4, 0.0, 0.0], T1),
    # A pair at max(ts) that sets t1 (L7 is off this lattice).
    "default-t1": ([-300.0, L7, -123.4, -12.5, -12.5], None),
    # Uniform's release times: every one on a lattice point.
    "uniform": (gde.Uniform(jnp.asarray(300.0), 4, 10).release_times(), T1),
}


def untie(sched, eps=1e-6):
    """Return a shift of ``sched.ts`` that unties it in the grid's order.

    Of ``m`` equal times, the ``k``-th (in the order of ``ts``) moves left by
    ``m - 1 - k`` units of ``eps``, and by one more if they sit on a lattice
    point, so each release stays before the lattice point it equalled. At
    ``min(ts)``, which sets the lattice, the first stays and the ``k``-th
    moves right by ``k`` units instead, so the lattice stays where it is.
    """
    ts, lat = np.asarray(sched.ts), np.asarray(sched.lattice())
    d = np.zeros_like(ts)
    for v in np.unique(ts):
        (idx,) = np.nonzero(ts == v)
        k = np.arange(len(idx))
        d[idx] = k if v == ts.min() else -(k[::-1] + np.any(lat == v))
    return eps * jnp.asarray(d)


def test_times_coincident_point() -> None:
    """Releases on a lattice point (and at t1) run, and sort before it."""
    ts = [-300.0, L7, -123.4, 0.0]
    sched = gde.Times(jnp.asarray(ts), 30, T1)
    assert jnp.sum(sched.grid() == L7) == 2
    # The release sorts before the lattice point, and the grid ends at t1.
    i = int(sched.release_index()[1])
    assert sched.grid()[i] == sched.grid()[i + 1] == L7
    assert int(jnp.searchsorted(sched.grid(), L7)) == i
    assert int(sched.release_index()[-1]) == sched.grid().shape[0] - 2
    res = make_sim()(KEY, HOST, W_TODAY, MSAT, sched)
    for a in (res.lead, res.trail, res.progenitor):
        assert jnp.all(jnp.isfinite(a))
    np.testing.assert_allclose(res.progenitor, W_TODAY, rtol=0, atol=1e-12)
    # The release at t1 is its release state.
    _, lead0, trail0 = release_ics(sched)
    np.testing.assert_allclose(res.lead[-1], lead0[-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(res.trail[-1], trail0[-1], rtol=0, atol=1e-13)


TIE_RUNS = [
    ("whole", "leapfrog"),
    ("whole", "adaptive"),
    ("copies-vmap", "leapfrog"),
    ("copies-vmap", "adaptive"),
    ("track-vmap", "leapfrog"),
]


@pytest.mark.parametrize("case", list(TIES))
@pytest.mark.parametrize(("mode", "solver"), TIE_RUNS)
def test_times_tie_grad(mode, solver, case) -> None:
    """At a tie, the gradient is the limit of the gradients at nearby untied times.

    With fixed steps the derivative jumps at a tie. Here every wrong order
    jumps by more than the 1e-4 tolerance, while a 1e-6 Myr shift moves the
    gradient far less, so the comparison tells a right limit from a wrong one.
    An adaptive solve has no jump at a tie, so its runs check only that the
    gradient is continuous there.
    For `Uniform`'s release times, the gradient also gives `Uniform`'s
    ``t_age`` gradient by the chain rule.
    """
    ts, t1 = TIES[case]
    ts = jnp.asarray(ts)
    sim = make_sim(mode, SOLVERS[solver])
    grad = jax.grad(wloss(sim, lambda x: gde.Times(x, 30, t1)))
    shifted = ts + untie(gde.Times(ts, 30, t1))
    # The shift unties the times without changing the grid's order.
    np.testing.assert_array_equal(
        gde.Times(shifted, 30, t1).release_index(),
        gde.Times(ts, 30, t1).release_index(),
    )
    g, near = grad(ts), grad(shifted)
    np.testing.assert_allclose(g, near, rtol=0, atol=1e-4 * jnp.max(jnp.abs(near)))
    if case == "uniform":
        loss = wloss(sim, lambda t_age: gde.Uniform(t_age, 4, 10, T1))
        # d ts / d t_age, from ts = t1 - t_age + 10 k h with h = t_age / 30.
        dts = -jnp.asarray([1.0, 2 / 3, 1 / 3, 0.0])
        np.testing.assert_allclose(
            g @ dts, jax.grad(loss)(jnp.asarray(300.0)), rtol=1e-9
        )


def test_times_no_leak() -> None:
    """Each tracer moves as a lone two-row solve from its release on the grid.

    The row pair (tracer, progenitor copy) starts from the release states and
    steps on the merged grid from the release to t1: no half-kick leaks across
    the release.
    """
    sched = gde.Times(TS, 30, T1)
    res = make_sim()(KEY, HOST, W_TODAY, MSAT, sched)
    ws, lead0, trail0 = release_ics(sched)
    grid, idx = sched.grid(), sched.release_index()
    terms = _CopiesField(HOST, plummer(MSAT)).terms(LEAP)
    for i in range(TS.shape[0]):
        j = int(idx[i])
        sub = grid[j:]
        for got, w0 in ((res.lead[i], lead0[i]), (res.trail[i], trail0[i])):
            y0 = (jnp.stack([w0[:3], ws[j, :3]]), jnp.stack([w0[3:], ws[j, 3:]]))
            want = jnp.concat(solve_on(LEAP, terms, y0, sub), axis=-1)
            np.testing.assert_allclose(got, want[0], rtol=0, atol=1e-11)
    np.testing.assert_allclose(res.progenitor, W_TODAY, rtol=0, atol=1e-12)


@pytest.mark.parametrize("solver", list(SOLVERS.values()), ids=list(SOLVERS))
def test_times_unsorted(solver) -> None:
    """Shuffled release times give the same per-release results, permuted.

    The DF's draws follow the position in ``ts``, so this feeds the same
    release states in, permuted.
    """
    perm = jnp.asarray([3, 0, 4, 2, 1])
    sched, schedp = gde.Times(TS, 30, T1), gde.Times(TS[perm], 30, T1)
    ws, lead0, trail0 = release_ics(sched, solver)
    wsp = progenitor_states(HOST, W_TODAY, schedp, solver)
    np.testing.assert_allclose(
        wsp[schedp.release_index()], ws[sched.release_index()][perm], atol=1e-13
    )
    sim = make_sim(solver=solver)
    args = (sim, HOST, plummer(MSAT))
    out = _evolve(sim.mode, *args, sched, W_TODAY, ws, lead0, trail0)
    outp = _evolve(sim.mode, *args, schedp, W_TODAY, wsp, lead0[perm], trail0[perm])
    np.testing.assert_allclose(outp[0], out[0][perm], rtol=0, atol=1e-12)
    np.testing.assert_allclose(outp[1], out[1][perm], rtol=0, atol=1e-12)
    np.testing.assert_allclose(outp[2], out[2], rtol=0, atol=1e-12)


@pytest.mark.parametrize("solver", list(SOLVERS.values()), ids=list(SOLVERS))
@pytest.mark.parametrize("mode", list(ALL))
def test_zero_span(mode, solver) -> None:
    """Schedules over zero time run, eagerly and under jit, and never raise.

    One release with the default ``t1 = max(ts)``, or a `Uniform` with
    ``t_age = 0``: every release is at ``t1``, so the stream is its release
    states.
    """
    sim = make_sim(mode, solver)
    cases = (
        (lambda ts: gde.Times(ts, 10), jnp.asarray([-100.0])),
        (lambda a: gde.Uniform(a, 4, 5), jnp.asarray(0.0)),
    )
    for make, x in cases:
        _, lead0, trail0 = release_ics(make(x), solver)

        def run(x, make=make):
            return flat(sim(KEY, HOST, W_TODAY, MSAT, make(x)))

        for got in (run(x), jax.jit(run)(x)):
            n = lead0.shape[0]
            np.testing.assert_allclose(got[:n], lead0, rtol=0, atol=1e-13)
            np.testing.assert_allclose(got[n : 2 * n], trail0, rtol=0, atol=1e-13)
            np.testing.assert_allclose(got[-1], W_TODAY, rtol=0, atol=1e-13)


@pytest.mark.parametrize("solver", list(SOLVERS))
@pytest.mark.parametrize("mode", ["whole", "copies-vmap"])
def test_zero_span_grad(mode, solver) -> None:
    """Gradients over zero time are the one-sided derivatives into the domain.

    A `Uniform` at ``t_age = 0`` can only grow ``t_age``; of two releases at an
    explicit ``t1``, the first moving left keeps the grid's order. An adaptive
    solve takes no steps there, so its tangents come from `_solve`'s
    first-order term; without it they were far outside the tolerance.
    """
    sim = make_sim(mode, SOLVERS[solver])
    e = 1e-4
    loss = wloss(sim, lambda a: gde.Uniform(a, 4, 5, T1))
    a = jnp.asarray(0.0)
    want = (loss(a + e) - loss(a)) / e
    np.testing.assert_allclose(jax.grad(loss)(a), want, rtol=1e-3)
    loss = wloss(sim, lambda ts: gde.Times(ts, 30, jnp.asarray(-100.0)))
    ts = jnp.asarray([-100.0, -100.0])
    want = (loss(ts) - loss(ts.at[0].add(-e))) / e
    np.testing.assert_allclose(jax.grad(loss)(ts)[0], want, rtol=1e-3)


@pytest.mark.parametrize("prog", ["plummer", "none"])
@pytest.mark.parametrize("mode", PER)
def test_times_per_particle_matches_whole_stream(mode, prog) -> None:
    """Per-particle `Times` with Leapfrog equals the whole stream.

    Copies and the host alone step on the merged grid like the whole stream,
    so they match it to round-off. The track's knots come from a separate
    Leapfrog solve on the uniform lattice, a slightly different orbit from the
    merged-grid one the releases come from, so the track matches only to the
    tolerances below (1e-2 kpc, 1e-3 kpc/Myr).
    """
    pp = plummer if prog == "plummer" else None
    sched = gde.Times(TS, 30, T1)
    want = make_sim(prog=pp)(KEY, HOST, W_TODAY, MSAT, sched)
    got = make_sim(mode, prog=pp)(KEY, HOST, W_TODAY, MSAT, sched)
    exact = mode.startswith("copies") or pp is None
    xtol, vtol = (1e-11, 1e-11) if exact else (1e-2, 1e-3)
    for g, w in ((got.lead, want.lead), (got.trail, want.trail)):
        assert jnp.all(jnp.isfinite(g))
        np.testing.assert_allclose(g[:, :3], w[:, :3], rtol=0, atol=xtol)
        np.testing.assert_allclose(g[:, 3:], w[:, 3:], rtol=0, atol=vtol)
    np.testing.assert_array_equal(got.release_times, want.release_times)
    np.testing.assert_allclose(got.progenitor, W_TODAY, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", ["copies-vmap", "track-vmap"])
def test_times_per_particle_adaptive(mode) -> None:
    """Adaptive per-particle solves start at each release time.

    They match a tight whole-stream reference.
    """
    sched = gde.Times(TS, 30, T1)
    ref = make_sim(solver=adaptive(1e-13, 100_000))(KEY, HOST, W_TODAY, MSAT, sched)
    got = make_sim(mode, adaptive(1e-10, 10_000))(KEY, HOST, W_TODAY, MSAT, sched)
    d = jnp.linalg.norm(
        jnp.concat([got.lead - ref.lead, got.trail - ref.trail])[:, :3], axis=1
    )
    assert jnp.max(d) < 1e-6, d


def test_list_w_today() -> None:
    sim, sched = make_sim(), gde.Times(TS, 30)
    want = sim(KEY, HOST, W_TODAY, MSAT, sched)
    got = sim(KEY, HOST, [float(x) for x in W_TODAY], MSAT, sched)
    np.testing.assert_array_equal(got.lead, want.lead)


@pytest.mark.parametrize("solver", list(SOLVERS))
@pytest.mark.parametrize("mode", list(ALL))
def test_vmap_and_bad_points(mode, solver) -> None:
    """Under jit(vmap), good lanes equal separate calls; bad lanes give NaN.

    The bad lanes are a NaN release time, a release after t1, and a NaN or
    negative ``t_age``. They never raise (that they don't grind to
    ``max_steps`` is checked in test_spray_per_particle.py and
    test_spray_whole_stream.py), and leave the good
    lanes as they are, to round-off. Batch size changes round-off in the last
    bits, which adaptive step control amplifies to about its tolerance, so the
    adaptive comparisons are at 1e-7.
    """
    sim = make_sim(mode, SOLVERS[solver])
    atol = 1e-12 if solver == "leapfrog" else 1e-7
    bad_ts = [TS.at[2].set(jnp.nan), TS.at[2].set(40.0)]
    cases = (
        (lambda ts: gde.Times(ts, 30, T1), jnp.stack([TS, TS + 3.1, *bad_ts])),
        (
            lambda t_age: gde.Uniform(t_age, 4, 5),
            jnp.asarray([300.0, 250.0, jnp.nan, -300.0]),
        ),
    )
    for make, xs in cases:

        def run(x, make=make):
            return flat(sim(KEY, HOST, W_TODAY, MSAT, make(x)))

        got = jax.jit(jax.vmap(run))(xs)
        for k in range(2):
            np.testing.assert_allclose(got[k], run(xs[k]), rtol=0, atol=atol)
        alone = jax.jit(jax.vmap(run))(xs[:2])
        np.testing.assert_allclose(got[:2], alone, rtol=0, atol=atol)
        assert jnp.all(jnp.isnan(got[2:]))


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
TD_MODES = ["whole", "copies-vmap", "track-vmap"]  # keys of ALL


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
    ws, lead0, trail0 = release_ics(sched, solver, TD_HOST, gde.Chen2024DF())
    idx, ts = sched.release_index(), sched.release_times()
    w0s = jnp.stack([lead0, trail0], axis=1)
    gen = gd.MockStreamGenerator(
        gd.ChenStreamDF(), TD_HOST, progenitor_potential=plummer(MSAT)
    )

    def pair(w0, t0, prog_w0):
        return gen._integrate_to(w0, u.Q(t0, "Myr"), u.Q(sched.t1, "Myr"), prog_w0)

    keep = ts < sched.t1  # a release at t1 is its release state
    return keep, jax.jit(jax.vmap(pair))(w0s[keep], ts[keep], ws[idx][keep])


@pytest.mark.parametrize("kind", ["uniform", "times"])
@pytest.mark.parametrize("mode", TD_MODES)
def test_time_dependent_host_matches_legacy(mode, kind) -> None:
    keep, (lead, trail) = td_legacy(kind)
    sim = make_sim(mode, adaptive(1e-7, 2**16), df=gde.Chen2024DF())
    res = sim(KEY, TD_HOST, W_TODAY, MSAT, td_schedule(kind))
    d = jnp.linalg.norm(
        jnp.concat([lead - res.lead[keep], trail - res.trail[keep]])[:, :3], axis=1
    )
    assert jnp.median(d) < 1e-5, d
    assert jnp.max(d) < 1e-4, d


@pytest.mark.parametrize("kind", ["uniform", "times"])
@pytest.mark.parametrize("mode", [*TD_MODES, "track-scan"])
def test_time_dependent_host_time_grads(mode, kind) -> None:
    """Gradients in ``t1`` and the release times (``t_age`` or ``ts``)."""
    base = td_schedule(kind)
    if kind == "uniform":
        x = jnp.stack([TD_T1, base.t_age])
        make = lambda x: gde.Uniform(x[1], 4, 5, x[0])
    else:
        x = jnp.concat([TD_T1[None], base.ts])
        make = lambda x: gde.Times(x[1:], 30, x[0])
    g = check_grad(wloss(make_sim(mode), make, TD_HOST), x)
    assert jnp.all(g != 0), g


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
    sim = make_sim(mode, SOLVERS[solver])
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
    want = flat(make_sim("whole", tsit)(KEY, HOST, W_TODAY, MSAT, sched))
    got = flat(make_sim("copies-vmap", tsit)(KEY, HOST, W_TODAY, MSAT, sched))
    assert jnp.all(jnp.isfinite(want))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-11)
