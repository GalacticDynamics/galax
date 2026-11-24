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


def release_ics(sched, solver=LEAP, pot=HOST, df=FARDAL):
    ws = progenitor_states(pot, W_TODAY, sched, solver)
    idx = sched.release_index()
    ts = sched.release_times()
    Ms = jnp.broadcast_to(MSAT, ts.shape)
    xl, vl, xt, vt = sample_releases(df, KEY, pot, ts, ws[idx, :3], ws[idx, 3:], Ms)
    return ws, jnp.concat([xl, vl], axis=-1), jnp.concat([xt, vt], axis=-1)


def check_fd(loss, x, eps=1e-4):
    """Check ``jax.grad(loss)`` is nonzero and matches central differences."""
    g = jax.grad(loss)(x)
    fd = jnp.stack(
        [
            (loss(x.at[i].add(eps)) - loss(x.at[i].add(-eps))) / (2 * eps)
            for i in range(x.shape[0])
        ]
    )
    assert jnp.all(g != 0), g
    np.testing.assert_allclose(g, fd, rtol=1e-5)
    return g


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


def _loss(sim, make=lambda ts: gde.Times(ts, 30, T1), pot=HOST):
    def loss(x: jax.Array) -> jax.Array:
        r = sim(KEY, pot, W_TODAY, MSAT, make(x))
        return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

    return loss


@pytest.mark.parametrize("mode", ["whole", "copies-vmap", "track-vmap", "track-scan"])
def test_times_grad(mode) -> None:
    """Release-time gradients match finite differences and `jax.jacfwd`."""
    g = check_fd(_loss(make_sim(mode)), TS)
    jf = jax.jacfwd(_loss(make_sim(mode, LEAP_FWD)))(TS)
    np.testing.assert_allclose(jf, g, rtol=1e-10)


# The lattice of `Times(ts, 30, T1)` with min(ts) = -300: h = 10 Myr.
TIE_LATTICE = gde.Uniform(jnp.asarray(300.0), 31, 1).lattice()
# Release times with ties, and per release the side of the derivative that AD
# must match with fixed steps and with adaptive ones: "L" or "R" one-sided,
# "C" central (no tie). A release sorts before a lattice point it equals, and
# tied releases keep their order in ts, so a release on a lattice point or at
# t1 takes the left derivative, and of two tied releases the first the left
# and the second the right. min(ts) sets the lattice: moving it right keeps
# every other release before its lattice point, so it takes the right one.
TIES = {
    "coincident": ([-300.0, float(TIE_LATTICE[7]), -123.4, 0.0], "RLCL", "CCCL"),
    "pair": ([-300.0, -123.4, -123.4, -12.5], "CLRC", "CLRC"),
    "uniform": (gde.Uniform(jnp.asarray(300.0), 4, 10).release_times(), "RLLL", "CCCL"),
}
TIE_W = jax.random.normal(jax.random.key(7), (2 * 4 + 1, 6))
TIE_EPS = 1e-3
# (rtol, atol) per solver; see `check_tie`.
TIE_TOL = {"leapfrog": (1e-4, 1e-10), "adaptive": (0.0, 5e-5)}


def tie_loss(sim):
    def loss(ts: jax.Array) -> jax.Array:
        f = flat(sim(KEY, HOST, W_TODAY, MSAT, gde.Times(ts, 30, T1)))
        return jnp.sum(f * TIE_W[: f.shape[0]])

    return loss


def check_tie(sim, ts, sides, tol=TIE_TOL["leapfrog"]):
    """Check ``jax.grad`` against second-order one-sided or central differences.

    With Leapfrog, the left and right derivatives at a tie differ by 3.4e-4 to
    1.2 relative (the grid's steps change there). The right side matches AD to
    5.3e-6 relative (measured), so rtol = 1e-4 tells the sides apart. Adaptive
    step control makes the differences noisy, up to 1.8e-5 (0.7%, for min(ts),
    which moves the first solve's start), and an adaptive solve has no kink
    at a tie; the adaptive gradients before the fix were off by 2.5e-3 or
    more, so atol = 5e-5.
    """
    loss = tie_loss(sim)
    ts = jnp.asarray(ts)
    g = jax.grad(loss)(ts)
    f0, e = loss(ts), TIE_EPS

    def fd(i, side):
        def f(k):
            return loss(ts.at[i].add(k * e))

        if side == "C":
            return (f(1) - f(-1)) / (2 * e)
        s = 1 if side == "R" else -1
        return s * (-3 * f0 + 4 * f(s) - f(2 * s)) / (2 * e)

    want = jnp.stack([fd(i, side) for i, side in enumerate(sides)])
    np.testing.assert_allclose(g, want, rtol=tol[0], atol=tol[1])
    return g


def test_times_coincident_point() -> None:
    """Releases on a lattice point (and at t1) run and are differentiable.

    Their gradients are the one-sided derivatives of the grid's order.
    """
    ts, sides, _ = TIES["coincident"]
    sched = gde.Times(jnp.asarray(ts), 30, T1)
    assert jnp.sum(sched.grid() == TIE_LATTICE[7]) == 2
    # The release sorts before the lattice point, and the grid ends at t1.
    i = int(sched.release_index()[1])
    assert sched.grid()[i] == sched.grid()[i + 1] == TIE_LATTICE[7]
    assert int(jnp.searchsorted(sched.grid(), TIE_LATTICE[7])) == i
    assert int(sched.release_index()[-1]) == sched.grid().shape[0] - 2
    res = make_sim()(KEY, HOST, W_TODAY, MSAT, sched)
    for a in (res.lead, res.trail, res.progenitor):
        assert jnp.all(jnp.isfinite(a))
    np.testing.assert_allclose(res.progenitor, W_TODAY, rtol=0, atol=1e-12)
    # The release at t1 is its release state.
    _, lead0, trail0 = release_ics(sched)
    np.testing.assert_allclose(res.lead[-1], lead0[-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(res.trail[-1], trail0[-1], rtol=0, atol=1e-13)
    check_tie(make_sim(), ts, sides)


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
    """Release-time gradients at ties are the right one-sided derivatives.

    For releases on `Uniform`'s, every one on a lattice point, they also give
    `Uniform`'s ``t_age`` gradient by the chain rule.
    """
    if (mode, solver, case) == ("whole", "leapfrog", "coincident"):
        pytest.skip("in test_times_coincident_point")
    ts, fixed, adapt = TIES[case]
    sim = make_sim(mode, SOLVERS[solver])
    g = check_tie(sim, ts, fixed if solver == "leapfrog" else adapt, TIE_TOL[solver])
    if case == "uniform":

        def loss(t_age):
            f = flat(sim(KEY, HOST, W_TODAY, MSAT, gde.Uniform(t_age, 4, 10, T1)))
            return jnp.sum(f * TIE_W[: f.shape[0]])

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


@pytest.mark.parametrize("mode", list(ALL))
def test_release_after_t1_is_nan(mode) -> None:
    """A release time after ``t1`` gives NaN under jit, not a wrong stream."""
    sim = make_sim(mode)

    def run(ts):
        return flat(sim(KEY, HOST, W_TODAY, MSAT, gde.Times(ts, 20, T1)))

    got = jax.jit(run)(jnp.asarray([-200.0, -50.0, 40.0]))
    assert jnp.all(jnp.isnan(got))


@pytest.mark.parametrize("prog", ["plummer", "none"])
@pytest.mark.parametrize("mode", PER)
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


@pytest.mark.parametrize("solver", list(SOLVERS.values()), ids=list(SOLVERS))
@pytest.mark.parametrize("mode", ["copies-vmap", "track-scan"])
def test_per_particle_nan_time(mode, solver) -> None:
    """A NaN release time or ``t_age`` gives NaN under jit(vmap).

    It never raises, for `Times` and for `Uniform`.
    """
    sim = make_sim(mode, solver)
    cases = (
        (lambda ts: gde.Times(ts, 30, T1), jnp.stack([TS, TS.at[2].set(jnp.nan)])),
        (lambda t_age: gde.Uniform(t_age, 4, 5), jnp.asarray([300.0, jnp.nan])),
    )
    for make, xs in cases:

        def run(x, make=make):
            r = sim(KEY, HOST, W_TODAY, MSAT, make(x))
            return jnp.concat([r.lead, r.trail])

        got = jax.jit(jax.vmap(run))(xs)
        assert jnp.all(jnp.isfinite(got[0]))
        assert jnp.all(jnp.isnan(got[1]))


def test_batched_t1_raises() -> None:
    sched = gde.Uniform(jnp.asarray(300.0), 2, 5, jnp.zeros(2))
    with pytest.raises(ValueError, match="t1"):
        make_sim(prog=None)(KEY, HOST, W_TODAY, MSAT, sched)


@pytest.mark.parametrize(
    "sched", [gde.Uniform(jnp.asarray(300.0), 4, 5), gde.Times(TS, 30)]
)
def test_list_w_today(sched) -> None:
    sim = make_sim()
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
    sim = make_sim(solver=SOLVERS[solver])
    atol = 1e-12 if solver == "leapfrog" else 1e-7

    def run(ts):
        return flat(sim(KEY, HOST, W_TODAY, MSAT, gde.Times(ts, 30, T1)))

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
@pytest.mark.parametrize("mode", TD_MODES)
def test_time_dependent_host_time_grads(mode, kind) -> None:
    """Gradients in ``t1`` and the release times (``t_age`` or ``ts``)."""
    base = td_schedule(kind)
    if kind == "uniform":
        x = jnp.stack([TD_T1, base.t_age])
        make = lambda x: gde.Uniform(x[1], 4, 5, x[0])
    else:
        x = jnp.concat([TD_T1[None], base.ts])
        make = lambda x: gde.Times(x[1:], 30, x[0])
    check_fd(_loss(make_sim(mode), make, TD_HOST), x)


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
