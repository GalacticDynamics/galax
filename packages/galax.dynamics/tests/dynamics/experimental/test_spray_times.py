"""Tests for the `Times` schedule (arbitrary release times) in whole-stream mode."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diffraxtra as dfxtra

import galax.dynamics as gd
import galax.fields as gf
from .test_spray_whole_stream import (
    HOST,
    KEY,
    LEAP,
    MSAT,
    W_TODAY,
    adaptive,
    plummer,
)
from galax.dynamics._src.experimental.df import sample_releases
from galax.dynamics._src.experimental.fields import _CopiesField
from galax.dynamics._src.experimental.simulator import _evolve
from galax.dynamics._src.experimental.solve import (
    index_time,
    progenitor_states,
    solve_on,
)

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
    np.testing.assert_array_equal(grid[sched.lattice_index()], sched.lattice())
    np.testing.assert_array_equal(sched.lattice(), sched.uniform().lattice())
    u = gde.Uniform(jnp.asarray(300.0), 4, 5)
    np.testing.assert_array_equal(u.grid(), u.lattice())
    np.testing.assert_array_equal(u.lattice_index(), jnp.arange(16))


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


@pytest.mark.parametrize("solver", list(SOLVERS))
def test_times_on_lattice_equals_uniform(solver) -> None:
    uni = gde.Uniform(jnp.asarray(300.0), 5, 8)
    times = gde.Times(uni.release_times(), uni.n_steps(), uni.t1)
    sim = whole(SOLVERS[solver])
    a = sim(KEY, HOST, W_TODAY, MSAT, uni)
    b = sim(KEY, HOST, W_TODAY, MSAT, times)
    for x, y in ((a.lead, b.lead), (a.trail, b.trail), (a.progenitor, b.progenitor)):
        assert jnp.all(jnp.isfinite(x))
        np.testing.assert_allclose(y, x, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(b.release_times, a.release_times)


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
        terms = index_time(copies.terms(LEAP), sub)
        steps = jnp.arange(sub.shape[0], dtype=float)
        for got, w0 in ((res.lead[i], lead0[i]), (res.trail[i], trail0[i])):
            y0 = (jnp.stack([w0[:3], ws[j, :3]]), jnp.stack([w0[3:], ws[j, 3:]]))
            want = jnp.concat(solve_on(LEAP, terms, y0, steps), axis=-1)
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


@pytest.mark.parametrize(
    "mode", [gde.PerParticle(gde.Copies()), gde.PerParticle(gde.SharedTrack())]
)
def test_times_per_particle_raises(mode) -> None:
    sim = gde.SpraySimulator(gde.Fardal2015DF(), mode, progenitor_potential=plummer)
    with pytest.raises(NotImplementedError, match="Times"):
        sim(KEY, HOST, W_TODAY, MSAT, gde.Times(TS, 30, T1))


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
