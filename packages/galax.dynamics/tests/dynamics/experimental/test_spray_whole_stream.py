"""Tests for `SpraySimulator` in whole-stream mode."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import diffraxtra as dfxtra
import unxt as u

import galax.dynamics as gd
import galax.fields as gf
import galax.potential as gp
from ..mockstream.test_progenitor_potential import gala_stream
from ..optional_deps import OptDeps
from galax.dynamics._src.experimental.df import sample_releases
from galax.dynamics._src.experimental.fields import _CopiesField, _WholeStreamField
from galax.dynamics._src.experimental.simulator import evolve_whole_stream
from galax.dynamics._src.experimental.solve import progenitor_states, solve_on

gde = gd.experimental
USYS = u.unitsystem("galactic")
HOST = gp.NFWPotential(m=1e12, r_s=15.0, units=USYS)
W_TODAY = jnp.concat(
    [
        jnp.asarray([30.0, 10.0, 20.0]),
        u.ustrip("kpc/Myr", u.Q(jnp.asarray([10.0, -150.0, -20.0]), "km/s")),
    ]
)
MSAT = jnp.asarray(1e5)
KEY = jr.key(0)
LEAP = dfxtra.DiffEqSolver(
    gf.Leapfrog(),
    stepsize_controller=dfx.ConstantStepSize(),
    adjoint=dfx.RecursiveCheckpointAdjoint(),
)


def plummer(m: jax.Array) -> gp.AbstractPotential:
    return gp.PlummerPotential(m_tot=m, r_s=4e-3, units=USYS)


def adaptive(tol: float, max_steps: int, adjoint=None) -> dfxtra.DiffEqSolver:
    return dfxtra.DiffEqSolver(
        dfx.Dopri8(scan_kind="bounded"),
        stepsize_controller=dfx.PIDController(rtol=tol, atol=tol),
        adjoint=dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=max_steps,
    )


def release_ics(df, pot, prog, sched, solver):
    """Return the release states the simulator samples (same key and orbit)."""
    ws = progenitor_states(pot, W_TODAY, sched, solver)
    idx = sched.release_index()
    Ms = jnp.broadcast_to(MSAT, (sched.n_release,))
    xl, vl, xt, vt = sample_releases(
        df, KEY, pot, sched.release_times(), ws[idx, :3], ws[idx, 3:], Ms
    )
    return ws, jnp.concat([xl, vl], axis=-1), jnp.concat([xt, vt], axis=-1)


DFS = [gde.Fardal2015DF, gde.Chen2024DF]
PROGS = {"none": None, "potential": plummer(MSAT), "callable": plummer}


@pytest.mark.filterwarnings("ignore:Chen2024DF without a progenitor:RuntimeWarning")
@pytest.mark.parametrize("prog", list(PROGS))
@pytest.mark.parametrize("df_cls", DFS)
def test_whole_stream_runs(df_cls, prog) -> None:
    sched = gde.Uniform(jnp.asarray(500.0), 8, 20)
    sim = gde.SpraySimulator(
        df_cls(), gde.WholeStream(), progenitor_potential=PROGS[prog]
    )
    res = sim(KEY, HOST, W_TODAY, MSAT, sched)
    assert isinstance(res, gde.SprayResult)
    assert res.lead.shape == (8, 6)
    assert res.trail.shape == (8, 6)
    assert res.release_times.shape == (8,)
    assert res.progenitor.shape == (6,)
    for a in (res.lead, res.trail, res.release_times, res.progenitor):
        assert jnp.all(jnp.isfinite(a))
    np.testing.assert_allclose(res.release_times, sched.release_times(), rtol=1e-15)
    np.testing.assert_allclose(res.progenitor, W_TODAY, rtol=0, atol=1e-12)
    assert float(res.t1) == 0.0


def test_gate_no_leak() -> None:
    """A tracer released at lattice point k moves as a lone solve from k.

    And it doesn't move at all before k: no half-kick leaks across the release.
    """
    sched = gde.Uniform(jnp.asarray(400.0), 9, 5)  # 40 steps, segments of 5
    lattice = sched.lattice()
    ws = progenitor_states(HOST, W_TODAY, sched, LEAP)
    prog_pot = plummer(MSAT)
    k = 15  # a release index (segment start)
    # Unbound from the cluster: a bound orbit at 10 Myr steps amplifies round-off.
    w_tr = ws[k] + jnp.asarray([0.1, -0.05, 0.02, 5e-3, -5e-3, 2e-3])
    w0 = jnp.stack([ws[0], w_tr])
    L, n_seg = 5, 8
    starts = jnp.arange(n_seg) * L
    grids = lattice[starts[:, None] + jnp.arange(L + 1)]
    active = (k <= starts)[:, None]
    field = _WholeStreamField(HOST, prog_pot)

    before = evolve_whole_stream(field, LEAP, w0, grids[: k // L], active[: k // L])
    np.testing.assert_array_equal(before[1], w_tr)
    np.testing.assert_allclose(before[0], ws[k], rtol=0, atol=1e-12)

    got = evolve_whole_stream(field, LEAP, w0, grids, active)
    copies = _CopiesField(HOST, prog_pot)
    y0 = (jnp.stack([w_tr[:3], ws[k, :3]]), jnp.stack([w_tr[3:], ws[k, 3:]]))
    want = jnp.concat(solve_on(LEAP, copies.terms(LEAP), y0, lattice[k:]), axis=-1)
    np.testing.assert_allclose(got[1], want[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got[0], want[1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("df_cls", DFS)
def test_whole_stream_matches_lone_solves(df_cls) -> None:
    sched = gde.Uniform(jnp.asarray(300.0), 5, 8)
    df = df_cls()
    sim = gde.SpraySimulator(df, gde.WholeStream(), progenitor_potential=plummer)
    res = sim(KEY, HOST, W_TODAY, MSAT, sched)

    ws, lead0, trail0 = release_ics(df, HOST, plummer, sched, LEAP)
    lattice, idx = sched.lattice(), sched.release_index()
    copies = _CopiesField(HOST, plummer(MSAT))
    terms = copies.terms(LEAP)
    for i in range(sched.n_release - 1):
        for got, w0 in ((res.lead[i], lead0[i]), (res.trail[i], trail0[i])):
            j = int(idx[i])
            y0 = (jnp.stack([w0[:3], ws[j, :3]]), jnp.stack([w0[3:], ws[j, 3:]]))
            want = jnp.concat(solve_on(LEAP, terms, y0, lattice[j:]), axis=-1)[0]
            np.testing.assert_allclose(got[:3], want[:3], rtol=0, atol=1e-11)
    # The release at t1 is its release state.
    np.testing.assert_allclose(res.lead[-1], lead0[-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.trail[-1], trail0[-1], rtol=0, atol=1e-12)


def test_whole_stream_adaptive() -> None:
    sched_lf = gde.Uniform(jnp.asarray(500.0), 8, 500)  # h ~ 0.14 Myr
    sched = gde.Uniform(jnp.asarray(500.0), 8)
    df = gde.Fardal2015DF()
    sim_lf = gde.SpraySimulator(df, gde.WholeStream(), progenitor_potential=plummer)
    sim = gde.SpraySimulator(
        df,
        gde.WholeStream(),
        progenitor_potential=plummer,
        solver=adaptive(1e-10, 10_000),
    )
    lf = sim_lf(KEY, HOST, W_TODAY, MSAT, sched_lf)
    ad = sim(KEY, HOST, W_TODAY, MSAT, sched)
    d = jnp.linalg.norm(
        jnp.concat([lf.lead - ad.lead, lf.trail - ad.trail])[:, :3], axis=1
    )
    assert jnp.median(d) < 1e-4, jnp.median(d)
    np.testing.assert_allclose(ad.progenitor, W_TODAY, rtol=0, atol=1e-8)

    def loss(w: jax.Array) -> jax.Array:
        r = sim(KEY, HOST, w, MSAT, sched)
        return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

    g = jax.grad(loss)(W_TODAY)
    assert jnp.all(jnp.isfinite(g))


def test_vmap_over_keys_and_t_age() -> None:
    sim = gde.SpraySimulator(
        gde.Chen2024DF(), gde.WholeStream(), progenitor_potential=plummer
    )

    def run(key, t_age):
        r = sim(key, HOST, W_TODAY, MSAT, gde.Uniform(t_age, 4, 10))
        return jnp.concat([r.lead, r.trail])

    keys = jr.split(KEY, 3)
    ages = jnp.asarray([200.0, 350.0])
    got = jax.jit(jax.vmap(jax.vmap(run, (0, None)), (None, 0)))(keys, ages)
    assert got.shape == (2, 3, 8, 6)
    for a in range(2):
        for k in range(3):
            np.testing.assert_allclose(
                got[a, k], run(keys[k], ages[a]), rtol=1e-12, atol=1e-12
            )


def test_one_release() -> None:
    sched = gde.Uniform(jnp.asarray(300.0), 1)
    sim = gde.SpraySimulator(
        gde.Fardal2015DF(), gde.WholeStream(), progenitor_potential=plummer
    )
    res = sim(KEY, HOST, W_TODAY, MSAT, sched)
    assert res.lead.shape == (1, 6)
    assert res.trail.shape == (1, 6)
    assert jnp.all(jnp.isfinite(jnp.concat([res.lead, res.trail])))
    np.testing.assert_allclose(res.release_times, [-300.0])
    # The pair moved over the whole age.
    _, lead0, _ = release_ics(gde.Fardal2015DF(), HOST, plummer, sched, LEAP)
    assert jnp.linalg.norm(res.lead[0, :3] - lead0[0, :3]) > 1.0


def test_to_mockstream() -> None:
    sched = gde.Uniform(jnp.asarray(300.0), 4, 5)
    sim = gde.SpraySimulator(gde.Fardal2015DF(), gde.WholeStream())
    res = sim(KEY, HOST, W_TODAY, MSAT, sched)
    mock = gde.to_mockstream(res, USYS)
    assert isinstance(mock, gd.MockStream)
    np.testing.assert_array_equal(mock["lead"].w(units=USYS), res.lead)
    np.testing.assert_array_equal(mock["trail"].w(units=USYS), res.trail)
    np.testing.assert_array_equal(
        u.ustrip("Myr", mock["lead"].release_time), res.release_times
    )
    np.testing.assert_array_equal(u.ustrip("Myr", mock["lead"].t), 0.0)


def _legacy_setup():
    """Chen + 4 pc Plummer, 1 Gyr, 16 releases; default adaptive settings."""
    df = gde.Chen2024DF()
    sched = gde.Uniform(jnp.asarray(1000.0), 16)
    solver = adaptive(1e-7, 2**16)
    sim = gde.SpraySimulator(
        df, gde.WholeStream(), progenitor_potential=plummer, solver=solver
    )
    res = sim(KEY, HOST, W_TODAY, MSAT, sched)
    ws, lead0, trail0 = release_ics(df, HOST, plummer, sched, solver)
    return sched, res, ws, lead0, trail0


def test_matches_legacy_generator() -> None:
    """The legacy generator's per-particle solves from our release states.

    It integrates each lead/trail pair from its release time to our ``t1`` (no
    end-time bump) with its own copy of the progenitor, started from our
    progenitor state at that release, with its default adaptive integrator.
    The release at ``t1`` is left out.
    """
    sched, res, ws, lead0, trail0 = _legacy_setup()
    t1 = u.Q(sched.t1, "Myr")
    gen = gd.MockStreamGenerator(
        gd.ChenStreamDF(), HOST, progenitor_potential=plummer(MSAT)
    )

    def pair(w0, t0, prog_w0):
        return gen._integrate_to(w0, u.Q(t0, "Myr"), t1, prog_w0)

    w0s = jnp.stack([lead0, trail0], axis=1)[:-1]
    prog_w0s = ws[sched.release_index()][:-1]
    lead, trail = jax.jit(jax.vmap(pair))(w0s, sched.release_times()[:-1], prog_w0s)
    d = jnp.linalg.norm(
        jnp.concat([lead - res.lead[:-1], trail - res.trail[:-1]])[:, :3], axis=1
    )
    assert jnp.median(d) < 1e-5, jnp.median(d)
    assert jnp.max(d) < 1e-4, jnp.max(d)


@pytest.mark.slow
@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_matches_gala() -> None:
    """Match gala's mockstream DOP853 from our release states, ending at t1."""
    sched, res, ws, lead0, trail0 = _legacy_setup()
    ts = np.asarray(sched.release_times())[:-1]
    w0 = np.concatenate([lead0[:-1], trail0[:-1]])
    ref = gala_stream(HOST, plummer(MSAT), np.asarray(ws[0]), ts, w0, 0.0)
    got = np.concatenate([res.lead[:-1], res.trail[:-1]])
    d = np.linalg.norm(got[:, :3] - ref[:, :3], axis=1)
    assert np.median(d) < 1e-5, np.median(d)
    assert np.max(d) < 1e-4, np.max(d)


MODES = {
    "whole": gde.WholeStream(),
    "copies": gde.PerParticle(gde.Copies()),
    "track": gde.PerParticle(gde.SharedTrack()),
}


@pytest.mark.parametrize("mode", list(MODES))
def test_chen_without_progenitor_warns(mode) -> None:
    """As the legacy generator: Chen with no progenitor is its "no progenitor" model."""
    with pytest.warns(RuntimeWarning, match="no progenitor") as record:
        sim = gde.SpraySimulator(gde.Chen2024DF(), MODES[mode])
    assert len(record) == 1
    res = sim(KEY, HOST, W_TODAY, MSAT, gde.Uniform(jnp.asarray(300.0), 4, 5))
    assert jnp.all(jnp.isfinite(res.lead))


def test_callable_progenitor_needs_scalar_msat() -> None:
    sched = gde.Uniform(jnp.asarray(300.0), 4, 5)
    sim = gde.SpraySimulator(
        gde.Fardal2015DF(), gde.WholeStream(), progenitor_potential=plummer
    )
    with pytest.raises(ValueError, match="scalar Msat"):
        sim(KEY, HOST, W_TODAY, jnp.full(4, 1e5), sched)
    # A per-release Msat is fine with a fixed potential.
    sim = gde.SpraySimulator(
        gde.Fardal2015DF(), gde.WholeStream(), progenitor_potential=plummer(MSAT)
    )
    res = sim(KEY, HOST, W_TODAY, jnp.full(4, 1e5), sched)
    assert jnp.all(jnp.isfinite(res.lead))


def test_batched_inputs_raise() -> None:
    sim = gde.SpraySimulator(gde.Fardal2015DF(), gde.WholeStream())
    sched = gde.Uniform(jnp.asarray(300.0), 4, 5)
    with pytest.raises(ValueError, match="vmap"):
        sim(KEY, HOST, jnp.stack([W_TODAY, W_TODAY]), MSAT, sched)
    with pytest.raises(ValueError, match="vmap"):
        sim(KEY, HOST, W_TODAY, MSAT, gde.Uniform(jnp.full(2, 300.0), 4, 5))


def test_whole_stream_needs_one_solver() -> None:
    sched = gde.Uniform(jnp.asarray(300.0), 4, 5)
    sim = gde.SpraySimulator(
        gde.Fardal2015DF(),
        gde.WholeStream(),
        progenitor_solver=adaptive(1e-8, 10_000),
    )
    with pytest.raises(ValueError, match="progenitor_solver"):
        sim(KEY, HOST, W_TODAY, MSAT, sched)
    # The same settings as `solver` are fine.
    sim = gde.SpraySimulator(
        gde.Fardal2015DF(), gde.WholeStream(), solver=LEAP, progenitor_solver=LEAP
    )
    assert jnp.all(jnp.isfinite(sim(KEY, HOST, W_TODAY, MSAT, sched).lead))


class _Counted(eqx.Module):
    """A field whose terms count their evaluations on the host."""

    field: _WholeStreamField

    def terms(self, solver):
        term = self.field.terms(solver)

        def vf(t, w, args):
            jax.debug.callback(lambda: COUNT.append(1))
            return term.vf(t, w, args)

        return dfx.ODETerm(vf)


COUNT: list[int] = []


def test_failed_segment_is_nan_and_cheap() -> None:
    """A failed segment makes every row NaN, without later segments stalling.

    The first segment (1000 Myr) needs more than ``max_steps``; the five after
    it (1 Myr each) need a few steps. Continuing from NaN, each of them would
    run to ``max_steps``.
    """
    solver = adaptive(1e-10, 30)
    ws = progenitor_states(HOST, W_TODAY, gde.Uniform(jnp.asarray(1.0), 2), LEAP)
    w0 = jnp.stack([ws[0], ws[0] + jnp.asarray([0.1, 0, 0, 0, 5e-3, 0])])
    field = _WholeStreamField(HOST, plummer(MSAT))
    active = jnp.ones((6, 1), bool)

    def grids(span):
        ends = jnp.concat([jnp.asarray([0.0, span]), span + jnp.arange(1.0, 6.0)])
        return jnp.stack([ends[:-1], ends[1:]], axis=1)

    run = jax.jit(lambda g: evolve_whole_stream(_Counted(field), solver, w0, g, active))
    COUNT.clear()
    good = run(grids(1.0))
    jax.effects_barrier()
    n_good = len(COUNT)
    assert jnp.all(jnp.isfinite(good))

    COUNT.clear()
    bad = run(grids(1000.0))
    jax.effects_barrier()
    assert jnp.all(jnp.isnan(bad))
    # Segment 0 at max_steps plus normal later segments, not 6 x max_steps.
    assert len(COUNT) < 30 * 16 + 2 * n_good, (len(COUNT), n_good)

    # Under vmap, a failing lane leaves the good one as it was.
    both = jax.jit(
        jax.vmap(lambda g: evolve_whole_stream(field, solver, w0, g, active))
    )(jnp.stack([grids(1.0), grids(1000.0)]))
    np.testing.assert_allclose(both[0], good, rtol=1e-13, atol=1e-13)
    assert jnp.all(jnp.isnan(both[1]))


def test_nan_start_is_nan_and_cheap() -> None:
    """A NaN start (a failed progenitor solve) costs less than a good run.

    Every segment would otherwise run from NaN to ``max_steps``.
    """
    solver = adaptive(1e-10, 30)
    ws = progenitor_states(HOST, W_TODAY, gde.Uniform(jnp.asarray(1.0), 2), LEAP)
    w0 = jnp.stack([ws[0], ws[0] + jnp.asarray([0.1, 0, 0, 0, 5e-3, 0])])
    field = _Counted(_WholeStreamField(HOST, plummer(MSAT)))
    ends = jnp.arange(0.0, 7.0)
    grids = jnp.stack([ends[:-1], ends[1:]], axis=1)
    active = jnp.ones((6, 1), bool)
    run = jax.jit(lambda w: evolve_whole_stream(field, solver, w, grids, active))

    COUNT.clear()
    assert jnp.all(jnp.isfinite(run(w0)))
    jax.effects_barrier()
    n_good = len(COUNT)

    COUNT.clear()
    assert jnp.all(jnp.isnan(run(w0.at[0, 0].set(jnp.nan))))
    jax.effects_barrier()
    assert len(COUNT) < n_good, (len(COUNT), n_good)
