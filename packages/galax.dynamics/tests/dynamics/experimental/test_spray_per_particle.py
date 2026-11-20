"""Tests for `SpraySimulator` in per-particle mode with progenitor copies."""

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
    release_ics,
)
from galax.dynamics._src.experimental.df import sample_releases
from galax.dynamics._src.experimental.fields import _CopiesField
from galax.dynamics._src.experimental.simulator import evolve_per_particle
from galax.dynamics._src.experimental.solve import progenitor_states

gde = gd.experimental
LOOPS = [lstrat.VMap, lstrat.Scan]


@pytest.mark.parametrize("prog", [None, plummer])
@pytest.mark.parametrize("loop", LOOPS)
def test_copies_match_whole_stream(loop, prog) -> None:
    sched = gde.Uniform(jnp.asarray(300.0), 5, 8)
    df = gde.Chen2024DF()
    whole = gde.SpraySimulator(df, gde.WholeStream(), progenitor_potential=prog)
    per = gde.SpraySimulator(
        df, gde.PerParticle(gde.Copies(), loop), progenitor_potential=prog
    )
    want = whole(KEY, HOST, W_TODAY, MSAT, sched)
    got = per(KEY, HOST, W_TODAY, MSAT, sched)
    for g, w in ((got.lead, want.lead), (got.trail, want.trail)):
        np.testing.assert_allclose(g[:, :3], w[:, :3], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(got.release_times, want.release_times)
    np.testing.assert_allclose(got.progenitor, W_TODAY, rtol=0, atol=1e-12)


@pytest.mark.parametrize("loop", LOOPS)
def test_copies_end_at_progenitor(loop) -> None:
    """Each copy takes the progenitor's lattice steps, so it ends at w_today."""
    sched = gde.Uniform(jnp.asarray(300.0), 5, 8)
    ws, lead0, trail0 = release_ics(gde.Chen2024DF(), HOST, plummer, sched, LEAP)
    idx, lattice = sched.release_index(), sched.lattice()
    w0 = jnp.stack([lead0, trail0], axis=1)
    w0 = jnp.stack([w0, jnp.broadcast_to(ws[idx, None], w0.shape)], axis=2)
    terms = _CopiesField(HOST, plummer(MSAT)).terms(LEAP)
    w = evolve_per_particle(terms, LEAP, w0, lattice[idx], lattice, loop)
    assert w.shape == (5, 2, 2, 6)
    np.testing.assert_allclose(
        w[:, :, 1], jnp.broadcast_to(W_TODAY, (5, 2, 6)), rtol=0, atol=1e-12
    )
    # The release at t1 is not integrated: it is its release state.
    np.testing.assert_array_equal(w[-1], w0[-1])


def test_copies_adaptive() -> None:
    sched = gde.Uniform(jnp.asarray(500.0), 8)
    df = gde.Fardal2015DF()
    solver = adaptive(1e-10, 10_000)
    whole = gde.SpraySimulator(
        df, gde.WholeStream(), progenitor_potential=plummer, solver=solver
    )
    per = gde.SpraySimulator(
        df, gde.PerParticle(gde.Copies()), progenitor_potential=plummer, solver=solver
    )
    a = whole(KEY, HOST, W_TODAY, MSAT, sched)
    b = per(KEY, HOST, W_TODAY, MSAT, sched)
    d = jnp.linalg.norm(jnp.concat([a.lead - b.lead, a.trail - b.trail])[:, :3], axis=1)
    assert jnp.median(d) < 1e-6, jnp.median(d)
    np.testing.assert_allclose(b.progenitor, W_TODAY, rtol=0, atol=1e-8)


def test_copies_own_progenitor_solver() -> None:
    """A progenitor solver unlike ``solver`` is allowed (the copies drift)."""
    sched = gde.Uniform(jnp.asarray(300.0), 4, 5)
    sim = gde.SpraySimulator(
        gde.Fardal2015DF(),
        gde.PerParticle(gde.Copies()),
        progenitor_potential=plummer,
        progenitor_solver=adaptive(1e-8, 10_000),
    )
    res = sim(KEY, HOST, W_TODAY, MSAT, sched)
    assert jnp.all(jnp.isfinite(jnp.concat([res.lead, res.trail])))


def test_copies_deep_release() -> None:
    """A release inside the core of a 1e8 Msun Plummer stays finite.

    Positions and gradients are finite, through the no-op steps before release
    and the tightly bound orbit after it.
    """
    m, r_s = jnp.asarray(1e8), 5.0

    def deep(mass: jax.Array) -> gp.AbstractPotential:
        return gp.PlummerPotential(m_tot=mass, r_s=r_s, units=USYS)

    df = gde.Fardal2015DF()
    sched = gde.Uniform(jnp.asarray(300.0), 6, 10)
    ws = progenitor_states(HOST, W_TODAY, sched, LEAP)
    wr = ws[sched.release_index()]
    xl, _, xt, _ = sample_releases(
        df, KEY, HOST, sched.release_times(), wr[:, :3], wr[:, 3:], jnp.full(6, m)
    )
    sep = jnp.linalg.norm(jnp.concat([xl, xt]) - jnp.tile(wr[:, :3], (2, 1)), axis=1)
    assert jnp.all(sep < r_s), sep

    sim = gde.SpraySimulator(
        df, gde.PerParticle(gde.Copies()), progenitor_potential=deep
    )

    def loss(w, mass, t_age):
        r = sim(KEY, HOST, w, mass, gde.Uniform(t_age, 6, 10))
        return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

    res = sim(KEY, HOST, W_TODAY, m, sched)
    assert jnp.all(jnp.isfinite(jnp.concat([res.lead, res.trail])))
    g = jax.grad(loss, argnums=(0, 1, 2))(W_TODAY, m, jnp.asarray(300.0))
    assert all(jnp.all(jnp.isfinite(x)) for x in g), g


@pytest.mark.parametrize("loop", LOOPS)
@pytest.mark.parametrize("df_cls", DFS)
def test_grad_copies(df_cls, loop) -> None:
    """Reverse mode vs central differences and vs ``jvp`` with ForwardMode."""
    fwd = dfxtra.DiffEqSolver(
        gf.Leapfrog(),
        stepsize_controller=dfx.ConstantStepSize(),
        adjoint=dfx.ForwardMode(),
    )

    def make(solver):
        sim = gde.SpraySimulator(
            df_cls(),
            gde.PerParticle(gde.Copies(), loop),
            progenitor_potential=plummer,
            solver=solver,
        )

        def f(s, w, t_age):
            pot = gp.NFWPotential(m=1e12 * s, r_s=15.0, units=USYS)
            r = sim(KEY, pot, w, MSAT, gde.Uniform(t_age, 4, 20))
            return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

        return f

    f, f_fwd = make(LEAP), make(fwd)
    x = (jnp.asarray(1.0), W_TODAY, jnp.asarray(300.0))
    g = jnp.concat([jnp.ravel(a) for a in jax.grad(f, argnums=(0, 1, 2))(*x)])
    jf = jnp.concat([jnp.ravel(a) for a in jax.jacfwd(f_fwd, argnums=(0, 1, 2))(*x)])
    np.testing.assert_allclose(g, jf, rtol=1e-10, atol=1e-10 * jnp.abs(g).max())

    flat = jnp.concat([jnp.ravel(a) for a in x])
    eps = jnp.concat([jnp.asarray([1e-6]), jnp.full(6, 1e-6), jnp.asarray([1e-4])])

    def f_flat(z):
        return f(z[0], z[1:7], z[7])

    fd = jnp.asarray(
        [
            (f_flat(flat.at[i].add(eps[i])) - f_flat(flat.at[i].add(-eps[i])))
            / (2 * eps[i])
            for i in range(flat.size)
        ]
    )
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7 * jnp.abs(g).max())


@pytest.mark.parametrize("loop", LOOPS)
def test_failed_particle_is_nan(loop) -> None:
    """A failed solve blanks its own particle, not the others."""
    solver = adaptive(1e-10, 30)
    lattice = jnp.asarray([-1000.0, -1.0, 0.0])
    w = W_TODAY + jnp.asarray([0.1, 0, 0, 0, 5e-3, 0])
    w0 = jnp.broadcast_to(jnp.stack([w, W_TODAY]), (2, 2, 2, 6))
    terms = _CopiesField(HOST, plummer(MSAT)).terms(solver)
    out = eqx.filter_jit(evolve_per_particle)(
        terms, solver, w0, lattice[:2], lattice, loop
    )
    assert jnp.all(jnp.isnan(out[0]))
    assert jnp.all(jnp.isfinite(out[1]))
