"""A fixed-step solver integrates both backward and forward (backport of #1024).

`diffrax.diffeqsolve` needs the sign of ``dt0`` to match ``t1 - t0``. galax
passes one solver, with one ``dt0``, to both legs of `evaluate_orbit` (back to
``t[0]``, then forward), so a constant-step solver used to fail on one leg.
"""

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr

import diffraxtra as dfxtra
import unxt as u
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from galax.dynamics import experimental
from galax.dynamics.fields import HamiltonianField

POT = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
W0 = jnp.array([15.0, 0.0, 10.0, 0.0, 0.17, -0.05])


W0_PSC = gc.PhaseSpaceCoordinate(
    q=u.Q(W0[:3], "kpc"), p=u.Q(W0[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
)


def _orbit_solver(max_steps=4096):
    return gd.OrbitSolver(
        dfx.SemiImplicitEuler(),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=max_steps,
    )


def _integrator(dt0):
    return gd.integrate.Integrator(
        _orbit_solver(500), diffeq_kw={"dt0": dt0, "max_steps": 500}
    )


def _diffeq_solver(max_steps):
    return dfxtra.DiffEqSolver(
        dfx.SemiImplicitEuler(),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=max_steps,
    )


def test_fixed_step_dt0_sign_follows_direction():
    """With a constant step, `dt0` sets the size; t0 -> t1 sets the direction."""
    field = HamiltonianField(POT)
    y0 = (W0[:3], W0[3:])
    s = _orbit_solver()
    want = s.solve(field, y0, 0.0, -100.0, dt0=-0.5, max_steps=201)
    got = s.solve(field, y0, 0.0, -100.0, dt0=0.5, max_steps=201)
    assert jnp.array_equal(got.ys[0], want.ys[0])
    assert jnp.array_equal(got.ys[1], want.ys[1])


def test_evaluate_orbit_fixed_step_back_then_forward():
    """One fixed-step integrator does `evaluate_orbit`'s backward-then-forward legs.

    The orbit starts at t = 0 and is saved from t = -100 to 0, so it is first
    integrated back to -100 (in steps of 0.5) and then forward again.
    """
    ts = u.Q(jnp.linspace(-100.0, 0.0, 11), "Myr")
    orbit = gd.evaluate_orbit(POT, W0_PSC, ts, integrator=_integrator(0.5))
    q = jnp.stack([u.ustrip("kpc", getattr(orbit.q, c)) for c in "xyz"], -1)

    # Both legs against direct diffrax solves, which take dt0 as given: back
    # to -100 in steps of -0.5, then forward to 0 in steps of +0.5.
    terms = HamiltonianField(POT).terms(dfx.SemiImplicitEuler())
    kw = {"stepsize_controller": dfx.ConstantStepSize(), "max_steps": 500}
    back = dfx.diffeqsolve(
        terms, dfx.SemiImplicitEuler(), 0.0, -100.0, -0.5, (W0[:3], W0[3:]), **kw
    )
    y_back = (back.ys[0][-1], back.ys[1][-1])
    fwd = dfx.diffeqsolve(
        terms, dfx.SemiImplicitEuler(), -100.0, 0.0, 0.5, y_back, **kw
    )
    assert jnp.allclose(q[0], y_back[0], rtol=1e-12, atol=0)
    assert jnp.allclose(q[-1], fwd.ys[0][-1], rtol=1e-12, atol=0)


def test_experimental_integrate_orbit_fixed_step_backward():
    """`experimental.integrate_orbit` (used by `StreamSimulator`) gets the fix.

    With ``dt0 = +0.5`` it steps backward by 0.5: it equals a direct
    `diffrax.diffeqsolve` given ``dt0 = -0.5``, which pins the size of ``dt0``.
    """
    solver = _diffeq_solver(200)
    y0 = (W0[:3], W0[3:])
    got = experimental.integrate_orbit(
        POT, y0, 0.0, -50.0, solver=solver, solver_kwargs=ImmutableMap({"dt0": 0.5})
    ).ys
    want = dfx.diffeqsolve(
        HamiltonianField(POT).terms(solver.solver), dfx.SemiImplicitEuler(),
        0.0, -50.0, -0.5, y0, stepsize_controller=dfx.ConstantStepSize(),
        max_steps=200,
    ).ys  # fmt: skip
    assert jnp.array_equal(got[0], want[0])
    assert jnp.array_equal(got[1], want[1])


def test_stream_simulator_fixed_step():
    """`StreamSimulator` runs with a fixed step, for either sign of dt0."""
    pot = gp.HernquistPotential(1e12, 10, units="galactic")
    qp0 = (jnp.array([15.0, 0.0, 0.0]), jnp.array([0.0, 0.225, 0.0]))
    solver = _diffeq_solver(1000)
    sim = gd.experimental.stream.StreamSimulator()
    leads = []
    for dt0 in (1.0, -1.0):
        kw = {"solver": solver, "solver_kwargs": ImmutableMap({"dt0": dt0})}
        ics = sim.init(
            pot, qp0, 0.0, release_times=jnp.linspace(-400.0, -150.0, 20),
            Msat=1e5, key=jr.key(0), **kw,
        )  # fmt: skip
        lead, _ = sim.run(pot, ics, t1=0.0, **kw)
        assert jnp.all(jnp.isfinite(lead[0]))
        leads.append(lead[0])
    assert jnp.array_equal(leads[0], leads[1])


def test_mockstreamgenerator_fixed_step():
    """`MockStreamGenerator` runs with fixed-step integrators, either sign of dt0."""
    ts = u.Q(jnp.linspace(-100.0, 0.0, 11), "Myr")
    out = []
    for dt0 in (1.0, -1.0):
        integ = _integrator(dt0)
        gen = gd.MockStreamGenerator(
            gd.FardalStreamDF(),
            POT,
            progenitor_integrator=integ,
            stream_integrator=integ,
        )
        stream, _ = gen.run(jr.key(0), ts, W0_PSC, u.Q(1e5, "Msun"))
        x = u.ustrip("kpc", stream["lead"].q.x)
        assert jnp.all(jnp.isfinite(x))
        out.append(x)
    assert jnp.array_equal(out[0], out[1])
