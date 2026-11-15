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


def _orbit_solver():
    return gd.OrbitSolver(
        dfx.SemiImplicitEuler(), stepsize_controller=dfx.ConstantStepSize()
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
    integrator = gd.integrate.Integrator(
        gd.OrbitSolver(
            dfx.SemiImplicitEuler(),
            stepsize_controller=dfx.ConstantStepSize(),
            max_steps=500,
        ),
        diffeq_kw={"dt0": 0.5, "max_steps": 500},
    )
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(W0[:3], "kpc"), p=u.Q(W0[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
    )
    ts = u.Q(jnp.linspace(-100.0, 0.0, 11), "Myr")
    orbit = gd.evaluate_orbit(POT, w0, ts, integrator=integrator)
    q = jnp.stack([u.ustrip("kpc", getattr(orbit.q, c)) for c in "xyz"], -1)

    # The first saved state is the backward leg's end: a plain backward solve.
    back = _orbit_solver().solve(
        HamiltonianField(POT), (W0[:3], W0[3:]), 0.0, -100.0, dt0=-0.5, max_steps=500
    )
    assert jnp.allclose(q[0], back.ys[0].reshape(3), rtol=1e-12, atol=0)
    assert jnp.all(jnp.isfinite(q))


def test_experimental_integrate_orbit_fixed_step_backward():
    """`experimental.integrate_orbit` (used by `StreamSimulator`) gets the fix."""
    solver = dfxtra.DiffEqSolver(
        dfx.SemiImplicitEuler(),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=200,
    )
    out = [
        experimental.integrate_orbit(
            POT, (W0[:3], W0[3:]), 0.0, -50.0, solver=solver,
            solver_kwargs=ImmutableMap({"dt0": dt0}),
        ).ys
        for dt0 in (0.5, -0.5)
    ]  # fmt: skip
    assert jnp.array_equal(out[0][0], out[1][0])
    assert jnp.array_equal(out[0][1], out[1][1])


def test_stream_simulator_fixed_step():
    """`StreamSimulator` runs with a fixed step, for either sign of dt0."""
    pot = gp.HernquistPotential(1e12, 10, units="galactic")
    qp0 = (jnp.array([15.0, 0.0, 0.0]), jnp.array([0.0, 0.225, 0.0]))
    solver = dfxtra.DiffEqSolver(
        dfx.SemiImplicitEuler(),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=1000,
    )
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
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(W0[:3], "kpc"), p=u.Q(W0[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
    )
    ts = u.Q(jnp.linspace(-100.0, 0.0, 11), "Myr")
    out = []
    for dt0 in (1.0, -1.0):
        integ = gd.integrate.Integrator(
            gd.OrbitSolver(
                dfx.SemiImplicitEuler(),
                stepsize_controller=dfx.ConstantStepSize(),
                max_steps=500,
            ),
            diffeq_kw={"dt0": dt0, "max_steps": 500},
        )
        gen = gd.MockStreamGenerator(
            gd.FardalStreamDF(),
            POT,
            progenitor_integrator=integ,
            stream_integrator=integ,
        )
        stream, _ = gen.run(jr.key(0), ts, w0, u.Q(1e5, "Msun"))
        x = u.ustrip("kpc", stream["lead"].q.x)
        assert jnp.all(jnp.isfinite(x))
        out.append(x)
    assert jnp.array_equal(out[0], out[1])
