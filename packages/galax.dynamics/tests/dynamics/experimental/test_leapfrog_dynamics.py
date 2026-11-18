"""Leapfrog tests that need `galax.dynamics` (the rest are in galax.orbits)."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

import diffraxtra as dfxtra
import unxt as u
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.dynamics as gd
import galax.fields as gf
import galax.potential as gp

W0 = jnp.array([15.0, 0.0, 10.0, 0.0, 0.17, -0.05])


def test_evaluate_orbit_fixed_step_back_then_forward():
    """One fixed-step integrator does `evaluate_orbit`'s backward-then-forward legs.

    The orbit starts at t = 0 and is saved from t = -100 to 0, so the progenitor
    is first integrated back to -100 and then forward again (as
    `MockStreamGenerator` does with its progenitor). It must come back to the
    start to round-off, and its first saved state must match a hand-written
    backward KDK scan.
    """
    pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    integrator = gd.integrate.Integrator(
        gd.OrbitSolver(
            gf.Leapfrog(),
            stepsize_controller=dfx.ConstantStepSize(),
            max_steps=200,
        ),
        diffeq_kw={"dt0": 1.0, "max_steps": 200},
    )
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(W0[:3], "kpc"), p=u.Q(W0[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
    )
    ts = u.Q(jnp.linspace(-100.0, 0.0, 11), "Myr")
    orbit = gd.evaluate_orbit(pot, w0, ts, integrator=integrator)
    q = jnp.stack([u.ustrip("kpc", getattr(orbit.q, c)) for c in "xyz"], -1)
    p = jnp.stack([u.ustrip("kpc/Myr", getattr(orbit.p, c)) for c in "xyz"], -1)

    def kdk(qp, _):
        q, p = qp
        h = -1.0
        p = p - 0.5 * h * pot._gradient(q, 0.0)
        q = q + h * p
        p = p - 0.5 * h * pot._gradient(q, 0.0)
        return (q, p), None

    (q_back, p_back), _ = jax.lax.scan(kdk, (W0[:3], W0[3:]), length=100)
    assert jnp.allclose(q[0], q_back, rtol=1e-12, atol=0)
    assert jnp.allclose(p[0], p_back, rtol=1e-12, atol=0)
    assert jnp.max(jnp.abs(q[-1] - W0[:3])) <= 1e-10
    assert jnp.max(jnp.abs(p[-1] - W0[3:])) <= 1e-12


def test_stream_simulator_fixed_step():
    """`StreamSimulator` runs with a fixed-step `Leapfrog`, for either sign of dt0.

    Its progenitor is integrated back to the first release time and then
    forward, so one constant-step solver has to go both ways.
    """
    pot = gp.HernquistPotential(1e12, 10, units="galactic")
    qp0 = (jnp.array([15.0, 0.0, 0.0]), jnp.array([0.0, 0.225, 0.0]))
    solver = dfxtra.DiffEqSolver(
        gf.Leapfrog(),
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
