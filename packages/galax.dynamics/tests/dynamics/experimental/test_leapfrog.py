# ruff: noqa: ARG005

import diffrax as dfx
import jax.numpy as jnp
import pytest

from galax.dynamics import experimental


def shaped_allclose(x, y, **kwargs):
    return jnp.shape(x) == jnp.shape(y) and jnp.allclose(x, y, **kwargs)


@pytest.mark.parametrize("solver", [experimental.Leapfrog()])
def test_symplectic_solvers(solver):
    dq_dt = dfx.ODETerm(lambda t, p, args: p)
    dp_dt = dfx.ODETerm(lambda t, q, args: -q)
    y0 = (1.0, -0.5)
    dt0 = 0.001
    sol1 = dfx.diffeqsolve(
        (dq_dt, dp_dt),
        solver,
        0,
        1,
        dt0,
        y0,
        max_steps=2000,
    )
    term_combined = dfx.ODETerm(lambda t, y, args: (y[1], -y[0]))
    sol2 = dfx.diffeqsolve(term_combined, dfx.Tsit5(), 0, 1, 0.001, y0)
    assert shaped_allclose(sol1.ys[0], sol2.ys[0])
    assert shaped_allclose(sol1.ys[1], sol2.ys[1])


def test_leapfrog_time_dependent_force():
    """The second kick must evaluate the force at ``t1`` (post-drift), not ``t0``.

    A time-independent force can't distinguish these -- galax's potentials are
    usually static, so this guards against a future refactor silently swapping
    the two.  With a time-dependent force, evaluating the second kick at the
    wrong time turns the O(h^2) global error into O(h).
    """
    dq_dt = dfx.ODETerm(lambda t, p, args: p)
    dp_dt = dfx.ODETerm(lambda t, q, args: -q + 0.3 * jnp.sin(2 * t))
    y0 = (1.0, -0.5)
    dt0 = 0.01

    sol = dfx.diffeqsolve(
        (dq_dt, dp_dt), experimental.Leapfrog(), 0, 1, dt0, y0, max_steps=1000
    )

    term_combined = dfx.ODETerm(lambda t, y, args: (y[1], -y[0] + 0.3 * jnp.sin(2 * t)))
    ref = dfx.diffeqsolve(
        term_combined,
        dfx.Tsit5(),
        0,
        1,
        0.0001,
        y0,
        stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-10),
        max_steps=200_000,
    )

    # Correct: ~1.5e-5. A t0/t1 swap in the 2nd kick: ~1e-3. 1e-4 cleanly
    # separates the two with margin on both sides.
    assert jnp.abs(sol.ys[0][-1] - ref.ys[0][-1]) < 1e-4
    assert jnp.abs(sol.ys[1][-1] - ref.ys[1][-1]) < 1e-4


def test_leapfrog_conserves_energy_over_many_periods():
    """A symplectic integrator's energy error oscillates; it should not drift.

    Uses a step size coarse enough (50 steps/period) that the per-step error
    is easily visible, over enough periods that a bug breaking the symplectic
    structure (e.g. the drift step using the wrong velocity) would show up as
    unbounded growth rather than being masked by an overly accurate step.
    """
    dq_dt = dfx.ODETerm(lambda t, p, args: p)
    dp_dt = dfx.ODETerm(lambda t, q, args: -q)
    y0 = (1.0, 0.0)

    period = 2 * jnp.pi
    n_periods = 100
    dt0 = period / 50
    t1 = n_periods * period
    saveat = dfx.SaveAt(ts=jnp.linspace(0, t1, n_periods * 10))

    sol = dfx.diffeqsolve(
        (dq_dt, dp_dt),
        experimental.Leapfrog(),
        0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        max_steps=1_000_000,
    )
    q, p = sol.ys
    energy = 0.5 * p**2 + 0.5 * q**2
    rel_err = jnp.abs(energy - energy[0]) / energy[0]

    half = len(rel_err) // 2
    first_half_max = rel_err[:half].max()
    second_half_max = rel_err[half:].max()

    # Bounded, not growing: 100 periods in, the error is still the same order
    # as it was at the start (a non-symplectic or otherwise broken step blows
    # up by many orders of magnitude over this many periods -- see e.g. a
    # drift step using v0 instead of v_half).
    assert second_half_max < 5 * first_half_max
    assert second_half_max < 0.05


# ---------------------------------------------------------------------------
# Direction (GalacticDynamics/galax#963)


def _nfw_leapfrog(t0, t1, w0):
    """`OrbitSolver(Leapfrog)` on an NFW, fixed |dt| = 1 Myr, final state (6,)."""
    import galax.dynamics as gd
    import galax.potential as gp
    from galax.dynamics.fields import HamiltonianField

    pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    solver = gd.OrbitSolver(
        experimental.Leapfrog(), stepsize_controller=dfx.ConstantStepSize()
    )
    n = int(abs(t1 - t0))
    soln = solver.solve(
        HamiltonianField(pot),
        (w0[:3], w0[3:]),
        t0,
        t1,
        dt0=jnp.sign(t1 - t0) * 1.0,
        max_steps=n + 1,
    )
    return pot, jnp.concat([soln.ys[0].reshape(-1), soln.ys[1].reshape(-1)])


W0 = jnp.array([15.0, 0.0, 10.0, 0.0, 0.17, -0.05])


def test_leapfrog_backward():
    """t1 < t0 integrates backward: matches a hand-written KDK scan with h = -dt."""
    import jax

    pot, w = _nfw_leapfrog(0.0, -1000.0, W0)

    def kdk(qp, _):
        q, p = qp
        h = -1.0
        p = p - 0.5 * h * pot._gradient(q, 0.0)
        q = q + h * p
        p = p - 0.5 * h * pot._gradient(q, 0.0)
        return (q, p), None

    (q, p), _ = jax.lax.scan(kdk, (W0[:3], W0[3:]), length=1000)
    assert jnp.allclose(w, jnp.concat([q, p]), rtol=1e-12, atol=0)


def test_leapfrog_round_trip():
    """Forward 1000 Myr then back returns the initial state to round-off."""
    _, w1 = _nfw_leapfrog(0.0, 1000.0, W0)
    _, w2 = _nfw_leapfrog(1000.0, 0.0, w1)
    assert jnp.max(jnp.abs(w2[:3] - W0[:3])) <= 1e-10
    assert jnp.max(jnp.abs(w2[3:] - W0[3:])) <= 1e-12


# ---------------------------------------------------------------------------
# A fixed step whose sign doesn't match the direction


def test_fixed_step_dt0_sign_follows_direction():
    """With a constant step, `dt0` sets the size; t0 -> t1 sets the direction."""
    import galax.dynamics as gd
    import galax.potential as gp
    from galax.dynamics.fields import HamiltonianField

    field = HamiltonianField(gp.NFWPotential(m=1e12, r_s=15, units="galactic"))
    solver = gd.OrbitSolver(
        experimental.Leapfrog(), stepsize_controller=dfx.ConstantStepSize()
    )
    y0 = (W0[:3], W0[3:])
    want = solver.solve(field, y0, 0.0, -100.0, dt0=-1.0, max_steps=101)
    got = solver.solve(field, y0, 0.0, -100.0, dt0=1.0, max_steps=101)
    assert jnp.array_equal(got.ys[0], want.ys[0])
    assert jnp.array_equal(got.ys[1], want.ys[1])


def test_evaluate_orbit_fixed_step_back_then_forward():
    """One fixed-step integrator does `evaluate_orbit`'s backward-then-forward legs.

    The orbit starts at t = 0 and is saved from t = -100 to 0, so the progenitor
    is first integrated back to -100 and then forward again (as
    `MockStreamGenerator` does with its progenitor). It must come back to the
    start to round-off, and its first saved state must match a hand-written
    backward KDK scan.
    """
    import jax

    import unxt as u

    import galax.coordinates as gc
    import galax.dynamics as gd
    import galax.potential as gp

    pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    integrator = gd.integrate.Integrator(
        gd.OrbitSolver(
            experimental.Leapfrog(),
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


def test_experimental_integrate_orbit_fixed_step_backward():
    """`experimental.integrate_orbit` (used by `StreamSimulator`) gets the same fix.

    A constant step with ``dt0 = +1`` integrates backward, matching a
    hand-written backward KDK scan.
    """
    import jax

    import diffraxtra as dfxtra
    from xmmutablemap import ImmutableMap

    import galax.potential as gp

    pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    solver = dfxtra.DiffEqSolver(
        experimental.Leapfrog(),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=100,
    )
    soln = experimental.integrate_orbit(
        pot,
        (W0[:3], W0[3:]),
        0.0,
        -50.0,
        solver=solver,
        solver_kwargs=ImmutableMap({"dt0": 1.0}),
    )

    def kdk(qp, _):
        q, p = qp
        h = -1.0
        p = p - 0.5 * h * pot._gradient(q, 0.0)
        q = q + h * p
        p = p - 0.5 * h * pot._gradient(q, 0.0)
        return (q, p), None

    (q, p), _ = jax.lax.scan(kdk, (W0[:3], W0[3:]), length=50)
    assert jnp.allclose(soln.ys[0][-1], q, rtol=1e-12, atol=0)
    assert jnp.allclose(soln.ys[1][-1], p, rtol=1e-12, atol=0)
