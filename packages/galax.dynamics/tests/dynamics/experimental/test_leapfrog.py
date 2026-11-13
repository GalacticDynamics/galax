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
# One force evaluation per step


class _TwoEvalLeapfrog(experimental.Leapfrog):
    """Reference: the same KDK step, evaluating the force twice per step."""

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        f, g = terms
        q0, p0 = y0
        cf, cg = f.contr(t0, t1), g.contr(t0, t1)
        p_half = p0 + g.vf_prod(t0, q0, args, 0.5 * cg)
        q1 = q0 + f.vf_prod(t0, p_half, args, cf)
        p1 = p_half + g.vf_prod(t1, q1, args, 0.5 * cg)
        y1 = (q1, p1)
        return y1, None, {"y0": y0, "y1": y1}, None, dfx.RESULTS.successful


def test_leapfrog_one_force_evaluation_per_step():
    """The end-of-step force is reused as the next step's start force."""
    import jax

    n_evals = []

    def force(t, q, args):
        jax.debug.callback(lambda: n_evals.append(1))
        return -q

    dfx.diffeqsolve(
        (dfx.ODETerm(lambda t, p, args: p), dfx.ODETerm(force)),
        experimental.Leapfrog(),
        0.0,
        1.0,
        0.1,
        (1.0, -0.5),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=20,
    )
    jax.effects_barrier()
    assert len(n_evals) == 11  # one at the start, then one per step


@pytest.mark.parametrize("t1", [1.0, -1.0])
def test_leapfrog_reused_force_matches_two_evaluations(t1):
    """Reusing the force changes nothing, forward or backward."""
    terms = (
        dfx.ODETerm(lambda t, p, args: p),
        dfx.ODETerm(lambda t, q, args: -q + 0.3 * jnp.sin(2 * t)),
    )
    kw = {"stepsize_controller": dfx.ConstantStepSize(), "max_steps": 200}
    got = dfx.diffeqsolve(
        terms, experimental.Leapfrog(), 0.0, t1, 0.01 * t1, (1.0, -0.5), **kw
    )
    ref = dfx.diffeqsolve(
        terms, _TwoEvalLeapfrog(), 0.0, t1, 0.01 * t1, (1.0, -0.5), **kw
    )
    assert jnp.allclose(got.ys[0], ref.ys[0], rtol=1e-13, atol=0)
    assert jnp.allclose(got.ys[1], ref.ys[1], rtol=1e-13, atol=0)


def test_leapfrog_recomputes_force_after_a_jump():
    """After a jump in the force, the stored force is stale and is recomputed."""
    terms = (
        dfx.ODETerm(lambda t, p, args: p),
        # The jump plus a smooth time dependence: a recompute at the wrong time
        # would show.
        dfx.ODETerm(
            lambda t, q, args: -q + jnp.where(t > 0.55, 1.0, 0.0) + 0.3 * jnp.sin(2 * t)
        ),
    )
    kw = {
        "stepsize_controller": dfx.ClipStepSizeController(
            dfx.ConstantStepSize(), jump_ts=jnp.array([0.55])
        ),
        "max_steps": 50,
    }
    got = dfx.diffeqsolve(
        terms, experimental.Leapfrog(), 0.0, 1.0, 0.1, (1.0, -0.5), **kw
    )
    ref = dfx.diffeqsolve(terms, _TwoEvalLeapfrog(), 0.0, 1.0, 0.1, (1.0, -0.5), **kw)
    assert jnp.allclose(got.ys[0], ref.ys[0], rtol=1e-13, atol=0)
    assert jnp.allclose(got.ys[1], ref.ys[1], rtol=1e-13, atol=0)


def test_leapfrog_one_force_evaluation_per_step_under_vmap():
    """Under vmap with possible jumps but none taken, still one force per step.

    The callback has no batched arguments, so it fires once per batched
    evaluation. Without `unvmap_any`, the recompute would run every step.
    """
    import jax

    n_evals = []

    def force(t, q, args):
        jax.debug.callback(lambda: n_evals.append(1))
        return -q

    def solve(t_jump):
        return dfx.diffeqsolve(
            (dfx.ODETerm(lambda t, p, args: p), dfx.ODETerm(force)),
            experimental.Leapfrog(),
            0.0,
            1.0,
            0.1,
            (1.0, -0.5),
            stepsize_controller=dfx.ClipStepSizeController(
                dfx.ConstantStepSize(), jump_ts=jnp.atleast_1d(t_jump)
            ),
            max_steps=20,
        ).ys

    # A batched jump time makes `made_jump` batched; none is reached.
    jax.vmap(solve)(jnp.array([5.0, 6.0, 7.0, 8.0]))
    jax.effects_barrier()
    assert len(n_evals) == 11


def test_leapfrog_orbitsolver_step_after_run():
    """`OrbitSolver.run` keeps no solver state; the next `step` still works."""
    import unxt as u

    import galax.coordinates as gc
    import galax.dynamics as gd
    import galax.potential as gp

    field = gd.fields.HamiltonianField(
        gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    )
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(W0[:3], "kpc"), p=u.Q(W0[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
    )
    out = []
    for leapfrog in (experimental.Leapfrog(), _TwoEvalLeapfrog()):
        solver = gd.OrbitSolver(
            solver=leapfrog, stepsize_controller=dfx.ConstantStepSize(), max_steps=200
        )
        state = solver.run(
            field, solver.init(field, w0, None), u.Q(50, "Myr"), None, dt0=1.0
        )
        out.append(solver.step(field, state, u.Q(60, "Myr"), None).y)
    assert jnp.allclose(out[0][0], out[1][0], rtol=1e-13, atol=0)
    assert jnp.allclose(out[0][1], out[1][1], rtol=1e-13, atol=0)
