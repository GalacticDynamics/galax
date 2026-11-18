# ruff: noqa: ARG005

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest

import diffraxtra as dfxtra
import unxt as u
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.fields as gf
import galax.orbits as go
import galax.orbits._src.experimental as experimental
import galax.potential as gp
from galax.orbits import HamiltonianField


def shaped_allclose(x, y, **kwargs):
    return jnp.shape(x) == jnp.shape(y) and jnp.allclose(x, y, **kwargs)


@pytest.mark.parametrize("solver", [gf.Leapfrog()])
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

    sol = dfx.diffeqsolve((dq_dt, dp_dt), gf.Leapfrog(), 0, 1, dt0, y0, max_steps=1000)

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
        gf.Leapfrog(),
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
    pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    solver = go.OrbitSolver(gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize())
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
    field = HamiltonianField(gp.NFWPotential(m=1e12, r_s=15, units="galactic"))
    solver = go.OrbitSolver(gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize())
    y0 = (W0[:3], W0[3:])
    want = solver.solve(field, y0, 0.0, -100.0, dt0=-1.0, max_steps=101)
    got = solver.solve(field, y0, 0.0, -100.0, dt0=1.0, max_steps=101)
    assert jnp.array_equal(got.ys[0], want.ys[0])
    assert jnp.array_equal(got.ys[1], want.ys[1])


@pytest.mark.parametrize("method", ["solve", "run"])
def test_fixed_step_dt0_quantity(method):
    """A `unxt.Quantity` ``dt0`` is converted to the field's time unit."""
    field = HamiltonianField(gp.NFWPotential(m=1e12, r_s=15, units="galactic"))
    solver = go.OrbitSolver(gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize())
    y0 = (W0[:3], W0[3:])

    def ys(dt0):
        if method == "solve":
            return solver.solve(field, y0, 0.0, -100.0, dt0=dt0, max_steps=101).ys
        state = solver.init(field, y0, 0.0, None)
        return solver.run(field, state, -100.0, None, dt0=dt0, max_steps=101).y

    want, got = ys(1.0), ys(u.Q(1000, "kyr"))
    # Same steps; diffrax 0.7.1 and later round a Python float `dt0` and a traced
    # one differently, ~1e-15 apart. A wrong unit is off by orders of magnitude.
    assert jnp.allclose(got[0], want[0], rtol=0, atol=1e-12)
    assert jnp.allclose(got[1], want[1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["forward", "backward"])
def test_compute_orbit_fixed_step_dt0(sign):
    """`compute_orbit` passes ``dt0`` on, so a constant-step solver runs through it.

    It matches `OrbitSolver.solve` with the same ``dt0``. A
    `PhaseSpaceCoordinate` starting before ``ts[0]`` (in the direction of
    integration) takes a first leg to ``ts[0]``, here 20 steps of 0.5 Myr.
    """
    field = HamiltonianField(gp.NFWPotential(m=1e12, r_s=15, units="galactic"))
    solver = go.OrbitSolver(
        gf.Leapfrog(),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=400,
    )
    ts = sign * jnp.linspace(10.0, 100.0, 10)
    dt0 = u.Q(0.5, "Myr")

    def via_solve(y0, t0):
        soln = solver.solve(
            field, y0, t0, ts[-1], saveat=dfx.SaveAt(ts=ts), dt0=0.5, unbatch_time=True
        )
        return jnp.concat(soln.ys, axis=-1)

    def via_compute(w0):
        orbit = go.compute_orbit(field, w0, u.Q(ts, "Myr"), solver=solver, dt0=dt0)
        q = [u.ustrip("kpc", getattr(orbit.q, c)) for c in "xyz"]
        p = [u.ustrip("kpc/Myr", getattr(orbit.p, c)) for c in "xyz"]
        return jnp.stack(q + p, axis=-1)

    q0, p0 = u.Q(W0[:3], "kpc"), u.Q(W0[3:], "kpc/Myr")

    # PhaseSpacePosition: w0 is at ts[0]
    got = via_compute(gc.PhaseSpacePosition(q=q0, p=p0))
    want = via_solve((W0[:3], W0[3:]), ts[0])
    assert jnp.allclose(got, want, rtol=1e-12, atol=0)

    # PhaseSpaceCoordinate: w0 is at t = 0, integrated to ts[0] first
    got = via_compute(gc.PhaseSpaceCoordinate(q=q0, p=p0, t=u.Q(0.0, "Myr")))
    leg0 = solver.solve(field, (W0[:3], W0[3:]), 0.0, ts[0], dt0=0.5, unbatch_time=True)
    want = via_solve(leg0.ys, ts[0])
    assert jnp.allclose(got, want, rtol=1e-12, atol=0)


def test_experimental_integrate_orbit_fixed_step_backward():
    """`experimental.integrate_orbit` (used by `StreamSimulator`) gets the same fix.

    A constant step with ``dt0 = +0.5`` integrates backward with steps of 0.5,
    matching a hand-written backward KDK scan.
    """
    pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    solver = dfxtra.DiffEqSolver(
        gf.Leapfrog(),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=100,
    )
    soln = experimental.integrate_orbit(
        pot,
        (W0[:3], W0[3:]),
        0.0,
        -50.0,
        solver=solver,
        solver_kwargs=ImmutableMap({"dt0": 0.5}),
    )

    def kdk(qp, _):
        q, p = qp
        h = -0.5
        p = p - 0.5 * h * pot._gradient(q, 0.0)
        q = q + h * p
        p = p - 0.5 * h * pot._gradient(q, 0.0)
        return (q, p), None

    (q, p), _ = jax.lax.scan(kdk, (W0[:3], W0[3:]), length=100)
    assert jnp.allclose(soln.ys[0][-1], q, rtol=1e-12, atol=0)
    assert jnp.allclose(soln.ys[1][-1], p, rtol=1e-12, atol=0)


# One force evaluation per step


class _TwoEvalLeapfrog(gf.Leapfrog):
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


def _counting_force():
    """Return a force -q that records each evaluation, and the record."""
    n_evals = []

    def force(t, q, args):
        jax.debug.callback(lambda: n_evals.append(1))
        return -q

    return force, n_evals


def test_leapfrog_one_force_evaluation_per_step():
    """The end-of-step force is reused as the next step's start force."""
    force, n_evals = _counting_force()

    dfx.diffeqsolve(
        (dfx.ODETerm(lambda t, p, args: p), dfx.ODETerm(force)),
        gf.Leapfrog(),
        0.0,
        1.0,
        0.1,
        (1.0, -0.5),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=20,
    )
    jax.effects_barrier()
    assert len(n_evals) == 11  # one at the start, then one per step


class _ConstantStep(dfx.AbstractAdaptiveStepSizeController):
    """A constant step size that `ClipStepSizeController` accepts.

    diffrax 0.7.1 and later only wrap adaptive controllers in
    `ClipStepSizeController`, and `Leapfrog` has no error estimate to drive
    one. This steps by `dt0` and ignores tolerances.
    """

    rtol: float = 0.0
    atol: float = 0.0
    norm: object = None

    def wrap(self, direction):
        return self

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        return t0 + dt0, dt0

    def adapt_step_size(
        self, t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state
    ):
        return (
            True,
            t1,
            t1 + controller_state,
            False,
            controller_state,
            dfx.RESULTS.successful,
        )


def _smooth(t, q, args):
    return -q + 0.3 * jnp.sin(2 * t)


def _jumpy(t, q, args):  # a jump at 0.55, plus smooth time dependence
    return -q + jnp.where(t > 0.55, 1.0, 0.0) + 0.3 * jnp.sin(2 * t)


@pytest.mark.parametrize(
    ("force", "t1", "dt0", "controller"),
    [
        (_smooth, 1.0, 0.01, dfx.ConstantStepSize()),
        (_smooth, -1.0, -0.01, dfx.ConstantStepSize()),
        (
            _jumpy,
            1.0,
            0.1,
            dfx.ClipStepSizeController(_ConstantStep(), jump_ts=jnp.array([0.55])),
        ),
    ],
    ids=["forward", "backward", "jump"],
)
def test_leapfrog_reused_force_matches_two_evaluations(force, t1, dt0, controller):
    """Reusing the force changes nothing, forward, backward, or after a jump."""
    terms = (dfx.ODETerm(lambda t, p, args: p), dfx.ODETerm(force))
    kw = {"stepsize_controller": controller, "max_steps": 200}
    got = dfx.diffeqsolve(terms, gf.Leapfrog(), 0.0, t1, dt0, (1.0, -0.5), **kw)
    ref = dfx.diffeqsolve(terms, _TwoEvalLeapfrog(), 0.0, t1, dt0, (1.0, -0.5), **kw)
    assert jnp.allclose(got.ys[0], ref.ys[0], rtol=1e-13, atol=0)
    assert jnp.allclose(got.ys[1], ref.ys[1], rtol=1e-13, atol=0)


def test_leapfrog_one_force_evaluation_per_step_under_vmap():
    """Under vmap with possible jumps but none taken, still one force per step.

    The callback has no batched arguments, so it fires once per batched
    evaluation. Without `unvmap_any`, the recompute would run every step.
    """
    force, n_evals = _counting_force()

    def solve(t_jump):
        return dfx.diffeqsolve(
            (dfx.ODETerm(lambda t, p, args: p), dfx.ODETerm(force)),
            gf.Leapfrog(),
            0.0,
            1.0,
            0.1,
            (1.0, -0.5),
            stepsize_controller=dfx.ClipStepSizeController(
                _ConstantStep(), jump_ts=jnp.atleast_1d(t_jump)
            ),
            max_steps=20,
        ).ys

    # A batched jump time makes `made_jump` batched; none is reached.
    jax.vmap(solve)(jnp.array([5.0, 6.0, 7.0, 8.0]))
    jax.effects_barrier()
    assert len(n_evals) == 11


def test_leapfrog_orbitsolver_step_after_run():
    """`OrbitSolver.run` keeps no solver state; the next `step` still works."""
    field = go.HamiltonianField(gp.NFWPotential(m=1e12, r_s=15, units="galactic"))
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(W0[:3], "kpc"), p=u.Q(W0[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
    )
    out = []
    for leapfrog in (gf.Leapfrog(), _TwoEvalLeapfrog()):
        solver = go.OrbitSolver(
            solver=leapfrog, stepsize_controller=dfx.ConstantStepSize(), max_steps=200
        )
        state = solver.run(
            field, solver.init(field, w0, None), u.Q(50, "Myr"), None, dt0=1.0
        )
        out.append(solver.step(field, state, u.Q(60, "Myr"), None).y)
    assert jnp.allclose(out[0][0], out[1][0], rtol=1e-13, atol=0)
    assert jnp.allclose(out[0][1], out[1][1], rtol=1e-13, atol=0)


@pytest.mark.parametrize(
    "solver",
    [gf.Leapfrog(), None],
    ids=["leapfrog", "default"],
)
def test_solve_from_init_state_with_batched_t1(solver):
    """A state from `OrbitSolver.init` solves to a batched end time.

    The batched solve is split per particle, so the stored solver state, which
    is batched over both particles, can't be reused as is.
    """
    field = go.HamiltonianField(
        gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
    )
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([[8.0, 0, 9], [9, 0, 3]], "kpc"),
        p=u.Q([[0.0, 220, 0], [0, 200, 10]], "km/s"),
        t=u.Q(0.0, "Myr"),
    )
    t1 = u.Q([100.0, 50.0], "Myr")
    if solver is None:
        s, kw = go.OrbitSolver(), {}
    else:
        s = go.OrbitSolver(solver, stepsize_controller=dfx.ConstantStepSize())
        kw = {"dt0": 0.5}
    got = s.solve(field, s.init(field, w0, None), t1, **kw).ys
    want = s.solve(field, w0, t1, **kw).ys
    assert jnp.allclose(got[0], want[0], rtol=1e-13, atol=0)
    assert jnp.allclose(got[1], want[1], rtol=1e-13, atol=0)


def test_solve_from_state_reuses_the_stored_force(monkeypatch, request):
    """An unbatched solve from a saved state reuses its force: 10 steps, 10 forces."""
    n_evals = []
    dv_dt = HamiltonianField.dv_dt

    def counted(self, t, q, args):
        jax.debug.callback(lambda: n_evals.append(1))
        return dv_dt(self, t, q, args)

    monkeypatch.setattr(HamiltonianField, "dv_dt", counted)
    # The patched method isn't part of the jit cache key: drop compiled solves
    # so this one retraces with the counter, and drop the counted trace after
    # the test so no later solve reuses it.
    jax.clear_caches()
    request.addfinalizer(jax.clear_caches)

    field = go.HamiltonianField(
        gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
    )
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([8.0, 0, 9], "kpc"), p=u.Q([0.0, 220, 0], "km/s"), t=u.Q(0.0, "Myr")
    )
    solver = go.OrbitSolver(gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize())
    state = solver.init(field, w0, None)
    jax.effects_barrier()
    n_evals.clear()
    solver.solve(field, state, u.Q(100.0, "Myr"), dt0=10.0).ys[0].block_until_ready()
    jax.effects_barrier()
    assert len(n_evals) == 10
