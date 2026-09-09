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
