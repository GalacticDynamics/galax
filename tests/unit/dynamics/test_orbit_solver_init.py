"""`OrbitSolver.init` with batched start times (GalacticDynamics/galax#1027)."""

import diffrax as dfx
import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp

FIELD = gd.fields.HamiltonianField(
    gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
)
Q = u.Q([[8.0, 0, 9], [9, 0, 3]], "kpc")
P = u.Q([[0.0, 220, 0], [0, 200, 10]], "km/s")
T0 = u.Q([0.0, 10.0], "Myr")
T1 = u.Q(100.0, "Myr")


def test_solve_from_batched_time_state():
    """Solving from the state equals solving from the initial conditions.

    This uses a solver without a solver state. A solver with one (e.g. the
    default Dopri8) also needs a batched solve not to hand every particle the
    whole batched solver state.
    """
    s = gd.OrbitSolver(
        dfx.SemiImplicitEuler(), stepsize_controller=dfx.ConstantStepSize()
    )
    got = s.solve(FIELD, s.init(FIELD, (Q, P), T0, None), T1, dt0=0.5).ys
    want = s.solve(FIELD, gc.PhaseSpaceCoordinate(q=Q, p=P, t=T0), T1, dt0=0.5).ys
    assert got[0].shape == want[0].shape
    assert jnp.allclose(got[0], want[0], rtol=1e-12, atol=0)
    assert jnp.allclose(got[1], want[1], rtol=1e-12, atol=0)


def test_scalar_start_time_unchanged():
    """A scalar start time with a batch of particles is one batched state."""
    state = gd.OrbitSolver().init(FIELD, (Q, P), T0[0], None)
    assert state.t.shape == ()
    assert state.y[0].shape == (2, 3)


def test_mismatched_batches_raise():
    """Start times and particles that don't broadcast are an error."""
    q3 = jnp.concat([Q, Q[:1]])
    p3 = jnp.concat([P, P[:1]])
    with pytest.raises(ValueError, match="broadcast"):
        gd.OrbitSolver().init(FIELD, (q3, p3), T0, None)


@pytest.mark.parametrize(
    "solver",
    [
        gd.OrbitSolver(),
        gd.OrbitSolver(
            dfx.SemiImplicitEuler(), stepsize_controller=dfx.ConstantStepSize()
        ),
    ],
    ids=["default", "semi-implicit-euler"],
)
@pytest.mark.parametrize(
    ("t0_shape", "q_shape"),
    [((2,), (2,)), ((2,), ()), ((2, 3), (2, 3)), ((2, 1), (3,)), ((2, 3), ())],
    ids=["2-2", "2-one", "2x3-2x3", "2x1-3", "2x3-one"],
)
def test_batched_state_equals_per_element_inits(solver, t0_shape, q_shape):
    """Every leaf, solver state included, equals a scalar `init` of that element."""
    rng = np.random.default_rng(0)
    t0 = jnp.asarray(rng.uniform(0, 50, t0_shape))
    q = jnp.asarray(rng.uniform(1, 10, (*q_shape, 3)))
    p = jnp.asarray(rng.uniform(-0.2, 0.2, (*q_shape, 3)))
    state = solver.init(FIELD, (q, p), u.Q(t0, "Myr"), None)

    batch = np.broadcast_shapes(t0_shape, q_shape)
    t0_b = jnp.broadcast_to(t0, batch)
    q_b, p_b = jnp.broadcast_to(q, (*batch, 3)), jnp.broadcast_to(p, (*batch, 3))
    assert state.t.shape == batch
    for idx in np.ndindex(*batch):
        one = solver.init(FIELD, (q_b[idx], p_b[idx]), u.Q(t0_b[idx], "Myr"), None)
        got, want = jax.tree.leaves(state), jax.tree.leaves(one)
        assert len(got) == len(want)
        for x, y in zip(got, want, strict=True):
            if isinstance(x, jax.Array):
                assert jnp.array_equal(x[idx], y)


def test_init_impl_checks_the_batch():
    """`_init_impl` rejects initial conditions that don't lead with t0's batch."""
    t0 = jnp.zeros((2, 3))
    y0 = (jnp.ones((3, 2, 3)), jnp.ones((3, 2, 3)))
    with pytest.raises(ValueError, match="lead with t0's batch shape"):
        gd.OrbitSolver()._init_impl(FIELD, t0, y0, None, FIELD.units)
