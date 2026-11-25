"""`Integrator(dense=True)` with batched start or end times (galax#1098)."""

import equinox as eqx
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.orbits as go
import galax.potential as gp

FIELD = go.HamiltonianField(gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic"))
W0 = gc.PhaseSpacePosition(
    q=u.Q([[10, 0, 0], [11, 0, 0]], "kpc"), p=u.Q([[0, 200, 0], [0, 210, 0]], "km/s")
)
INTEGRATOR = go.integrate.Integrator()

# (t0, t1) pairs: batched start, batched end.
CASES = [
    (u.Q([0, 100], "Myr"), u.Q(1000, "Myr")),
    (u.Q(0, "Myr"), u.Q([900, 1000], "Myr")),
]


@pytest.mark.parametrize(("t0", "t1"), CASES)
def test_dense_matches_sparse(t0, t1) -> None:
    """The dense result has the sparse result's shape and end points."""
    sparse = INTEGRATOR(FIELD, W0, t0, t1)
    dense = INTEGRATOR(FIELD, W0, t0, t1, dense=True)

    assert dense.shape == sparse.shape == (2,)
    assert dense.t.shape == (2,)
    assert dense.interpolant.batch_shape == (2,)
    assert jnp.allclose(dense.q.x, sparse.q.x, atol=u.Q(1e-12, "kpc"))


@pytest.mark.parametrize(("t0", "t1"), CASES)
@pytest.mark.parametrize("ts", [u.Q(500.0, "Myr"), u.Q([200.0, 500, 900], "Myr")])
def test_dense_evaluates_each_member(t0, t1, ts) -> None:
    """Evaluating the batch matches integrating each member on its own."""
    dense = INTEGRATOR(FIELD, W0, t0, t1, dense=True)
    out = dense(ts)
    assert out.q.shape == (2, *ts.shape)

    t0s, t1s = jnp.broadcast_to(t0, (2,)), jnp.broadcast_to(t1, (2,))
    for i in range(2):
        wi = gc.PhaseSpacePosition(q=W0.q[i], p=W0.p[i])
        ref = INTEGRATOR(FIELD, wi, t0s[i], t1s[i], saveat=jnp.atleast_1d(ts))
        got = jnp.atleast_1d(out.q.x[i])
        assert jnp.allclose(got, ref.q.x, rtol=1e-7, atol=u.Q(1e-10, "kpc"))


def test_dense_rejects_time_outside_one_member() -> None:
    """A time inside one member's window but not the other's is out of bounds."""
    t0, t1 = CASES[0]
    dense = INTEGRATOR(FIELD, W0, t0, t1, dense=True)
    with pytest.raises(eqx.EquinoxRuntimeError, match="Time out of bounds"):
        dense(u.Q(50.0, "Myr"))


def test_dense_scalar_times_matches_sparse_shape() -> None:
    """With scalar times, the dense result has the sparse result's shape."""
    w0 = gc.PhaseSpacePosition(q=W0.q[0], p=W0.p[0])
    t0, t1 = u.Q(0, "Myr"), u.Q(1000, "Myr")
    dense = INTEGRATOR(FIELD, w0, t0, t1, dense=True)
    sparse = INTEGRATOR(FIELD, w0, t0, t1)
    assert dense.shape == sparse.shape == ()
    assert dense.t.shape == sparse.t.shape == ()


@pytest.mark.parametrize(
    "ts", [u.Q([200.0, 500], "Myr"), u.Q([200.0, 500, 900], "Myr")]
)
def test_orbit_from_batched_solution_evaluates_each_member(ts) -> None:
    """An `Orbit` built from a batched-`t0` dense solve evaluates without transposing.

    The time count equal to the batch size used to come back transposed.
    """
    soln = go.OrbitSolver().solve(
        FIELD,
        (
            u.Q([[10.0, 0, 0], [11, 0, 0]], "kpc"),
            u.Q([[0.0, 200, 0], [0, 210, 0]], "km/s"),
        ),
        u.Q([0.0, 100], "Myr"),
        u.Q(1000.0, "Myr"),
        dense=True,
        vectorize_interpolation=True,
    )
    orbit = go.Orbit.from_(soln, frame=gc.frames.simulation_frame, units=FIELD.units)
    got = u.ustrip("kpc", orbit.interpolant(ts).q.x)
    ref = soln.evaluate(u.ustrip("Myr", ts))[0][..., 0]
    assert got.shape == (2, *ts.shape)
    assert jnp.allclose(got, ref, rtol=1e-12)
