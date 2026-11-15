"""`UniformAcceleration` with batched positions and times."""

import interpax as ipx
import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.dynamics as gd
import galax.potential as gp
from galax.dynamics._src.examples.uniform_acceleration import UniformAcceleration

TS = jnp.linspace(-10.0, 10.0, 50)
VS = jnp.stack([jnp.sin(TS / 5), TS**2 / 100, TS / 7], axis=-1)
POT = UniformAcceleration(velocity_func=ipx.CubicSpline(TS, VS), units="galactic")


@pytest.mark.parametrize(
    ("xyz_shape", "t_shape"),
    [((3,), ()), ((1, 3), ()), ((1, 3), (1,)), ((4, 3), ()), ((4, 3), (4,))],
)
def test_gradient_and_hessian_batch(xyz_shape, t_shape):
    """Batched calls have the position's shape and match unbatched calls."""
    xyz = jnp.arange(1.0, 1 + jnp.prod(jnp.array(xyz_shape))).reshape(xyz_shape)
    t = jnp.linspace(-1.0, 2.0, t_shape[0]) if t_shape else jnp.asarray(0.5)

    grad = POT._gradient(xyz, t)
    hess = POT._hessian(xyz, t)
    assert grad.shape == xyz_shape
    assert hess.shape == (*xyz_shape, 3)
    assert jnp.all(hess == 0)

    xyz_b = xyz.reshape(-1, 3)
    t_b = jnp.broadcast_to(t, xyz_b.shape[:-1])
    for g, x, ti in zip(grad.reshape(-1, 3), xyz_b, t_b, strict=True):
        assert jnp.array_equal(g, POT._gradient(x, ti))


@pytest.mark.parametrize("n", [1, 3])
def test_orbit_in_mw_lmc_potential(n):
    """Orbits integrate in the `make_mw_lmc_potential` composite."""
    pot = gd.examples.make_mw_lmc_potential(
        gp.HernquistPotential(m_tot=1e12, r_s=10, units="galactic"),
        gp.HernquistPotential(m_tot=1e11, r_s=5, units="galactic"),
        (u.Q([0.0, 0, 0], "kpc"), u.Q([0.0, 0, 0], "km/s")),
        (u.Q([-0.8, -41.5, -26.8], "kpc"), u.Q([-56.0, -219, 186], "km/s")),
        t_interp=u.Q(jnp.linspace(0, -100, 21), "Myr"),
    )
    q = u.Q(jnp.array([[10.0, 0, 0], [0, 15, 0], [0, 0, 20]])[:n], "kpc")
    p = u.Q(jnp.array([[0.0, 200, 0], [-180, 0, 0], [0, 150, 50]])[:n], "km/s")
    field = gd.fields.HamiltonianField(pot)
    soln = gd.OrbitSolver().solve(
        field, (q, p), u.Q(0.0, "Myr"), u.Q(-50.0, "Myr"), dt0=-1.0
    )
    assert soln.ys[0].shape == (1, n, 3)
    assert jnp.all(jnp.isfinite(soln.ys[0]))


def _vel(t):
    """Velocity [kpc/Myr] at time ``t`` [Myr]."""
    return jnp.stack([jnp.sin(t / 5), t**2 / 100, t / 7], axis=-1)


def _vel_q(t):
    """Unit-aware `_vel`, taking ``t`` in any time unit."""
    return u.Q(_vel(u.ustrip("Myr", t)), "kpc/Myr")


# A plain function isn't a pytree, so it is wrapped to pass through `jax.jit`.
POT_FN = UniformAcceleration(
    velocity_func=jax.tree_util.Partial(_vel), units="galactic"
)
POT_Q = UniformAcceleration(
    velocity_func=jax.tree_util.Partial(_vel_q),
    units="galactic",
    func_supports_units=True,
)


@pytest.mark.parametrize(
    ("xyz_shape", "t_shape"), [((3,), ()), ((4, 3), ()), ((4, 3), (4,))]
)
def test_unit_aware_velocity_func(xyz_shape, t_shape):
    """A unit-aware velocity function gives the unitless gradient."""
    xyz = jnp.ones(xyz_shape)
    t = jnp.linspace(-1.0, 2.0, t_shape[0]) if t_shape else jnp.asarray(0.5)
    want = POT_FN._gradient(xyz, t)
    assert want.shape == xyz_shape

    got = POT_Q._gradient(u.Q(xyz, "kpc"), u.Q(t, "Myr"))
    assert got.shape == xyz_shape
    assert jnp.allclose(got, want, rtol=1e-12, atol=0)
    # A time in other units is converted, and the call works inside `jit`.
    got = jax.jit(lambda x, t: POT_Q._gradient(x, t))(xyz, u.Q(t / 1e3, "Gyr"))
    assert jnp.allclose(got, want, rtol=1e-12, atol=0)


@pytest.mark.parametrize("pot", [POT_FN, POT_Q], ids=["unitless", "unit-aware"])
def test_orbit_with_function_velocity(pot):
    """Orbits integrate with a plain-function velocity, with or without units."""
    host = gp.HernquistPotential(m_tot=1e12, r_s=10, units="galactic")
    field = gd.fields.HamiltonianField(gp.CompositePotential(host=host, acc=pot))
    q = u.Q([[10.0, 0, 0], [0, 15, 0]], "kpc")
    p = u.Q([[0.0, 200, 0], [-180, 0, 0]], "km/s")
    soln = gd.OrbitSolver().solve(
        field, (q, p), u.Q(0.0, "Myr"), u.Q(-20.0, "Myr"), dt0=-1.0
    )
    assert jnp.all(jnp.isfinite(soln.ys[0]))
