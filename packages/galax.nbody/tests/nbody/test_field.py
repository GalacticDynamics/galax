"""Tests for `galax.nbody.NBodyField`."""

import jax
import jax.numpy as jnp
import numpy as np

import unxt as u

import galax.nbody as gn
import galax.potential as gp


def test_pairwise_accelerations_conserve_momentum() -> None:
    """With no external potential, the mass-weighted accelerations sum to zero."""
    field = gn.NBodyField(
        masses=u.Q([1.0, 2.0, 3.0], "Msun"),
        eps=u.Q(1e-3, "AU"),
        external_potential=gp.NullPotential(units="solarsystem"),
    )
    q = u.Q([[0.0, 0, 0], [1, 0, 0], [0, 2, 1]], "AU")
    p = u.Q(jnp.zeros((3, 3)), "km/s")

    v, a = field(0.0, (q, p), None)

    assert jnp.allclose(v, 0)
    assert jnp.allclose(jnp.sum(jnp.asarray([1.0, 2, 3])[:, None] * a, axis=0), 0)
    # The two closest bodies pull toward each other.
    assert a[0, 0] > 0
    assert a[1, 0] < 0


def _old_formula(field_units, G, masses, eps, pot, x, t):
    """Return the pre-generalisation NBodyField acceleration, kept for reference."""
    diffs = x[:, None, :] - x[None, :, :]
    soft_d2s = jnp.sum(diffs**2, axis=-1)[:, :, None] + eps**2
    soft_d3s = soft_d2s * jnp.sqrt(soft_d2s)
    m2s = masses[:, None, None] * masses[None, :, None]
    forces = G * m2s / soft_d3s * diffs
    forces = forces * (1 - jnp.eye(len(x))[:, :, None])
    a_self = jnp.sum(forces, axis=0) / masses[:, None]
    return a_self - pot._gradient(x, t)


Q4 = jnp.array([[0.0, 0, 0], [1, 0, 0], [0, 2, 1], [-1, 0.5, 0]])


def test_old_constructor_reproduces_old_formula() -> None:
    pot = gp.NullPotential(units="solarsystem")
    field = gn.NBodyField(
        masses=u.Q([1.0, 2.0, 3.0, 0.5], "Msun"),
        eps=u.Q(1e-3, "AU"),
        external_potential=pot,
    )
    us = pot.units
    G = pot.constants["G"].ustrip(us["length"] ** 3 / (us["mass"] * us["time"] ** 2))
    expected = _old_formula(
        us, G, jnp.array([1.0, 2.0, 3.0, 0.5]), 1e-3, pot, Q4, jnp.array(0.0)
    )
    _, a = field(0.0, (Q4, jnp.zeros_like(Q4)), None)
    np.testing.assert_allclose(a, expected, rtol=1e-12)


def test_zero_mass_particle_is_finite_and_matches_test_particle() -> None:
    pot = gp.NullPotential(units="galactic")
    q = Q4[:2]
    field = gn.NBodyField(
        masses=jnp.array([1e9, 0.0]), eps=0.01, external_potential=pot
    )
    tracer = gn.NBodyField(
        masses=jnp.array([1e9, 0.0]), eps=0.01, external_potential=pot, n_massive=1
    )
    a = field.acceleration(0.0, q)
    assert bool(jnp.all(jnp.isfinite(a)))
    np.testing.assert_allclose(a, tracer.acceleration(0.0, q), rtol=1e-12)


def test_inactive_particles_neither_pull_nor_feel() -> None:
    pot = gp.NullPotential(units="galactic")
    masses = jnp.array([1e9, 2e9, 3e9, 4e9])
    field = gn.NBodyField(
        masses,
        external_potential=pot,
        softening=jnp.full(4, 0.01),
        active=jnp.array([True, False, True, True]),
    )
    keep = jnp.array([0, 2, 3])
    sub = gn.NBodyField(masses[keep], 0.01, pot)
    a = field.acceleration(0.0, Q4)
    np.testing.assert_array_equal(a[1], 0.0)
    np.testing.assert_allclose(a[keep], sub.acceleration(0.0, Q4[keep]), rtol=1e-12)

    v_in = jnp.ones_like(Q4)
    v, _ = field(0.0, (Q4, v_in), None)
    np.testing.assert_array_equal(v, v_in)


def test_all_tracers_feel_only_external() -> None:
    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    field = gn.NBodyField(
        jnp.zeros(4) + 1e9, eps=0.01, external_potential=pot, n_massive=0
    )
    a = field.acceleration(0.0, Q4 + 1.0)
    np.testing.assert_allclose(
        a, -gp.gradient(pot, Q4 + 1.0, jnp.array(0.0)), rtol=1e-12
    )


def test_from_system_matches_direct_construction() -> None:
    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    stars = gn.Component(
        name="stars",
        q=u.Q(Q4[:2], "kpc"),
        p=u.Q(Q4[:2], "km/s"),
        m=u.Q(1e9, "Msun"),
        h=u.Q(10.0, "pc"),
    )
    tracers = gn.Component(
        name="tracers",
        q=u.Q(Q4[2:], "kpc"),
        p=u.Q(Q4[2:], "km/s"),
        m=0.0,
        h=0.0,
        massive=False,
    )
    # Build the system in different units from the potential.
    sys = gn.System.from_components(tracers, stars, units="solarsystem")
    field = gn.NBodyField.from_system(sys, pot)
    direct = gn.NBodyField(
        jnp.array([1e9, 1e9, 0, 0]),
        external_potential=pot,
        softening=jnp.array([0.01, 0.01, 0, 0]),
        n_massive=2,
    )
    q = jnp.concat([Q4[:2], Q4[2:]])
    assert field.n_massive == 2
    np.testing.assert_allclose(field.masses, direct.masses, rtol=1e-12)
    np.testing.assert_allclose(field.softening, direct.softening, rtol=1e-12)
    np.testing.assert_allclose(
        field.acceleration(0.0, q), direct.acceleration(0.0, q), rtol=1e-12
    )


def test_stopgrad_backend_still_differentiates_external() -> None:
    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    field = gn.NBodyField(
        jnp.full(4, 1e9),
        eps=0.01,
        external_potential=pot,
        backend=gn.DirectSum(grad="stop"),
    )
    g = jax.grad(lambda q: jnp.sum(field.acceleration(0.0, q)))(Q4 + 1.0)
    ext = jax.grad(lambda q: jnp.sum(-gp.gradient(pot, q, jnp.array(0.0))))(Q4 + 1.0)
    assert bool(jnp.any(g != 0))
    np.testing.assert_allclose(g, ext, rtol=1e-12)


def test_inactive_rows_singular_or_nan_stay_safe() -> None:
    """Inactive rows at the origin or at NaN poison neither values nor grads."""
    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    masses = jnp.array([1e9, 2e9, 3e9, 4e9])
    active = jnp.array([True, False, True, False])
    field = gn.NBodyField(masses, 0.01, pot, active=active)
    q = (Q4 + 1.0).at[1].set(0.0).at[3].set(jnp.nan)
    keep = jnp.array([0, 2])
    sub = gn.NBodyField(masses[keep], 0.01, pot)

    a = field.acceleration(0.0, q)
    assert bool(jnp.all(jnp.isfinite(a)))
    np.testing.assert_array_equal(a[1], 0.0)
    np.testing.assert_array_equal(a[3], 0.0)
    np.testing.assert_allclose(a[keep], sub.acceleration(0.0, q[keep]), rtol=1e-12)

    g = jax.grad(lambda q: jnp.sum(field.acceleration(0.0, q)))(q)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_stopgrad_on_mass_keeps_gradient_wrt_G() -> None:
    """G multiplies outside the gradient policy, so StopGrad(m) keeps dG."""
    us = u.unitsystem("galactic")
    G0 = gp.NullPotential(units=us).constants["G"]
    masses, q = jnp.array([1e9, 2e9, 3e9, 4e9]), Q4
    backend = gn.DirectSum(grad=gn.StopGrad(wrt=frozenset({"m"})))

    def total(scale):
        pot = gp.NullPotential(units=us, constants={"G": G0 * scale})
        field = gn.NBodyField(masses, 0.01, pot, backend=backend)
        return jnp.sum(field.acceleration(0.0, q))

    # The pairwise acceleration is linear in G, so d total / d scale = total.
    value, dG = jax.value_and_grad(total)(1.0)
    assert value != 0
    np.testing.assert_allclose(dG, value, rtol=1e-12)
    plain = gn.NBodyField(masses, 0.01, gp.NullPotential(units=us))
    np.testing.assert_allclose(value, jnp.sum(plain.acceleration(0.0, q)), rtol=1e-12)
