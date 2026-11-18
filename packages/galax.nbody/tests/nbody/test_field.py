"""Tests for `galax.nbody.NBodyField`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    field = gn.NBodyField(jnp.zeros(4), eps=0.01, external_potential=pot, n_massive=0)
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


@pytest.mark.parametrize("n_massive", [-1, 3])
def test_n_massive_out_of_range_raises(n_massive: int) -> None:
    with pytest.raises(ValueError, match="n_massive"):
        gn.NBodyField(
            masses=jnp.zeros(2),
            eps=0.0,
            external_potential=gp.NullPotential(units="galactic"),
            n_massive=n_massive,
        )


def test_nonzero_mass_on_tracer_row_raises() -> None:
    pot = gp.NullPotential(units="galactic")
    with pytest.raises(ValueError, match="massless tracers"):
        gn.NBodyField(jnp.array([1.0, 1.0, 1.0]), 0.0, pot, n_massive=2)
    # Zero masses on the tracer rows are fine.
    gn.NBodyField(jnp.array([1.0, 1.0, 0.0]), 0.0, pot, n_massive=2)


def test_tracer_mass_check_is_skipped_under_tracing() -> None:
    pot = gp.NullPotential(units="galactic")

    @jax.jit
    def build(m):
        return gn.NBodyField(m, 0.0, pot, n_massive=1).masses

    np.testing.assert_array_equal(build(jnp.array([1.0, 1.0])), [1.0, 1.0])


@pytest.mark.parametrize(
    "masses",
    [jnp.array([1.0, 2.0, 0.0]), u.Q([1.0, 2.0, 0.0], "Msun")],
    ids=["array", "quantity"],
)
@pytest.mark.parametrize("n_massive", [None, 2])
def test_closed_over_masses_build_under_jit(masses, n_massive) -> None:
    """Concrete masses closed over by a jitted function don't stage the check."""
    pot = gp.NullPotential(units="galactic")
    q = jnp.eye(3)

    @jax.jit
    def acc(q):
        return gn.NBodyField(masses, 0.01, pot, n_massive=n_massive).acceleration(
            0.0, q
        )

    expected = gn.NBodyField(masses, 0.01, pot, n_massive=n_massive).acceleration(
        0.0, q
    )
    np.testing.assert_allclose(acc(q), expected, rtol=1e-12)


def test_closed_over_tracer_mass_still_raises_under_jit() -> None:
    pot = gp.NullPotential(units="galactic")
    masses = jnp.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="massless tracers"):
        jax.jit(lambda q: gn.NBodyField(masses, 0.01, pot, n_massive=2).masses + q)(1.0)


@pytest.mark.parametrize("masses", [jnp.ones((3, 1)), jnp.asarray(1.0)])
def test_masses_must_be_1d(masses) -> None:
    with pytest.raises(ValueError, match="1-D"):
        gn.NBodyField(masses, 0.01, gp.NullPotential(units="galactic"))


def test_negative_masses_or_softening_raise() -> None:
    pot = gp.NullPotential(units="galactic")
    with pytest.raises(ValueError, match="masses"):
        gn.NBodyField(jnp.array([1.0, -1.0]), 0.01, pot)
    with pytest.raises(ValueError, match="softening"):
        gn.NBodyField(jnp.ones(2), -0.01, pot)


def test_n_massive_must_be_int() -> None:
    with pytest.raises(TypeError):
        gn.NBodyField(
            jnp.ones(2), 0.0, gp.NullPotential(units="galactic"), n_massive=2.0
        )


# Input forms for closed-over construction under jit. Each pair is a valid and
# an invalid (negative) version, plus the potential that sets the units.
_SOLAR = gp.NullPotential(units="solarsystem")
_GAL = gp.NullPotential(units="galactic")
_FORMS = {
    "numpy": (lambda x: np.array(x, dtype=float), _GAL),
    "list": (lambda x: [float(v) for v in x], _GAL),
    "int-jnp": (lambda x: jnp.array(x, dtype=int), _GAL),
    "float32": (lambda x: np.array(x, dtype=np.float32), _GAL),
    "int-quantity": (lambda x: u.Q(list(x), "Msun"), _SOLAR),
    "converted-quantity": (lambda x: u.Q(np.array(x, dtype=float), "kg"), _GAL),
}


@pytest.mark.parametrize("form", list(_FORMS))
def test_closed_over_invalid_masses_raise_under_jit(form) -> None:
    make, pot = _FORMS[form]
    negative, on_tracer = make([-1, 1]), make([1, 1, 1])
    with pytest.raises(ValueError, match="non-negative"):
        jax.jit(lambda: gn.NBodyField(negative, 0.1, pot).masses)()
    with pytest.raises(ValueError, match="massless tracers"):
        jax.jit(lambda: gn.NBodyField(on_tracer, 0.1, pot, n_massive=2).masses)()


@pytest.mark.parametrize("form", list(_FORMS))
def test_closed_over_valid_masses_match_eager_under_jit(form) -> None:
    make, pot = _FORMS[form]
    m = make([1, 2, 0])
    eager = gn.NBodyField(m, 0.1, pot, n_massive=2).masses
    jitted = jax.jit(lambda: gn.NBodyField(m, 0.1, pot, n_massive=2).masses)()
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)


@pytest.mark.parametrize(
    "eps", [-0.1, np.array(-0.1), [0.1, -0.1], u.Q(-1.0, "pc")], ids=str
)
def test_closed_over_negative_softening_raises_under_jit(eps) -> None:
    with pytest.raises(ValueError, match="softening"):
        jax.jit(lambda: gn.NBodyField(np.ones(2), eps=eps).masses)()


def test_traced_masses_skip_checks() -> None:
    def build(m):
        return gn.NBodyField(m, 0.1, _GAL, n_massive=1).masses

    # Invalid values pass through when traced, since they can't be checked.
    m = jnp.array([-1.0, 1.0])
    np.testing.assert_array_equal(jax.jit(build)(m), m)
    np.testing.assert_array_equal(jax.vmap(build)(jnp.stack([m, m])), [m, m])
    jaxpr = jax.make_jaxpr(build)(jnp.array([1.0, 0.0]))
    assert not jaxpr.eqns


@pytest.mark.parametrize("active", [[True], [True, False, True], np.ones((2, 1))])
@pytest.mark.parametrize("jit", [False, True])
def test_active_shape_must_match_masses(active, jit) -> None:
    def build():
        return gn.NBodyField(np.ones(2), 0.1, _GAL, active=active).active

    with pytest.raises(ValueError, match="active"):
        jax.jit(build)() if jit else build()
