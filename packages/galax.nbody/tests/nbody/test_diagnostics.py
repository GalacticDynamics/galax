"""Tests for `galax.nbody.total_energy`."""

import jax.numpy as jnp

import unxt as u

import galax.nbody as gn
import galax.potential as gp

_NULL = gp.NullPotential(units="galactic")


def test_two_body_energy_matches_analytic() -> None:
    """E = 1/2 m1 v1^2 + 1/2 m2 v2^2 - G m1 m2 / sqrt(r^2 + h^2)."""
    m1, m2, h, r = 2.0, 3.0, 0.1, 1.5
    field = gn.NBodyField(
        masses=u.Q([m1, m2], "Msun"), eps=u.Q(h, "kpc"), external_potential=_NULL
    )
    q = jnp.asarray([[0.0, 0, 0], [r, 0, 0]])
    p = jnp.asarray([[0.0, 1, 0], [0.5, 0, 0]])
    G = field._G
    expect = 0.5 * m1 * 1.0 + 0.5 * m2 * 0.25 - G * m1 * m2 / jnp.sqrt(r**2 + h**2)
    assert jnp.allclose(gn.total_energy(field, 0.0, q, p), expect)


def test_tracers_add_no_self_energy() -> None:
    """Massless rows contribute zero kinetic and zero self energy."""
    kw = {"eps": u.Q(0.1, "kpc"), "external_potential": _NULL}
    two = gn.NBodyField(masses=u.Q([2.0, 3.0], "Msun"), **kw)
    three = gn.NBodyField(masses=u.Q([2.0, 3.0, 0.0], "Msun"), n_massive=2, **kw)
    q = jnp.asarray([[0.0, 0, 0], [1.5, 0, 0]])
    p = jnp.asarray([[0.0, 1, 0], [0.5, 0, 0]])
    q3 = jnp.concat([q, jnp.asarray([[0.0, 4, 0]])])
    p3 = jnp.concat([p, jnp.asarray([[7.0, 7, 7]])])
    assert jnp.allclose(
        gn.total_energy(three, 0.0, q3, p3), gn.total_energy(two, 0.0, q, p)
    )


def test_energy_with_kepler_external_potential() -> None:
    """E = sum 1/2 m v^2 - G m1 m2 / sqrt(r^2 + h^2) - sum G M m / |q|."""
    M, m1, m2, h = 1e11, 2.0, 3.0, 0.1
    pot = gp.KeplerPotential(m_tot=M, units="galactic")
    field = gn.NBodyField(
        masses=u.Q([m1, m2], "Msun"), eps=u.Q(h, "kpc"), external_potential=pot
    )
    q = jnp.asarray([[3.0, 0, 0], [0.0, 4, 0]])
    p = jnp.asarray([[0.0, 1, 0], [0.5, 0, 0]])
    G = field._G
    kinetic = 0.5 * m1 * 1.0 + 0.5 * m2 * 0.25
    self_pot = -G * m1 * m2 / jnp.sqrt(5.0**2 + h**2)
    ext = -G * M * (m1 / 3.0 + m2 / 4.0)
    expect = kinetic + self_pot + ext
    assert jnp.allclose(gn.total_energy(field, 0.0, q, p), expect, rtol=1e-12)


def test_inactive_nonfinite_rows_are_ignored() -> None:
    """NaN or inf positions and velocities on an inactive row leave E finite."""
    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    kw = {"eps": u.Q(0.1, "kpc"), "external_potential": pot}
    two = gn.NBodyField(masses=u.Q([2.0, 3.0], "Msun"), **kw)
    three = gn.NBodyField(
        masses=u.Q([2.0, 3.0, 4.0], "Msun"), active=[True, True, False], **kw
    )
    q = jnp.asarray([[3.0, 0, 0], [0.0, 4, 0]])
    p = jnp.asarray([[0.0, 1, 0], [0.5, 0, 0]])
    expect = gn.total_energy(two, 0.0, q, p)
    for bad in (jnp.nan, jnp.inf):
        q3 = jnp.concat([q, jnp.full((1, 3), bad)])
        p3 = jnp.concat([p, jnp.full((1, 3), bad)])
        assert jnp.allclose(gn.total_energy(three, 0.0, q3, p3), expect)


def test_total_energy_ignores_non_finite_tracer() -> None:
    """A tracer that has gone NaN or inf does not spoil the energy."""
    pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    field = gn.NBodyField(
        masses=jnp.asarray([1e9, 1e9, 0.0]),
        eps=0.1,
        external_potential=pot,
        n_massive=2,
    )
    q = jnp.asarray([[8.0, 0.0, 0.0], [-8.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    p = jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.0]])
    e = gn.total_energy(field, 0.0, q, p)
    bad_q = q.at[2].set(jnp.inf)
    bad_p = p.at[2].set(jnp.nan)
    assert gn.total_energy(field, 0.0, bad_q, bad_p) == e
