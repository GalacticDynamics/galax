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
