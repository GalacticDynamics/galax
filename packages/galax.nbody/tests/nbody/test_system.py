"""Tests for `galax.nbody.Component` and `galax.nbody.System`."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import galax.nbody as gn


def _comp(name, n, *, kind="massive", m=1.0, offset=0.0):
    q = jnp.arange(3.0 * n).reshape(n, 3) + offset
    m = m if kind == "massive" else 0.0
    return gn.Component(name=name, q=q, p=-q, m=m, h=0.1, kind=kind)


def test_from_components_sorts_massive_first_and_keeps_ids() -> None:
    tracers = _comp("tracers", 3, kind="tracer")
    stars = _comp("stars", 2, offset=100.0)
    sys = gn.System.from_components(tracers, stars, units="galactic")

    assert sys.names == ("stars", "tracers")
    assert sys.slices == ((0, 2), (2, 5))
    assert sys.n_massive == 2
    np.testing.assert_array_equal(sys.particles.id, [3, 4, 0, 1, 2])
    np.testing.assert_array_equal(sys.particles.q[:2], stars.q)
    np.testing.assert_array_equal(sys.particles.m, [1, 1, 0, 0, 0])


def test_split_round_trips_by_name() -> None:
    a, b = _comp("a", 2), _comp("b", 3, kind="tracer", offset=10.0)
    sys = gn.System.from_components(a, b, units="galactic")
    parts = sys.split(sys.particles.q)
    np.testing.assert_array_equal(parts["a"], a.q)
    np.testing.assert_array_equal(parts["b"], b.q)


def test_scalar_mass_broadcasts() -> None:
    sys = gn.System.from_components(_comp("a", 4, m=2.5), units="galactic")
    assert sys.particles.m.shape == (4,)
    assert sys.particles.h.shape == (4,)
    np.testing.assert_array_equal(sys.particles.m, 2.5)


def test_rejects_duplicate_names_and_bad_shapes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        gn.System.from_components(_comp("a", 2), _comp("a", 1), units="galactic")

    q = jnp.zeros((2, 3))
    bad_p = gn.Component(name="x", q=q, p=jnp.zeros((3, 3)), m=1.0, h=0.0)
    with pytest.raises(ValueError, match="shape"):
        gn.System.from_components(bad_p, units="galactic")

    bad_m = gn.Component(name="x", q=q, p=q, m=jnp.ones(3), h=0.0)
    with pytest.raises(ValueError, match="shape"):
        gn.System.from_components(bad_m, units="galactic")

    neg = gn.Component(name="x", q=q, p=q, m=-1.0, h=0.0)
    with pytest.raises(ValueError, match="negative"):
        gn.System.from_components(neg, units="galactic")

    for bad in (
        {"m": jnp.nan, "h": 0.0},
        {"m": 1.0, "h": jnp.nan},
        {"m": jnp.inf, "h": 0.0},
        {"m": 1.0, "h": jnp.inf},
    ):
        nan = gn.Component(name="x", q=q, p=q, **bad)
        with pytest.raises(ValueError, match="negative"):
            gn.System.from_components(nan, units="galactic")

    massless = gn.Component(name="x", q=q, p=q, m=1.0, h=0.0, kind="tracer")
    with pytest.raises(ValueError, match="is a tracer but has nonzero m"):
        gn.System.from_components(massless, units="galactic")


def test_quantities_converted_to_units() -> None:
    c = gn.Component(
        name="c",
        q=u.Q([[1.0, 2, 3]], "pc"),
        p=u.Q([[1.0, 0, 0]], "km/s"),
        m=u.Q(1e6, "Msun"),
        h=u.Q(10.0, "pc"),
    )
    sys = gn.System.from_components(c, units="galactic")
    us = u.unitsystem("galactic")
    np.testing.assert_allclose(sys.particles.q, [[1e-3, 2e-3, 3e-3]], rtol=1e-12)
    vel = u.Q(1.0, "km/s").ustrip(us["speed"])
    np.testing.assert_allclose(sys.particles.p, [[vel, 0, 0]], rtol=1e-12)
    np.testing.assert_allclose(sys.particles.m, [1e6], rtol=1e-12)
    np.testing.assert_allclose(sys.particles.h, [1e-2], rtol=1e-12)
    assert sys.units == us


def test_system_serialises_round_trip(tmp_path) -> None:
    def build(scale):
        a = _comp("a", 2, m=scale)
        b = _comp("b", 3, kind="tracer", offset=scale)
        return gn.System.from_components(a, b, units="galactic")

    sys = build(3.0)
    path = tmp_path / "system.eqx"
    eqx.tree_serialise_leaves(path, sys)
    loaded = eqx.tree_deserialise_leaves(path, build(1.0))

    for x, y in zip(
        jax.tree.leaves(sys.particles), jax.tree.leaves(loaded.particles), strict=True
    ):
        np.testing.assert_array_equal(x, y)
    assert loaded.names == sys.names
    assert loaded.n_massive == sys.n_massive


def test_rejects_zero_d_q_and_negative_h() -> None:
    zero_d = gn.Component(
        name="x", q=jnp.asarray(0.0), p=jnp.asarray(0.0), m=1.0, h=0.0
    )
    with pytest.raises(ValueError, match="shapes"):
        gn.System.from_components(zero_d, units="galactic")

    q = jnp.zeros((2, 3))
    neg_h = gn.Component(name="x", q=q, p=q, m=1.0, h=-0.1)
    with pytest.raises(ValueError, match="negative or non-finite h"):
        gn.System.from_components(neg_h, units="galactic")


def test_component_kind_is_checked() -> None:
    q = jnp.zeros((1, 3))
    with pytest.raises(ValueError, match="kind must be one of"):
        gn.Component(name="x", q=q, p=q, m=0.0, h=0.0, kind="virtual")


def test_two_massive_species_keep_their_masses_and_softening() -> None:
    """Dark matter and stars: both massive, each with its own m and h."""
    q = jnp.arange(12.0).reshape(4, 3)
    dm = gn.Component(name="dm", q=q[:3], p=q[:3], m=1.25e6, h=0.2)
    stars = gn.Component(name="stars", q=q[3:], p=q[3:], m=5e4, h=0.05)
    sys = gn.System.from_components(dm, stars, units="galactic")
    assert sys.n_massive == 4
    np.testing.assert_array_equal(sys.particles.m, [1.25e6] * 3 + [5e4])
    np.testing.assert_array_equal(sys.particles.h, [0.2] * 3 + [0.05])
