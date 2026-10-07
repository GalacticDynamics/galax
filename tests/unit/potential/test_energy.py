"""The energy functions that take a potential and a coordinate."""

import jax.random as jr
import pytest

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.potential as gp

potentials = [
    gp.KeplerPotential(m_tot=1e12, units="galactic"),
    gp.MilkyWayPotential(),
]


@pytest.fixture(params=[(10,), (5, 4)], ids=str)
def w_batch(request) -> gc.PhaseSpaceCoordinate:
    """Return a random batched phase-space coordinate."""
    shape = request.param
    kq, kp, kt = jr.split(jr.key(0), 3)
    return gc.PhaseSpaceCoordinate(
        q=u.Q(jr.normal(kq, (*shape, 3)), "kpc"),
        p=u.Q(jr.normal(kp, (*shape, 3)), "km/s"),
        t=u.Q(jr.normal(kt, shape), "Myr"),
        frame=gc.frames.simulation_frame,
    )


@pytest.fixture
def w() -> gc.PhaseSpaceCoordinate:
    """Return a phase-space coordinate at 8 kpc on a circular-ish orbit."""
    return gc.PhaseSpaceCoordinate(
        q=cx.CartesianPos3D.from_(u.Q([8.0, 0.0, 0.0], "kpc")),
        p=cx.CartesianVel3D.from_(u.Q([0.0, 220.0, 0.0], "km/s")),
        t=u.Q(0.0, "Myr"),
    )


@pytest.fixture
def pot() -> gp.KeplerPotential:
    """Return a Kepler potential."""
    return gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")


def test_potential_energy_matches_the_potential(w, pot) -> None:
    """`potential_energy` is the potential evaluated at the coordinate."""
    got = gp.potential_energy(pot, w)
    assert got == pot.potential(w.q, t=w.t)


def test_potential_energy_is_negative(w, pot) -> None:
    """A Kepler well is negative everywhere outside the origin."""
    assert float(gp.potential_energy(pot, w).value) < 0


def test_total_energy_is_kinetic_plus_potential(w, pot) -> None:
    """`total_energy` composes the coordinate's kinetic term with the well."""
    got = gp.total_energy(pot, w)
    expect = w.kinetic_energy() + gp.potential_energy(pot, w)
    assert got == expect


def test_the_old_methods_are_gone(w, pot) -> None:
    """Removed, not deprecated -- calling them must fail, not quietly work."""
    assert not hasattr(w, "potential_energy")
    assert not hasattr(w, "total_energy")


@pytest.mark.parametrize("pot", potentials, ids=lambda p: type(p).__name__)
def test_potential_energy_batched(w_batch, pot) -> None:
    """`potential_energy` has the coordinate's shape and is `pot.potential`."""
    pe = gp.potential_energy(pot, w_batch)
    assert pe.shape == w_batch.shape
    assert jnp.all(pe <= u.Q(0, "km2/s2"))
    # definitional
    assert jnp.allclose(
        pe, pot.potential(w_batch.q, t=w_batch.t), atol=u.Q(1e-10, pe.unit)
    )


@pytest.mark.parametrize("pot", potentials, ids=lambda p: type(p).__name__)
def test_total_energy_batched(w_batch, pot) -> None:
    """`total_energy` has the coordinate's shape and is kinetic plus potential."""
    etot = gp.total_energy(pot, w_batch)
    assert etot.shape == w_batch.shape
    # definitional
    assert jnp.allclose(
        etot,
        w_batch.kinetic_energy() + pot.potential(w_batch.q, t=w_batch.t),
        atol=u.Q(1e-10, etot.unit),
    )
