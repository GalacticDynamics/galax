"""The energy functions that take a potential and a coordinate."""

import pytest

import coordinax as cx
import unxt as u

import galax.coordinates as gc
import galax.potential as gp


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
