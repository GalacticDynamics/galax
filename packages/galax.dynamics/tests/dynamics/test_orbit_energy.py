"""The energy functions, exercised against an `Orbit`.

`gp.potential_energy` and `gp.total_energy` dispatch on
`AbstractPhaseSpaceCoordinate`, so `Orbit` is covered by inheritance rather
than by its own registration. That is exactly the case worth a test: an orbit
carries a *vector* of times, so the shape and time-broadcast contract is
non-trivial in a way a single coordinate's is not.

These live in the dynamics tree, not beside the other energy tests, because
`Orbit` is a `galax.dynamics` type. `dynamics` may depend on `potential`; the
reverse is what the distribution split exists to prevent.
"""

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp


@pytest.fixture
def pot() -> gp.KeplerPotential:
    """Return a Kepler potential."""
    return gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")


@pytest.fixture
def orbit(pot: gp.KeplerPotential) -> gd.Orbit:
    """Return an orbit sampled at 10 times."""
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([8.0, 0.0, 0.0], "kpc"),
        p=u.Q([0.0, 220.0, 0.0], "km/s"),
        t=u.Q(0.0, "Myr"),
    )
    ts = u.Q(jnp.linspace(0.0, 100.0, 10), "Myr")
    return gd.evaluate_orbit(pot, w0, ts)


def test_potential_energy_follows_the_orbit(orbit, pot) -> None:
    """One energy per sampled time, matching the potential at each point."""
    got = gp.potential_energy(pot, orbit)

    assert got.shape == orbit.t.shape
    assert jnp.allclose(
        got, pot.potential(orbit.q, t=orbit.t), atol=u.Q(1e-10, got.unit)
    )


def test_total_energy_follows_the_orbit(orbit, pot) -> None:
    """Kinetic plus potential, per sampled time."""
    got = gp.total_energy(pot, orbit)

    assert got.shape == orbit.t.shape
    assert jnp.allclose(
        got,
        orbit.kinetic_energy() + gp.potential_energy(pot, orbit),
        atol=u.Q(1e-10, got.unit),
    )


def test_total_energy_is_conserved_along_a_kepler_orbit(orbit, pot) -> None:
    """The physical check the shape assertions cannot make.

    A Kepler orbit is a closed system, so its specific total energy is the same
    at every sampled time. This catches a q/t misalignment that a shape check
    would pass.
    """
    e = gp.total_energy(pot, orbit)

    # `atol` must carry e's unit: the default is a bare float, and unxt 2
    # refuses to compare one against a quantity.
    assert jnp.allclose(e, e[0], rtol=1e-6, atol=u.Q(1e-10, e.unit))
