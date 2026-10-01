"""The astropy interop distribution registers itself and converts."""

from importlib.metadata import entry_points

import astropy.units as apyu

import unxt as u

import galax.potential as gp


def test_declares_its_entry_points() -> None:
    """All three groups must name this distribution's modules."""
    expected = {
        "galax.coordinates.interop": "galax.interop.astropy.coordinates",
        "galax.potential.interop": "galax.interop.astropy.potential",
        "galax.dynamics.interop": "galax.interop.astropy.dynamics",
    }
    for group, value in expected.items():
        eps = {ep.name: ep.value for ep in entry_points(group=group)}
        assert eps.get("astropy") == value, f"{group} missing astropy entry point"


def test_registration_actually_happened() -> None:
    """Assert behaviour, not import.

    An entry point that loads but registers nothing produces silence, so
    checking the module imported is not evidence the interop works.
    """
    pot = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10, "kpc"), units="galactic"
    )
    got = pot.potential(
        apyu.Quantity([8.0, 0.0, 0.0], "kpc"), apyu.Quantity(0.0, "Gyr")
    )
    assert got is not None
