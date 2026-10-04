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

    The load-bearing assertion is that the result is a *unxt* quantity rather
    than an astropy one: that is what shows the astropy inputs were converted
    into galax's own types and evaluated, not merely passed through. Checking
    only that something non-`None` came back would pass on a pass-through.
    """
    pot = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10, "kpc"), units="galactic"
    )
    got = pot.potential(
        apyu.Quantity([8.0, 0.0, 0.0], "kpc"), apyu.Quantity(0.0, "Gyr")
    )

    assert isinstance(got, u.quantity.AbstractQuantity)
    assert not isinstance(got, apyu.Quantity)

    # Converted into the potential's own unit system, not left in the input's.
    assert u.dimension_of(got) == u.dimension("specific energy")
    assert got.unit == pot.units["specific energy"]

    # A Hernquist well is finite and negative everywhere outside the origin,
    # so this also catches a conversion that silently produced nan.
    assert got.shape == ()
    assert float(got.value) < 0
