"""The gala interop distribution registers itself and reports its build facts."""

from importlib.metadata import entry_points

from packaging.version import Version

from galax.interop.gala.optional_deps import (
    GALA_VERSION,
    GSL_ENABLED,
)


def test_declares_its_entry_points() -> None:
    expected = {
        "galax.coordinates.interop": "galax.interop.gala.coordinates",
        "galax.potential.interop": "galax.interop.gala.potential",
    }
    for group, value in expected.items():
        eps = {ep.name: ep.value for ep in entry_points(group=group)}
        assert eps.get("gala") == value, f"{group} missing gala entry point"


def test_build_facts_are_usable() -> None:
    """The two things a dependency pin cannot express.

    `gala>=1.10` is required here, so whether gala is installed is not in
    question. Whether it was *built against GSL* is, and so is its exact
    version -- the MilkyWayPotential conversion is gated on 1.11.
    """
    assert isinstance(GSL_ENABLED, bool)
    assert Version("1.10") <= GALA_VERSION


def test_registration_actually_happened() -> None:
    """Assert converted behaviour, not import."""
    import gala.potential as galap

    import unxt as u

    import galax.potential as gp

    pot = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10, "kpc"), units="galactic"
    )
    converted = gp.io.convert_potential(gp.io.GalaLibrary, pot)
    assert isinstance(converted, galap.PotentialBase)
