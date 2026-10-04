"""The gala interop distribution registers itself and reports its build facts."""

from importlib.metadata import entry_points

import gala.potential as galap
import pytest
from packaging.version import Version

import unxt as u

import galax.potential as gp
from galax.interop.gala.optional_deps import GALA_VERSION, GSL_ENABLED


@pytest.mark.parametrize(
    ("group", "value"),
    [
        ("galax.coordinates.interop", "galax.interop.gala.coordinates"),
        ("galax.potential.interop", "galax.interop.gala.potential"),
    ],
)
def test_declares_its_entry_points(group: str, value: str) -> None:
    """Both groups must name this distribution's modules."""
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
    pot = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10, "kpc"), units="galactic"
    )
    converted = gp.io.convert_potential(gp.io.GalaLibrary, pot)
    assert isinstance(converted, galap.PotentialBase)
