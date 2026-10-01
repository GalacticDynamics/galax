"""The gala interop distribution registers itself, and degrades without gala."""

from importlib.metadata import entry_points

import pytest

from galax.interop.gala.optional_deps import GSL_ENABLED, OptDeps


def test_declares_its_entry_points() -> None:
    expected = {
        "galax.coordinates.interop": "galax.interop.gala.coordinates",
        "galax.potential.interop": "galax.interop.gala.potential",
    }
    for group, value in expected.items():
        eps = {ep.name: ep.value for ep in entry_points(group=group)}
        assert eps.get("gala") == value, f"{group} missing gala entry point"


def test_gsl_enabled_is_a_bool_either_way() -> None:
    """`GSL_ENABLED` answers `False` without gala rather than raising."""
    assert isinstance(GSL_ENABLED, bool)
    if not OptDeps.GALA.installed:
        assert GSL_ENABLED is False


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
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
