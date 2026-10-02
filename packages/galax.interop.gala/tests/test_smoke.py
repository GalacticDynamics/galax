"""The gala interop distribution registers itself and reports its build facts."""

import builtins
import sys
from importlib.metadata import entry_points

import pytest
from packaging.version import Version

from galax.interop.gala.optional_deps import (
    GALA_VERSION,
    GSL_ENABLED,
    _gsl_enabled,
)


def test_declares_its_entry_points() -> None:
    expected = {
        "galax.coordinates.interop": "galax.interop.gala.coordinates",
        "galax.potential.interop": "galax.interop.gala.potential",
    }
    for group, value in expected.items():
        eps = {ep.name: ep.value for ep in entry_points(group=group)}
        assert eps.get("gala") == value, f"{group} missing gala entry point"


def test_gsl_falls_back_when_gala_lacks_cconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gala built without GSL ships no `gala._cconfig`.

    CI's gala has GSL, so this branch is otherwise unreachable -- and an
    unreachable fallback is one nobody finds out is wrong. Simulated by making
    the import raise, which is exactly what such a build does.
    """
    real_import = builtins.__import__

    def no_cconfig(name: str, *args: object, **kwargs: object) -> object:
        if name == "gala._cconfig":
            msg = "No module named 'gala._cconfig'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_cconfig)
    monkeypatch.delitem(sys.modules, "gala._cconfig", raising=False)

    assert _gsl_enabled() is False


def test_build_facts_are_usable() -> None:
    """The two things a dependency pin cannot express.

    `gala>=1.10` is required here, so whether gala is installed is not in
    question. Whether it was *built against GSL* is, and so is its exact
    version -- several conversions are gated on 1.8.2 and 1.11.
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
