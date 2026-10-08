"""`pip install galax` installs the core portions and the astropy interop.

gala, galpy and matplotlib stay optional, behind extras.
"""

from importlib.metadata import metadata, requires

import pytest


def _required_names() -> set[str]:
    """Distribution names `galax` requires unconditionally."""
    return {
        r.split(";")[0].split(">")[0].split("[")[0].strip().lower().replace("_", "-")
        for r in (requires("galax") or [])
        if "extra ==" not in r
    }


def test_astropy_interop_is_required_not_an_extra() -> None:
    """Demoting it would silently stop astropy conversions registering.

    A missing entry point produces no error at all, so this must be pinned by
    metadata as well as by behaviour.
    """
    assert "galax-interop-astropy" in _required_names()


@pytest.mark.parametrize(
    "name", ["galax-interop-gala", "galax-interop-galpy", "galax-interop-matplotlib"]
)
def test_heavy_interop_stays_optional(name: str) -> None:
    """gala, galpy and matplotlib are extras, not required dependencies."""
    assert name not in _required_names()


@pytest.mark.parametrize(
    "extra",
    [
        "all",
        "interop-all",
        "interop-astropy",
        "interop-gala",
        "interop-galpy",
        "plot-all",
        "plot-matplotlib",
    ],
)
def test_every_published_extra_still_exists(extra: str) -> None:
    """Removing an extra breaks `pip install galax[...]` for existing users."""
    provided = metadata("galax").get_all("Provides-Extra") or []
    assert extra in provided


def test_coordinates_is_required_not_an_extra() -> None:
    """`pip install galax` must still bring the coordinates portion.

    Demoting it to an extra would leave `import galax.coordinates` failing on a
    default install, with nothing in the metadata to say why.
    """
    assert "galax-coordinates" in _required_names()


def test_potential_is_required_not_an_extra() -> None:
    """`pip install galax` must still bring the potential portion.

    Same reasoning as the coordinates portion above: demoting it to an extra
    would leave `import galax.potential` failing on a default install, with
    nothing in the metadata to say why.
    """
    assert "galax-potential" in _required_names()
