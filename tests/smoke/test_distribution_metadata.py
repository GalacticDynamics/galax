"""`pip install galax` must keep meaning what it meant before the split."""

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

    Review Focus 1: a missing entry point produces no error at all, so this
    must be pinned by metadata as well as by behaviour.
    """
    assert "galax-interop-astropy" in _required_names()


@pytest.mark.parametrize(
    "name", ["galax-interop-gala", "galax-interop-galpy", "galax-interop-matplotlib"]
)
def test_heavy_interop_stays_optional(name: str) -> None:
    """gala, galpy and matplotlib were extras before the split and stay extras."""
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
def test_every_pre_split_extra_still_exists(extra: str) -> None:
    """Removing an extra breaks `pip install galax[...]` for existing users."""
    provided = metadata("galax").get_all("Provides-Extra") or []
    assert extra in provided
