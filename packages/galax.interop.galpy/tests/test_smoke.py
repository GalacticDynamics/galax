"""The galpy interop distribution registers itself."""

from importlib.metadata import entry_points

import pytest


def test_declares_its_entry_point() -> None:
    """Galpy contributes exactly one entry point, to `potential`."""
    eps = {ep.name: ep.value for ep in entry_points(group="galax.potential.interop")}
    assert eps.get("galpy") == "galax.interop.galpy.potential"


@pytest.mark.parametrize(
    "group", ["galax.coordinates.interop", "galax.dynamics.interop"]
)
def test_declares_no_other_groups(group: str) -> None:
    """Galpy extends only `potential`; a stray entry point would be a bug."""
    eps = {ep.name for ep in entry_points(group=group)}
    assert "galpy" not in eps
