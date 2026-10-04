"""The matplotlib interop distribution registers itself."""

from importlib.metadata import entry_points

import pytest


@pytest.mark.parametrize(
    ("group", "value"),
    [
        ("galax.potential.interop", "galax.interop.matplotlib.potential"),
        ("galax.dynamics.interop", "galax.interop.matplotlib.orbit"),
    ],
)
def test_declares_its_entry_points(group: str, value: str) -> None:
    """Note the two groups point at *different* modules."""
    eps = {ep.name: ep.value for ep in entry_points(group=group)}
    assert eps.get("matplotlib") == value, f"{group} wrong matplotlib target"


def test_does_not_extend_coordinates() -> None:
    """Matplotlib plots potentials and orbits; it adds no coordinate frames."""
    eps = {ep.name for ep in entry_points(group="galax.coordinates.interop")}
    assert "matplotlib" not in eps
