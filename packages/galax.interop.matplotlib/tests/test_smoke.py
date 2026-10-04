"""The matplotlib interop distribution registers itself."""

from importlib.metadata import entry_points


def test_declares_its_entry_points() -> None:
    """Note the two groups point at *different* modules."""
    expected = {
        "galax.potential.interop": "galax.interop.matplotlib.potential",
        "galax.dynamics.interop": "galax.interop.matplotlib.orbit",
    }
    for group, value in expected.items():
        eps = {ep.name: ep.value for ep in entry_points(group=group)}
        assert eps.get("matplotlib") == value, f"{group} wrong matplotlib target"


def test_does_not_extend_coordinates() -> None:
    eps = {ep.name for ep in entry_points(group="galax.coordinates.interop")}
    assert "matplotlib" not in eps
