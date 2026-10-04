"""The galpy interop distribution registers itself."""

from importlib.metadata import entry_points


def test_declares_its_entry_point() -> None:
    eps = {ep.name: ep.value for ep in entry_points(group="galax.potential.interop")}
    assert eps.get("galpy") == "galax.interop.galpy.potential"


def test_declares_no_other_groups() -> None:
    """Galpy extends only `potential`; a stray entry point would be a bug."""
    for group in ("galax.coordinates.interop", "galax.dynamics.interop"):
        eps = {ep.name for ep in entry_points(group=group)}
        assert "galpy" not in eps
