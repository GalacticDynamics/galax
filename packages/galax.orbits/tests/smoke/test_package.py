"""Testing :mod:`galax.orbits` module."""

import galax.orbits as go


def test_all() -> None:
    """Test the `galax.orbits` API."""
    assert set(go.__all__) == {"AbstractOrbitField", "HamiltonianField"}
    for name in go.__all__:
        assert hasattr(go, name)
