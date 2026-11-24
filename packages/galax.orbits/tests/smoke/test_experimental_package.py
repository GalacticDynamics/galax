"""Testing :mod:`galax.orbits.experimental` module."""

import galax.orbits.experimental as gox


def test_all() -> None:
    """Test the `galax.orbits.experimental` API."""
    assert set(gox.__all__) == {"SymplecticSolverT", "integrate_orbit"}
    for name in gox.__all__:
        assert hasattr(gox, name)
