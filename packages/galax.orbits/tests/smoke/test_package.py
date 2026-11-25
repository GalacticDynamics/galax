"""Testing :mod:`galax.orbits` module."""

import galax.orbits as go


def test_all() -> None:
    """Test the `galax.orbits` API."""
    assert set(go.__all__) == {
        "AbstractOrbit",
        "AbstractOrbitField",
        "HamiltonianField",
        "Orbit",
        "OrbitSolver",
        "PhaseSpaceInterpolation",
        "circulation",
        "compute_orbit",
        "angular_speed",
        "specific_angular_momentum",
    }
    for name in go.__all__:
        assert hasattr(go, name)
