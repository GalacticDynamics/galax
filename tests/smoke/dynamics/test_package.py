"""Testing :mod:`galax.dynamics` module."""

import galax.dynamics as gd


def test_all() -> None:
    """Test the `galax.potential` API."""
    assert set(gd.__all__) == {
        # modules
        "integrate",
        "mockstream",
        "plot",
        # core
        "Orbit",
        "evaluate_orbit",
        "MockStreamArm",
        "MockStream",
        "MockStreamGenerator",
        "AbstractStreamDF",
        "ChenStreamDF",
        "FardalStreamDF",
        # functions
        "specific_angular_momentum",
        "tidal_radius",
        "lagrange_points",
    }
