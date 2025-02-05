"""Testing :mod:`galax.dynamics` module."""

import galax.dynamics as gd


def test_all() -> None:
    """Test the `galax.potential` API."""
    assert set(gd.__all__) == {
        # modules
        "fields",
        "solve",
        "integrate",
        "cluster",
        "mockstream",
        "plot",
        # solve
        "AbstractSolver",
        "DynamicsSolver",
        # orbit
        "Orbit",
        # integrate
        "evaluate_orbit",
        # mockstream
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
        "omega",
    }
