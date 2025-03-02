"""Testing :mod:`galax.dynamics` module."""

import galax.dynamics as gd


def test_all() -> None:
    """Test the `galax.potential` API."""
    assert set(gd.__all__) == {
        # Modules
        "cluster",
        "fields",
        "mockstream",
        "orbit",
        "plot",
        # fields
        "AbstractField",
        "AbstractOrbitField",
        "HamiltonianField",
        "NBodyField",
        # solver
        "AbstractSolver",
        "SolveState",
        "integrate_field",
        # orbit
        "compute_orbit",
        "Orbit",
        "OrbitSolver",
        # mockstream
        "MockStreamArm",
        "MockStream",
        "MockStreamGenerator",
        # mockstream.df
        "AbstractStreamDF",
        "FardalStreamDF",
        "ChenStreamDF",
        # functions
        "specific_angular_momentum",
        "lagrange_points",
        "tidal_radius",
        "omega",
        "parse_time_specification",
        # Diffraxtra compat
        "DiffEqSolver",
        # Legacy
        "integrate",
        "evaluate_orbit",
    }
