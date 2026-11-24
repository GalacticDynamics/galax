"""Testing :mod:`galax.dynamics` module."""

import pytest

import galax.dynamics as gd
import galax.orbits as go


def test_all() -> None:
    """Test the `galax.dynamics` API."""
    assert set(gd.__all__) == {
        # Modules
        "cluster",
        "fields",
        "mockstream",
        "nbody",
        "orbit",
        "plot",
        "examples",
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
        "angular_speed",
        "parse_time_specification",
        # Diffraxtra compat
        "DiffEqSolver",
        # Legacy
        "integrate",
        "evaluate_orbit",
    }


def test_omega_is_deprecated_alias() -> None:
    """`gd.omega` still works but warns; unknown names still raise."""
    with pytest.warns(DeprecationWarning, match="use `angular_speed`"):
        assert gd.omega is go.angular_speed
    with pytest.raises(AttributeError):
        _ = gd.not_a_name
