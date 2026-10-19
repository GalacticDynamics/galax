"""Testing :mod:`galax.dynamics.integrate` module."""

from galax.dynamics import integrate


def test_all() -> None:
    """Test the `galax.dynamics.integrate` API."""
    assert set(integrate.__all__) == {
        "evaluate_orbit",
        "Integrator",
        "PhaseSpaceInterpolation",
        "parse_time_specification",
        "InterpolatedPhaseSpaceCoordinate",
    }
