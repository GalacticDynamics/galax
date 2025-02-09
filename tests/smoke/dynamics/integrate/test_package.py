"""Test the :mod:`galax.integrate` module."""

from galax.dynamics import integrate


def test_all() -> None:
    """Test the API."""
    assert set(integrate.__all__) == {
        "evaluate_orbit",
        "Integrator",
        "Interpolant",
        "parse_time_specification",
        "InterpolatedPhaseSpaceCoordinate",
    }
