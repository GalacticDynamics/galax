"""Testing :mod:`galax.orbits.integrate` module."""

import galax.orbits.integrate as goi


def test_all() -> None:
    """Test the `galax.orbits.integrate` API."""
    assert set(goi.__all__) == {
        "evaluate_orbit",
        "Integrator",
        "PhaseSpaceInterpolation",
        "parse_time_specification",
        "InterpolatedPhaseSpaceCoordinate",
        "default_integrator",
    }
    for name in goi.__all__:
        assert hasattr(goi, name)
