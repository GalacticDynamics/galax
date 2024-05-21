"""Testing :mod:`galax.dynamics` module."""

import galax.dynamics as gd


def test_all() -> None:
    """Test the `galax.potential` API."""
    assert set(gd.__all__) == {
        # TODO: it would be better to instead find and read the __all__ from the
        #       .pyi file
        "integrate",
        "mockstream",
        "AbstractOrbit",
        "Orbit",
        "InterpolatedOrbit",
        "evaluate_orbit",
        "MockStreamArm",
        "MockStream",
        "MockStreamGenerator",
        "AbstractStreamDF",
        "FardalStreamDF",
    }
