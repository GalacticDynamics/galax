"""Testing :mod:`galax.dynamics` module."""

import galax.coordinates as gc


def test_all() -> None:
    """Test the `galax.coordinates` API."""
    assert set(gc.__all__) == {
        "ops",
        "frames",
        # Base
        "AbstractPhaseSpaceObject",
        # Coordinates
        "AbstractBasicPhaseSpaceCoordinate",
        "AbstractPhaseSpaceCoordinate",
        "PhaseSpaceCoordinate",
        "AbstractCompositePhaseSpaceCoordinate",
        "CompositePhaseSpaceCoordinate",
        "ComponentShapeTuple",
        # PSPs
        "PhaseSpacePosition",
        "PSPComponentShapeTuple",
        # Protocols
        "PhaseSpaceObjectInterpolant",
    }
