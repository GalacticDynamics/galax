"""Testing :mod:`galax.dynamics.actionangle` module."""

from galax.dynamics import actionangle


def test_all() -> None:
    """Test the `galax.dynamics.actionangle` API."""
    assert set(actionangle.__all__) == {
        "ActionsAnglesFrequencies",
        "isochrone_actions_angles_freqs",
    }
