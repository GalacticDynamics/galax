"""Testing :mod:`galax.dynamics.mockstream` module."""

from galax.dynamics import mockstream
from galax.dynamics._src.mockstream import arm, core, df, mockstream_generator


def test_all() -> None:
    """Test the `galax.dynamics.mockstream` API."""
    assert set(mockstream.__all__) == set(
        df.__all__ + arm.__all__ + core.__all__ + mockstream_generator.__all__
    )
