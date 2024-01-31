"""Testing :mod:`galax.dynamics.mockstream` module."""

from galax.dynamics import mockstream
from galax.dynamics._dynamics.mockstream import core, df, mockstream_generator


def test_all() -> None:
    """Test the `galax.dynamics.mockstream` API."""
    assert set(mockstream.__all__) == set(
        df.__all__ + core.__all__ + mockstream_generator.__all__
    )
