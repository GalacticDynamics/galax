"""Testing :mod:`galax.dynamics.mockstream` module."""

from galax.dynamics import mockstream


def test_all():
    """Test the `galax.dynamics.mockstream` API."""
    assert set(mockstream.__all__) == set(
        mockstream._df.__all__ + mockstream._mockstream_generator.__all__
    )
